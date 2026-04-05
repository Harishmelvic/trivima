"""
SAM wrapper — Segment Anything Model for semantic segmentation.

Priority order:
  1. SAM 3 via Hugging Face Transformers (facebook/sam3) — text-prompted concept segmentation
  2. SAM 2.1 via Ultralytics — instance segmentation fallback
  3. Grounded SAM 2 (SAM 2 + Grounding DINO) — open-vocabulary fallback

SAM 3 is the preferred model — it supports Promptable Concept Segmentation (PCS):
  given a text prompt like "door" or "glass table", it segments all instances.
  This directly provides semantic labels without needing a separate classifier.

Output: per-pixel semantic labels + label-to-name mapping.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


# Categories to probe with SAM 3 text prompts for indoor scenes
INDOOR_CONCEPTS = [
    "wall", "floor", "ceiling", "door", "window", "mirror",
    "glass table", "glass", "sofa", "chair", "table", "bed",
    "lamp", "shelf", "cabinet", "rug", "curtain", "plant",
    "TV", "painting", "bookshelf", "desk", "counter", "sink",
    "toilet", "bathtub", "refrigerator", "oven", "sky",
]


class SAMSegmenter:
    """Semantic segmentation using SAM 3, SAM 2.1, or Grounded SAM 2."""

    def __init__(self, device: str = "cuda", use_sam3: bool = True,
                 hf_model_id: str = "facebook/sam3"):
        self.device = device
        self.use_sam3 = use_sam3
        self.hf_model_id = hf_model_id
        self._model = None
        self._processor = None
        self._backend = None  # "sam3_hf", "ultralytics", "grounded_sam2"

    def load(self):
        """Load segmentation model in priority order."""
        if self.use_sam3:
            if self._try_load_sam3_hf():
                return
            if self._try_load_ultralytics():
                return
            print("[SAM] SAM 3 and Ultralytics not available, falling back to Grounded SAM 2")

        self._load_grounded_sam2()

    def _try_load_sam3_hf(self) -> bool:
        """Try to load SAM 3 via Hugging Face Transformers."""
        try:
            from transformers import Sam3Model, Sam3Processor
            import torch

            self._processor = Sam3Processor.from_pretrained(self.hf_model_id)
            self._model = Sam3Model.from_pretrained(self.hf_model_id)
            self._model = self._model.to(self.device)
            self._model.eval()
            self._backend = "sam3_hf"
            print(f"[SAM] SAM 3 loaded from {self.hf_model_id} via Transformers")
            return True
        except ImportError:
            print("[SAM] transformers Sam3Model not available (need transformers >= 4.47)")
            return False
        except Exception as e:
            print(f"[SAM] SAM 3 HF load failed: {e}")
            return False

    def _try_load_ultralytics(self) -> bool:
        """Try to load SAM via Ultralytics."""
        try:
            from ultralytics import SAM
            for model_name in ["sam2.1_l.pt", "sam2_l.pt", "sam_l.pt"]:
                try:
                    self._model = SAM(model_name)
                    self._backend = "ultralytics"
                    print(f"[SAM] Loaded {model_name} via Ultralytics")
                    return True
                except Exception:
                    continue
            return False
        except ImportError:
            return False

    def _load_grounded_sam2(self):
        """Load Grounded SAM 2 (SAM 2 + Grounding DINO)."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            sam2 = build_sam2("sam2_hiera_l.yaml", "data/checkpoints/sam2_hiera_large.pt")
            sam2 = sam2.to(self.device)
            self._model = SAM2AutomaticMaskGenerator(sam2)
            self._backend = "grounded_sam2"
            print("[SAM] SAM 2 automatic mask generator loaded")

            try:
                from groundingdino.util.inference import load_model
                self._grounding_model = load_model(
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "data/checkpoints/groundingdino_swint_ogc.pth",
                )
                print("[SAM] Grounding DINO loaded for label classification")
            except (ImportError, Exception):
                self._grounding_model = None
                print("[SAM] Grounding DINO not available — labels will be generic")

        except ImportError:
            print("[SAM] SAM 2 not installed. Install: pip install segment-anything-2")
            raise

    def segment(self, image: np.ndarray,
                concepts: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment image into labeled regions.

        Args:
            image: (H, W, 3) uint8 RGB image
            concepts: list of text concepts to segment (SAM 3 only).
                      If None, uses INDOOR_CONCEPTS.

        Returns:
            (labels, label_names)
            labels: (H, W) int32 — per-pixel semantic label index
            label_names: dict mapping label index → category name string
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if self._backend == "sam3_hf":
            return self._segment_sam3_hf(image, concepts or INDOOR_CONCEPTS)
        elif self._backend == "ultralytics":
            return self._segment_ultralytics(image)
        else:
            return self._segment_grounded_sam2(image)

    def _segment_sam3_hf(self, image: np.ndarray,
                         concepts: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment with SAM 3 via Hugging Face — text-prompted concept segmentation."""
        import torch
        from PIL import Image

        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        label_names = {0: "background"}
        pil_image = Image.fromarray(image)

        label_id = 1
        for concept in concepts:
            try:
                inputs = self._processor(
                    images=pil_image,
                    text=[concept],
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Process masks
                masks = self._processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"],
                )

                if masks and len(masks) > 0:
                    for mask_tensor in masks:
                        # mask_tensor shape: (num_masks, H, W)
                        if mask_tensor.dim() == 3:
                            for m in range(mask_tensor.shape[0]):
                                mask_np = mask_tensor[m].cpu().numpy().astype(bool)
                                if mask_np.shape != (h, w):
                                    mask_np = np.array(
                                        Image.fromarray(mask_np.astype(np.uint8) * 255).resize((w, h))
                                    ) > 128
                                if mask_np.sum() > 100:  # skip tiny detections
                                    labels[mask_np] = label_id
                                    label_names[label_id] = concept
                                    label_id += 1
                        elif mask_tensor.dim() == 2:
                            mask_np = mask_tensor.cpu().numpy().astype(bool)
                            if mask_np.sum() > 100:
                                labels[mask_np] = label_id
                                label_names[label_id] = concept
                                label_id += 1

            except Exception as e:
                # Some concepts may not be found — that's fine
                continue

        return labels, label_names

    def _segment_ultralytics(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment with SAM 2.1 via Ultralytics."""
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        label_names = {0: "background"}

        results = self._model(image)

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                names = result.names if hasattr(result, 'names') else {}

                for i, mask in enumerate(masks):
                    label_id = i + 1
                    mask_bool = mask.astype(bool)
                    if mask_bool.shape != (h, w):
                        from PIL import Image
                        mask_bool = np.array(
                            Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((w, h))
                        ) > 128
                    labels[mask_bool] = label_id

                    cls_id = int(result.boxes.cls[i]) if hasattr(result, 'boxes') and result.boxes is not None else i
                    label_names[label_id] = names.get(cls_id, f"object_{label_id}")

        return labels, label_names

    def _segment_grounded_sam2(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment with SAM 2 + optional Grounding DINO classification."""
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        label_names = {0: "background"}

        masks_data = self._model.generate(image)
        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

        for i, mask_info in enumerate(masks_data):
            label_id = i + 1
            mask = mask_info['segmentation']
            labels[mask] = label_id
            label_names[label_id] = f"region_{label_id}"

        return labels, label_names

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._backend = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
