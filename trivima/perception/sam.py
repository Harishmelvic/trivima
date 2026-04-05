"""
SAM wrapper — Segment Anything Model for semantic segmentation.

Preferred: SAM 3 (text-prompted, 270K+ concepts)
Fallback: Grounded SAM 2 (SAM 2 + Grounding DINO) for open-vocabulary detection
Avoid: CLIP-per-mask (too slow, adds complexity)

Output: per-pixel semantic labels + label-to-name mapping.
The label names are used by failure_modes.py to detect glass/mirror/sky/etc.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List


class SAMSegmenter:
    """Semantic segmentation using SAM 3 or Grounded SAM 2."""

    def __init__(self, device: str = "cuda", use_sam3: bool = True):
        self.device = device
        self.use_sam3 = use_sam3
        self._model = None
        self._grounding_model = None

    def load(self):
        """Load segmentation model."""
        if self.use_sam3:
            if self._try_load_sam3():
                return
            print("[SAM] SAM 3 not available, falling back to Grounded SAM 2")

        self._load_grounded_sam2()

    def _try_load_sam3(self) -> bool:
        """Try to load SAM 3."""
        try:
            from ultralytics import SAM
            self._model = SAM("sam3_l.pt")  # or latest SAM 3 checkpoint
            print("[SAM] SAM 3 loaded")
            return True
        except (ImportError, Exception) as e:
            print(f"[SAM] SAM 3 not available: {e}")
            return False

    def _load_grounded_sam2(self):
        """Load Grounded SAM 2 (SAM 2 + Grounding DINO)."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            sam2 = build_sam2("sam2_hiera_l.yaml", "data/checkpoints/sam2_hiera_large.pt")
            sam2 = sam2.to(self.device)
            self._model = SAM2AutomaticMaskGenerator(sam2)
            print("[SAM] SAM 2 automatic mask generator loaded")

            # Try to load Grounding DINO for classification
            try:
                from groundingdino.util.inference import load_model
                self._grounding_model = load_model(
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "data/checkpoints/groundingdino_swint_ogc.pth",
                )
                print("[SAM] Grounding DINO loaded for label classification")
            except (ImportError, Exception):
                print("[SAM] Grounding DINO not available — labels will be generic")
                self._grounding_model = None

        except ImportError:
            print("[SAM] SAM 2 not installed.")
            print("  Install: pip install segment-anything-2")
            raise

    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment image into labeled regions.

        Args:
            image: (H, W, 3) uint8 RGB image

        Returns:
            (labels, label_names)
            labels: (H, W) int32 — per-pixel semantic label index
            label_names: dict mapping label index → category name string
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        if self.use_sam3 and hasattr(self._model, 'predict'):
            return self._segment_sam3(image)
        else:
            return self._segment_grounded_sam2(image)

    def _segment_sam3(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment with SAM 3 (text-prompted, returns semantic labels)."""
        results = self._model(image)
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        label_names = {0: "background"}

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
                        mask_resized = np.array(Image.fromarray(mask_bool.astype(np.uint8) * 255).resize((w, h))) > 128
                        mask_bool = mask_resized
                    labels[mask_bool] = label_id

                    cls_id = int(result.boxes.cls[i]) if hasattr(result, 'boxes') else i
                    label_names[label_id] = names.get(cls_id, f"object_{label_id}")

        return labels, label_names

    def _segment_grounded_sam2(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[int, str]]:
        """Segment with SAM 2 + optional Grounding DINO classification."""
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        label_names = {0: "background"}

        # Generate masks with SAM 2
        masks_data = self._model.generate(image)

        # Sort by area (largest first) so smaller objects overwrite larger ones
        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

        for i, mask_info in enumerate(masks_data):
            label_id = i + 1
            mask = mask_info['segmentation']
            labels[mask] = label_id

            # Classify with Grounding DINO if available
            if self._grounding_model is not None:
                name = self._classify_mask_region(image, mask)
            else:
                name = f"region_{label_id}"

            label_names[label_id] = name

        return labels, label_names

    def _classify_mask_region(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Classify a masked region using Grounding DINO."""
        # Common indoor categories to probe
        categories = [
            "wall", "floor", "ceiling", "door", "window", "mirror",
            "glass table", "sofa", "chair", "table", "bed", "lamp",
            "shelf", "cabinet", "rug", "curtain", "plant", "TV",
            "painting", "bookshelf", "desk", "counter", "sink",
            "toilet", "bathtub", "refrigerator", "oven", "sky",
        ]

        try:
            from groundingdino.util.inference import predict
            import torch
            from PIL import Image

            # Crop to mask bounding box for faster classification
            ys, xs = np.where(mask)
            if len(ys) == 0:
                return "unknown"
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            crop = image[y0:y1+1, x0:x1+1]

            pil_crop = Image.fromarray(crop)
            text_prompt = ". ".join(categories) + "."

            boxes, logits, phrases = predict(
                model=self._grounding_model,
                image=pil_crop,
                caption=text_prompt,
                box_threshold=0.25,
                text_threshold=0.2,
            )

            if len(phrases) > 0:
                return phrases[0].strip()

        except Exception:
            pass

        return "object"

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._grounding_model is not None:
            del self._grounding_model
            self._grounding_model = None
        import torch
        torch.cuda.empty_cache()
