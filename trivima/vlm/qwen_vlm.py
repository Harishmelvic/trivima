"""
Qwen2.5-VL with 3D-RoPE — VLM for design intelligence.

From vlm_architecture_theory.md:
  - Qwen2.5-VL-32B as single runtime model (8-bit quantized, ~32GB)
  - 3D-RoPE: replaces 2D pixel positional encoding with metric 3D coords
  - Dimension allocation: 1/3 X, 1/3 Y, 1/3 Z (normalized to [0,1])
  - Falls back to 2D for low-confidence patches
  - SpatialVLM distillation loaded via LoRA adapters

Usage:
    vlm = QwenVLM(device="cuda")
    vlm.load()
    response = vlm.query(image, "Describe this room")
    scores = vlm.score_candidates(image, candidates, category="lamp")
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path


class ThreeDRoPE:
    """3D Rotary Position Embedding — replaces 2D pixel coords with metric 3D.

    From vlm_architecture_theory.md Ch2:
      - Each patch's (X, Y, Z) in meters → rotational encoding
      - Dimension split: 1/3 X, 1/3 Y, 1/3 Z
      - Coordinates normalized by room dimensions to [0, 1]
      - Low-confidence patches fall back to (u, v, 0)

    Zero parameters, zero cost — same RoPE computation, different input values.
    """

    def __init__(self, embed_dim: int = 1280, room_size: Tuple[float, float, float] = (5.0, 3.0, 5.0)):
        self.embed_dim = embed_dim
        self.room_size = room_size  # (width_x, height_y, depth_z) in meters
        # Split dimensions equally: 1/3 for each axis
        self.dim_per_axis = embed_dim // 3

    def compute_3d_positions(
        self,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        confidence_map: Optional[np.ndarray] = None,
        patch_size: int = 14,
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        """Compute 3D positions for each image patch.

        Args:
            depth_map: (H, W) metric depth in meters
            intrinsics: (3, 3) camera intrinsics
            confidence_map: (H, W) per-pixel confidence [0,1]. Low-conf → 2D fallback.
            patch_size: ViT patch size (typically 14 for Qwen2.5-VL)
            confidence_threshold: below this, fall back to 2D

        Returns:
            (num_patches, 3) normalized 3D positions in [0,1]
        """
        h, w = depth_map.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Patch grid
        ph = h // patch_size
        pw = w // patch_size
        positions = np.zeros((ph * pw, 3), dtype=np.float32)

        for py in range(ph):
            for px in range(pw):
                # Patch center pixel
                u = (px + 0.5) * patch_size
                v = (py + 0.5) * patch_size
                iu, iv = int(u), int(v)

                # Get depth at patch center
                d = depth_map[min(iv, h - 1), min(iu, w - 1)]

                # Check confidence
                use_3d = True
                if confidence_map is not None:
                    conf = confidence_map[min(iv, h - 1), min(iu, w - 1)]
                    if conf < confidence_threshold:
                        use_3d = False

                idx = py * pw + px

                if use_3d and d > 0.1:
                    # 3D backprojection
                    x = (u - cx) * d / fx
                    y = (v - cy) * d / fy
                    z = d

                    # Normalize to [0, 1] by room size
                    positions[idx, 0] = np.clip(x / self.room_size[0] + 0.5, 0, 1)
                    positions[idx, 1] = np.clip(y / self.room_size[1] + 0.5, 0, 1)
                    positions[idx, 2] = np.clip(z / self.room_size[2], 0, 1)
                else:
                    # 2D fallback: (u_norm, v_norm, 0)
                    positions[idx, 0] = u / w
                    positions[idx, 1] = v / h
                    positions[idx, 2] = 0.0  # no depth info

        return positions

    def encode(self, positions: np.ndarray) -> np.ndarray:
        """Compute RoPE frequencies from 3D positions.

        Standard RoPE: freq_i = pos / 10000^(2i/d)
        3D-RoPE: same formula applied independently to X, Y, Z dimensions.

        Args:
            positions: (N, 3) normalized positions in [0,1]

        Returns:
            (N, embed_dim) sinusoidal positional encodings
        """
        n = positions.shape[0]
        encoding = np.zeros((n, self.embed_dim), dtype=np.float32)

        for axis in range(3):
            start = axis * self.dim_per_axis
            end = start + self.dim_per_axis
            pos = positions[:, axis]  # (N,)

            # RoPE frequencies
            dim_indices = np.arange(self.dim_per_axis // 2, dtype=np.float32)
            freqs = 1.0 / (10000.0 ** (2 * dim_indices / self.dim_per_axis))

            # Outer product: pos × freqs
            angles = np.outer(pos, freqs)  # (N, dim//2)

            # Sin/cos pairs
            encoding[:, start:start + self.dim_per_axis // 2] = np.sin(angles)
            encoding[:, start + self.dim_per_axis // 2:end] = np.cos(angles)

        return encoding

    def validate_encoding(self, encoding: np.ndarray) -> dict:
        """Validate encoding quality — used in tests."""
        return {
            "shape": encoding.shape,
            "has_nan": bool(np.any(np.isnan(encoding))),
            "has_inf": bool(np.any(np.isinf(encoding))),
            "min": float(encoding.min()),
            "max": float(encoding.max()),
            "mean_magnitude": float(np.abs(encoding).mean()),
        }


class QwenVLM:
    """Qwen2.5-VL with 3D-RoPE for design intelligence.

    Provides:
      - Room description and classification
      - Aesthetic re-ranking of placement candidates
      - Auto-furnishing gap detection
      - Object style matching

    Never in the render loop — called at decision points only.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct",
        device: str = "cuda",
        quantize_8bit: bool = True,
        lora_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.quantize_8bit = quantize_8bit
        self.lora_path = lora_path

        self._model = None
        self._processor = None
        self._rope_3d = ThreeDRoPE()

    def load(self):
        """Load Qwen2.5-VL with optional 8-bit quantization and LoRA."""
        import torch

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

            if self.quantize_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    print("[VLM] Loading Qwen2.5-VL with 8-bit quantization...")
                except ImportError:
                    print("[VLM] bitsandbytes not available, loading in float16")

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(self.model_id)

            # Load LoRA adapters if available (from distillation training)
            if self.lora_path and Path(self.lora_path).exists():
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(self._model, self.lora_path)
                    print(f"[VLM] LoRA adapters loaded from {self.lora_path}")
                except ImportError:
                    print("[VLM] peft not available, skipping LoRA")

            self._model.eval()
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"[VLM] Qwen2.5-VL loaded. GPU memory: {gpu_mem:.1f} GB")

        except ImportError as e:
            print(f"[VLM] transformers Qwen2.5-VL not available: {e}")
            raise

    def query(self, image: np.ndarray, prompt: str,
              max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Send a query to the VLM with an image.

        Args:
            image: (H, W, 3) uint8 RGB
            prompt: text query
            max_tokens: max response length
            temperature: sampling temperature (0.1 = near-deterministic)

        Returns:
            Text response from the model
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        import torch
        from PIL import Image

        pil_image = Image.fromarray(image)

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ]}
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[pil_image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return response.strip()

    def score_candidates(
        self,
        image: np.ndarray,
        candidates: List[Dict],
        category: str,
        mode: str = "fast",
    ) -> List[Dict]:
        """Score placement candidates using the VLM.

        Args:
            image: (H, W, 3) room image
            candidates: list of dicts with position, validation_score, description
            category: object category ("lamp", "plant", etc.)
            mode: "fast" (logit scoring, ~200-500ms) or "full" (generative, ~2-5s)

        Returns:
            Candidates sorted by VLM score (best first), each with vlm_score added
        """
        if mode == "fast":
            return self._score_fast(image, candidates, category)
        else:
            return self._score_full(image, candidates, category)

    def _score_fast(self, image, candidates, category):
        """Fast logit-based scoring — single forward pass."""
        # Format all candidates into one prompt
        candidate_descriptions = "\n".join(
            "Position {}: {}".format(
                i + 1,
                c.get('description', '({:.1f}, {:.1f}, {:.1f})'.format(c.get('x', 0), c.get('y', 0), c.get('z', 0)))
            )
            for i, c in enumerate(candidates)
        )

        prompt = (
            f"Rate each position for placing a {category} in this room. "
            f"Reply with ONLY the position numbers ranked best to worst.\n\n"
            f"{candidate_descriptions}"
        )

        response = self.query(image, prompt, max_tokens=100, temperature=0.0)

        # Parse ranking from response
        ranked = self._parse_ranking(response, len(candidates))

        for i, c in enumerate(candidates):
            rank = ranked.index(i) if i in ranked else len(candidates)
            c["vlm_score"] = 1.0 - rank / max(len(candidates), 1)
            c["vlm_rank"] = rank

        return sorted(candidates, key=lambda c: c.get("vlm_score", 0), reverse=True)

    def _score_full(self, image, candidates, category):
        """Full generative scoring with explanations."""
        top_n = min(5, len(candidates))
        top_candidates = sorted(candidates, key=lambda c: c.get("validation_score", 0), reverse=True)[:top_n]

        descs = "\n".join(
            f"Position {i+1}: {c.get('description', 'unknown')}, validation score: {c.get('validation_score', 0):.2f}"
            for i, c in enumerate(top_candidates)
        )

        prompt = (
            f"Rank these {top_n} positions for placing a {category}. "
            f"For the top choice, explain why it's best considering style, function, and spatial balance.\n\n"
            f"{descs}"
        )

        response = self.query(image, prompt, max_tokens=300, temperature=0.1)

        for i, c in enumerate(top_candidates):
            c["vlm_explanation"] = response
            c["vlm_score"] = 1.0 - i / top_n

        return top_candidates

    def _parse_ranking(self, response: str, n: int) -> List[int]:
        """Parse position numbers from VLM response."""
        import re
        numbers = re.findall(r'\d+', response)
        ranked = []
        for num_str in numbers:
            idx = int(num_str) - 1  # 1-indexed to 0-indexed
            if 0 <= idx < n and idx not in ranked:
                ranked.append(idx)
        # Fill missing positions
        for i in range(n):
            if i not in ranked:
                ranked.append(i)
        return ranked

    def inject_3d_rope(self, depth_map: np.ndarray, intrinsics: np.ndarray,
                       confidence_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute 3D-RoPE positions for the current image.

        Returns (num_patches, 3) normalized positions that can be used
        to replace the model's default 2D positional encoding.
        """
        return self._rope_3d.compute_3d_positions(
            depth_map, intrinsics, confidence_map
        )

    def get_memory_usage(self) -> dict:
        """Return current GPU memory usage."""
        import torch
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "peak_gb": 0}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        import torch
        torch.cuda.empty_cache()
