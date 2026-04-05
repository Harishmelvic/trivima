"""
Qwen3-VL — VLM for design intelligence.

From vlm_architecture_theory.md (v2 — Qwen3-VL):
  - Qwen3-VL-8B-Instruct as primary model (~16GB fp16)
  - Native 3D grounding via Interleaved-MRoPE (no custom 3D-RoPE hack)
  - Prompt-based 3D context from cell grid (distances, surfaces, positions)
  - SpatialVLM distillation loaded via LoRA adapters
  - Thinking variant available for complex design reasoning

Model selection by use case:
  - Qwen3-VL-8B-Instruct: fast re-ranking, distillation testing
  - Qwen3-VL-8B-Thinking: auto-furnishing planning
  - Qwen3-VL-30B-A3B-Instruct: production quality (30B params, 3B active)

Never in the render loop — called at decision points only.
"""

import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import re


class SpatialContextBuilder:
    """Builds prompt-based 3D context from the cell grid for VLM queries.

    Instead of injecting 3D coordinates into the model's positional encoding
    (the old 3D-RoPE approach), we provide explicit spatial data in the prompt.
    Qwen3-VL reasons about this data using its native 3D understanding.

    This is simpler (no model modification), more portable (works with any VLM),
    and more robust (uses the VLM's trained reasoning path).
    """

    def __init__(self, cell_size: float = 0.05):
        self.cell_size = cell_size

    def build_room_context(self, grid_data: dict, label_names: Dict[int, str]) -> str:
        """Build a text description of the room from the cell grid.

        Used for auto-furnishing and environment classification prompts.
        """
        # Room dimensions from cell bounds
        keys = list(grid_data.keys())
        if not keys:
            return "Empty scene."

        xs = [k[0] for k in keys]
        ys = [k[1] for k in keys]
        zs = [k[2] for k in keys]
        width = (max(xs) - min(xs) + 1) * self.cell_size
        height = (max(ys) - min(ys) + 1) * self.cell_size
        depth = (max(zs) - min(zs) + 1) * self.cell_size

        # Existing furniture from semantic labels
        from collections import Counter
        label_counts = Counter()
        for cell in grid_data.values():
            label_idx = cell.get("label", 0)
            name = label_names.get(label_idx, "").lower()
            if name and name not in ("background", "floor", "wall", "ceiling", ""):
                label_counts[name] += 1

        furniture = [f"{name} ({count} cells)" for name, count in label_counts.most_common(10)]
        furniture_str = ", ".join(furniture) if furniture else "none detected"

        return (
            f"Room dimensions: {width:.1f}m × {depth:.1f}m × {height:.1f}m.\n"
            f"Existing furniture: {furniture_str}.\n"
            f"Total cells: {len(grid_data):,}."
        )

    def build_candidate_context(
        self,
        candidates: List[Dict],
        category: str = "",
    ) -> str:
        """Build text description of placement candidates for re-ranking.

        Each candidate includes its 3D position and validation scores.
        """
        lines = []
        for i, c in enumerate(candidates):
            desc = c.get('description', '')
            if not desc:
                desc = "({:.1f}, {:.1f}, {:.1f})".format(
                    c.get('x', 0), c.get('y', 0), c.get('z', 0)
                )

            parts = [f"Position {i+1}: {desc}"]
            if 'validation_score' in c:
                parts.append(f"validation={c['validation_score']:.2f}")
            if 'clearance' in c:
                parts.append(f"clearance={c['clearance']:.2f}m")
            if 'surface_type' in c:
                parts.append(f"surface={c['surface_type']}")

            lines.append(", ".join(parts))

        return "\n".join(lines)


class QwenVLM:
    """Qwen3-VL for design intelligence.

    Provides:
      - Room description and classification
      - Aesthetic re-ranking of placement candidates
      - Auto-furnishing gap detection
      - Object style matching

    Uses native Qwen3-VL spatial encoding — no custom 3D-RoPE injection.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        lora_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.lora_path = lora_path

        self._model = None
        self._processor = None
        self._context_builder = SpatialContextBuilder()

    def load(self):
        """Load Qwen3-VL model."""
        import torch

        try:
            # Try Qwen3-VL first (transformers >= 4.57)
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
                from transformers import AutoProcessor
                print(f"[VLM] Loading {self.model_id} via Qwen2_5_VLForConditionalGeneration...")
            except ImportError:
                from transformers import AutoModelForCausalLM as QwenVLModel
                from transformers import AutoProcessor
                print(f"[VLM] Loading {self.model_id} via AutoModelForCausalLM...")

            # Disable cuDNN if it causes issues (common on cloud GPUs)
            if torch.cuda.is_available():
                try:
                    conv_test = torch.nn.Conv2d(1, 1, 1).cuda()
                    conv_test(torch.randn(1, 1, 1, 1).cuda())
                    del conv_test
                except RuntimeError:
                    torch.backends.cudnn.enabled = False
                    print("[VLM] cuDNN disabled (driver mismatch)")

            self._model = QwenVLModel.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )

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
            print(f"[VLM] Loaded. GPU memory: {gpu_mem:.1f} GB")

        except ImportError as e:
            print(f"[VLM] transformers not available: {e}")
            raise

    def query(self, image: np.ndarray, prompt: str,
              max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Send a query to the VLM with an image.

        Args:
            image: (H, W, 3) uint8 RGB
            prompt: text query
            max_tokens: max response length
            temperature: sampling temperature

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

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[pil_image],
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        input_len = inputs["input_ids"].shape[1]
        response = self._processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
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
            Candidates sorted by VLM score (best first)
        """
        if mode == "fast":
            return self._score_fast(image, candidates, category)
        else:
            return self._score_full(image, candidates, category)

    def _score_fast(self, image, candidates, category):
        """Fast scoring — single forward pass, ranking only."""
        context = self._context_builder.build_candidate_context(candidates, category)

        prompt = (
            f"Rate each position for placing a {category} in this room. "
            f"Reply with ONLY the position numbers ranked best to worst.\n\n"
            f"{context}"
        )

        response = self.query(image, prompt, max_tokens=100, temperature=0.0)
        ranked = self._parse_ranking(response, len(candidates))

        for i, c in enumerate(candidates):
            rank = ranked.index(i) if i in ranked else len(candidates)
            c["vlm_score"] = 1.0 - rank / max(len(candidates), 1)
            c["vlm_rank"] = rank

        return sorted(candidates, key=lambda c: c.get("vlm_score", 0), reverse=True)

    def _score_full(self, image, candidates, category):
        """Full generative scoring with explanations."""
        top_n = min(5, len(candidates))
        top = sorted(candidates, key=lambda c: c.get("validation_score", 0), reverse=True)[:top_n]

        context = self._context_builder.build_candidate_context(top, category)

        prompt = (
            f"Rank these {top_n} positions for placing a {category}. "
            f"For the top choice, explain why it's best.\n\n"
            f"{context}"
        )

        response = self.query(image, prompt, max_tokens=300, temperature=0.1)

        for i, c in enumerate(top):
            c["vlm_explanation"] = response
            c["vlm_score"] = 1.0 - i / top_n

        return top

    def _parse_ranking(self, response: str, n: int) -> List[int]:
        """Parse position numbers from VLM response."""
        numbers = re.findall(r'\d+', response)
        ranked = []
        for num_str in numbers:
            idx = int(num_str) - 1
            if 0 <= idx < n and idx not in ranked:
                ranked.append(idx)
        for i in range(n):
            if i not in ranked:
                ranked.append(i)
        return ranked

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
