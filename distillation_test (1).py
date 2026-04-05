#!/usr/bin/env python3
"""
Trivima — Small-Scale Distillation Test
========================================
Validates the SpatialVLM → Qwen distillation pipeline before committing
to the full 7-12 day training run.

Hardware: 1× A40 48GB (or any 48GB+ GPU)
Time: ~3-4 hours total
Cost: ~$5-10 at RunPod rates

Usage:
    # Full pipeline (data gen → baseline → train → eval)
    python distillation_test.py --image_dir data/sample_images/ --run all

    # Individual steps
    python distillation_test.py --image_dir data/sample_images/ --run generate
    python distillation_test.py --run baseline
    python distillation_test.py --run train
    python distillation_test.py --run evaluate
    python distillation_test.py --run canary
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import torch


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """All configurable parameters in one place."""
    
    # Model
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "float16"
    
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # Training
    num_epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 2
    
    # Data
    image_dir: str = "data/sample_images/"
    output_dir: str = "distillation_test/"
    qa_file: str = "distillation_test/spatial_qa.json"
    holdout_file: str = "distillation_test/spatial_qa_holdout.json"
    results_file: str = "distillation_test/results.json"
    holdout_ratio: float = 0.1  # 10% held out for evaluation
    
    # Evaluation
    num_eval_samples: int = 50
    spatial_error_threshold: float = 0.25  # 25% — pass if below this
    aesthetic_degradation_threshold: float = 0.15  # 15% max degradation
    perplexity_increase_threshold: float = 0.15  # 15% max increase
    
    # 3D-RoPE (set to True to test with 3D positional encoding)
    use_3d_rope: bool = False  # Start without, compare later
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))


# ============================================================================
# Step 1: Generate Spatial QA Data
# ============================================================================

def generate_spatial_qa(config: Config):
    """
    Generate spatial QA pairs from existing cell grids.
    Uses Depth Pro depth maps + SAM 3 segmentation to compute ground truth
    distances and heights. No SpatialVLM needed.
    """
    print("\n" + "=" * 60)
    print("STEP 1: Generating Spatial QA Data")
    print("=" * 60)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    image_dir = Path(config.image_dir)
    image_files = sorted(
        [f for f in image_dir.iterdir() 
         if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    )
    
    if not image_files:
        print(f"No images found in {config.image_dir}")
        print("Creating synthetic spatial QA data instead...")
        return _generate_synthetic_qa(config)
    
    print(f"Found {len(image_files)} images")
    
    # Try to import Trivima perception pipeline
    try:
        from trivima.perception.pipeline import PerceptionPipeline
        from trivima.construction.point_to_grid import PointToGrid
        pipeline = PerceptionPipeline()
        grid_builder = PointToGrid(cell_size=0.05)
        use_trivima = True
        print("Using Trivima perception pipeline for QA generation")
    except ImportError:
        use_trivima = False
        print("Trivima not available — using synthetic QA generation")
        return _generate_synthetic_qa(config)
    
    all_pairs = []
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        try:
            # Run perception
            result = pipeline.run(str(img_path))
            grid = grid_builder.convert(result)
            
            # Find object clusters by semantic label
            objects = _find_object_clusters(grid)
            print(f"  Found {len(objects)} objects")
            
            if len(objects) < 2:
                print(f"  Skipping — too few objects for spatial QA")
                continue
            
            # Generate distance QA pairs
            for i, obj_a in enumerate(objects):
                for j, obj_b in enumerate(objects):
                    if i >= j:
                        continue
                    
                    dist = np.linalg.norm(
                        np.array(obj_a["centroid"]) - np.array(obj_b["centroid"])
                    )
                    
                    if dist < 0.1 or dist > 15.0:
                        continue  # Skip implausible distances
                    
                    all_pairs.append({
                        "image": str(img_path),
                        "question": f"How far is the {obj_a['label']} from the {obj_b['label']}?",
                        "answer": f"{dist:.2f} meters",
                        "numeric_answer": round(dist, 2),
                        "type": "distance"
                    })
                
                # Height QA
                height = obj_a["centroid"][1]
                if 0 < height < 4.0:
                    all_pairs.append({
                        "image": str(img_path),
                        "question": f"What is the height of the {obj_a['label']}?",
                        "answer": f"{height:.2f} meters",
                        "numeric_answer": round(height, 2),
                        "type": "height"
                    })
            
            # Unload models between images to free memory
            pipeline.unload()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_pairs:
        print("No QA pairs generated from images. Falling back to synthetic.")
        return _generate_synthetic_qa(config)
    
    return _save_qa_pairs(all_pairs, config)


def _find_object_clusters(grid):
    """Group cells by semantic label and compute centroids."""
    from collections import defaultdict
    
    label_cells = defaultdict(list)
    
    for idx in range(grid.count):
        label = grid.get_semantic_label(idx)
        pos = grid.get_position(idx)
        if label and label not in ("floor", "ceiling", "wall", "unknown"):
            label_cells[label].append(pos)
    
    objects = []
    for label, positions in label_cells.items():
        positions = np.array(positions)
        centroid = positions.mean(axis=0)
        objects.append({
            "label": label,
            "centroid": centroid.tolist(),
            "cell_count": len(positions)
        })
    
    return objects


def _generate_synthetic_qa(config: Config):
    """
    Generate synthetic spatial QA when no images or perception pipeline available.
    Creates realistic room scenarios with known ground truth.
    """
    print("Generating synthetic spatial QA data...")
    
    rooms = [
        {
            "name": "living_room_1",
            "objects": [
                {"label": "sofa", "pos": [2.0, 0.45, 3.0]},
                {"label": "coffee table", "pos": [2.0, 0.40, 1.8]},
                {"label": "TV", "pos": [2.0, 1.20, 0.3]},
                {"label": "floor lamp", "pos": [0.5, 1.50, 3.0]},
                {"label": "bookshelf", "pos": [4.5, 1.00, 2.5]},
                {"label": "plant", "pos": [0.3, 0.60, 0.5]},
                {"label": "rug", "pos": [2.0, 0.01, 2.0]},
                {"label": "window", "pos": [0.0, 1.50, 2.5]},
            ]
        },
        {
            "name": "bedroom_1",
            "objects": [
                {"label": "bed", "pos": [2.5, 0.50, 2.0]},
                {"label": "nightstand", "pos": [1.0, 0.55, 2.0]},
                {"label": "nightstand", "pos": [4.0, 0.55, 2.0]},
                {"label": "dresser", "pos": [2.5, 0.80, 4.5]},
                {"label": "desk", "pos": [0.5, 0.75, 4.0]},
                {"label": "desk chair", "pos": [0.5, 0.45, 3.5]},
                {"label": "lamp", "pos": [1.0, 0.85, 2.0]},
                {"label": "mirror", "pos": [4.8, 1.50, 3.0]},
            ]
        },
        {
            "name": "kitchen_1",
            "objects": [
                {"label": "counter", "pos": [1.0, 0.90, 0.5]},
                {"label": "dining table", "pos": [3.0, 0.75, 3.0]},
                {"label": "dining chair", "pos": [2.5, 0.45, 3.5]},
                {"label": "dining chair", "pos": [3.5, 0.45, 3.5]},
                {"label": "refrigerator", "pos": [0.3, 1.00, 0.3]},
                {"label": "sink", "pos": [1.0, 0.85, 0.3]},
                {"label": "window", "pos": [0.0, 1.50, 2.0]},
            ]
        },
        {
            "name": "office_1",
            "objects": [
                {"label": "desk", "pos": [2.5, 0.75, 1.0]},
                {"label": "office chair", "pos": [2.5, 0.50, 1.8]},
                {"label": "monitor", "pos": [2.5, 1.10, 0.8]},
                {"label": "bookshelf", "pos": [4.5, 1.20, 2.5]},
                {"label": "filing cabinet", "pos": [0.3, 0.70, 3.0]},
                {"label": "plant", "pos": [4.8, 0.40, 0.3]},
                {"label": "lamp", "pos": [2.0, 1.00, 0.8]},
                {"label": "window", "pos": [0.0, 1.50, 1.5]},
            ]
        },
        {
            "name": "living_room_2",
            "objects": [
                {"label": "sofa", "pos": [1.5, 0.45, 4.0]},
                {"label": "armchair", "pos": [0.5, 0.45, 2.5]},
                {"label": "coffee table", "pos": [1.5, 0.40, 2.8]},
                {"label": "side table", "pos": [3.0, 0.55, 4.0]},
                {"label": "TV stand", "pos": [1.5, 0.50, 0.3]},
                {"label": "floor lamp", "pos": [3.2, 1.60, 4.0]},
                {"label": "plant", "pos": [3.5, 0.80, 0.3]},
                {"label": "window", "pos": [0.0, 1.50, 3.0]},
                {"label": "door", "pos": [5.0, 1.00, 2.5]},
            ]
        },
    ]
    
    all_pairs = []
    
    for room in rooms:
        objects = room["objects"]
        
        # Add random noise to create variants (10 variants per room)
        for variant in range(10):
            noisy_objects = []
            for obj in objects:
                noisy_pos = [
                    obj["pos"][0] + np.random.normal(0, 0.15),
                    obj["pos"][1] + np.random.normal(0, 0.05),
                    obj["pos"][2] + np.random.normal(0, 0.15),
                ]
                noisy_objects.append({"label": obj["label"], "pos": noisy_pos})
            
            # Distance pairs
            for i, obj_a in enumerate(noisy_objects):
                for j, obj_b in enumerate(noisy_objects):
                    if i >= j:
                        continue
                    
                    dist = np.sqrt(sum(
                        (a - b) ** 2 for a, b in zip(obj_a["pos"], obj_b["pos"])
                    ))
                    
                    if dist < 0.1 or dist > 15.0:
                        continue
                    
                    # Use room name as image placeholder
                    all_pairs.append({
                        "image": f"synthetic/{room['name']}_v{variant}.jpg",
                        "question": f"How far is the {obj_a['label']} from the {obj_b['label']}?",
                        "answer": f"{dist:.2f} meters",
                        "numeric_answer": round(dist, 2),
                        "type": "distance"
                    })
                
                # Height
                height = max(0.01, obj_a["pos"][1])
                all_pairs.append({
                    "image": f"synthetic/{room['name']}_v{variant}.jpg",
                    "question": f"What is the height of the {obj_a['label']}?",
                    "answer": f"{height:.2f} meters",
                    "numeric_answer": round(height, 2),
                    "type": "height"
                })
            
            # Spatial relationship pairs
            for i, obj_a in enumerate(noisy_objects):
                for j, obj_b in enumerate(noisy_objects):
                    if i >= j:
                        continue
                    
                    dx = obj_b["pos"][0] - obj_a["pos"][0]
                    dz = obj_b["pos"][2] - obj_a["pos"][2]
                    
                    if abs(dx) > abs(dz):
                        rel = "to the right of" if dx > 0 else "to the left of"
                    else:
                        rel = "in front of" if dz < 0 else "behind"
                    
                    all_pairs.append({
                        "image": f"synthetic/{room['name']}_v{variant}.jpg",
                        "question": f"Is the {obj_b['label']} to the left, right, in front, or behind the {obj_a['label']}?",
                        "answer": rel,
                        "numeric_answer": None,
                        "type": "spatial_relation"
                    })
    
    return _save_qa_pairs(all_pairs, config)


def _save_qa_pairs(all_pairs: list, config: Config):
    """Split into train/holdout and save."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    np.random.shuffle(all_pairs)
    
    holdout_count = max(50, int(len(all_pairs) * config.holdout_ratio))
    holdout = all_pairs[:holdout_count]
    train = all_pairs[holdout_count:]
    
    with open(config.qa_file, "w") as f:
        json.dump(train, f, indent=2)
    
    with open(config.holdout_file, "w") as f:
        json.dump(holdout, f, indent=2)
    
    print(f"\nGenerated {len(all_pairs)} total QA pairs")
    print(f"  Training: {len(train)}")
    print(f"  Holdout:  {len(holdout)}")
    print(f"  Types: {_count_types(all_pairs)}")
    print(f"  Saved to: {config.qa_file}")
    
    return train, holdout


def _count_types(pairs):
    from collections import Counter
    return dict(Counter(p["type"] for p in pairs))


# ============================================================================
# Step 2: Baseline Evaluation
# ============================================================================

def run_baseline(config: Config):
    """
    Evaluate base Qwen (no training) on spatial questions.
    This is the "before" measurement.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Baseline Evaluation (Before Distillation)")
    print("=" * 60)
    
    model, processor = _load_qwen(config)
    
    with open(config.holdout_file) as f:
        holdout = json.load(f)
    
    # Filter to numeric questions only (distance + height)
    numeric_qs = [q for q in holdout if q["numeric_answer"] is not None]
    eval_qs = numeric_qs[:config.num_eval_samples]
    
    print(f"Evaluating on {len(eval_qs)} numeric spatial questions...")
    
    results = _evaluate_spatial(model, processor, eval_qs)
    
    # Also run aesthetic canary
    aesthetic_score = _evaluate_aesthetic_canary(model, processor)
    
    # Save baseline
    baseline = {
        "spatial_error": results["mean_error"],
        "spatial_median_error": results["median_error"],
        "correct_within_25pct": results["correct_within_25pct"],
        "correct_within_50pct": results["correct_within_50pct"],
        "aesthetic_score": aesthetic_score,
        "num_evaluated": len(eval_qs),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    baseline_path = os.path.join(config.output_dir, "baseline.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    
    print(f"\nBaseline Results:")
    print(f"  Spatial mean error:    {results['mean_error']:.1%}")
    print(f"  Spatial median error:  {results['median_error']:.1%}")
    print(f"  Within 25% accuracy:   {results['correct_within_25pct']:.1%}")
    print(f"  Within 50% accuracy:   {results['correct_within_50pct']:.1%}")
    print(f"  Aesthetic canary:      {aesthetic_score:.2f}")
    print(f"  Saved to: {baseline_path}")
    
    _cleanup_model(model, processor)
    return baseline


def _evaluate_spatial(model, processor, questions):
    """Run spatial evaluation and compute error metrics."""
    errors = []
    details = []
    
    for i, qa in enumerate(questions):
        try:
            prompt = (
                f"Answer with ONLY a number in meters, nothing else.\n"
                f"Question: {qa['question']}"
            )
            
            response = _generate(model, processor, prompt, max_tokens=20)
            predicted = _extract_number(response)
            actual = qa["numeric_answer"]
            
            if predicted is not None and actual > 0:
                error = abs(predicted - actual) / actual
                errors.append(error)
                details.append({
                    "question": qa["question"],
                    "actual": actual,
                    "predicted": predicted,
                    "error": error
                })
            
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(questions)}...")
                
        except Exception as e:
            print(f"  Error on question {i}: {e}")
            continue
    
    if not errors:
        return {"mean_error": 1.0, "median_error": 1.0, 
                "correct_within_25pct": 0.0, "correct_within_50pct": 0.0,
                "details": []}
    
    errors = np.array(errors)
    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "correct_within_25pct": float(np.mean(errors < 0.25)),
        "correct_within_50pct": float(np.mean(errors < 0.50)),
        "details": details
    }


def _evaluate_aesthetic_canary(model, processor):
    """
    Quick aesthetic evaluation — can the model still judge room quality?
    Returns a coherence score (0-1).
    """
    prompts = [
        "A living room has a sofa against the east wall, a coffee table centered in front of it, "
        "and a floor lamp in the corner. On a scale of 1-10, how well-arranged is this room? "
        "Answer with just the number.",
        
        "A bedroom has the bed blocking the doorway, a desk facing the wall, and clothes on the floor. "
        "On a scale of 1-10, how well-arranged is this room? Answer with just the number.",
        
        "An office has a desk by the window for natural light, a bookshelf against the back wall, "
        "and a comfortable chair. On a scale of 1-10, how well-arranged is this room? "
        "Answer with just the number.",
    ]
    
    expected_order = [7, 3, 8]  # good, bad, good
    scores = []
    
    for prompt in prompts:
        try:
            response = _generate(model, processor, prompt, max_tokens=10)
            score = _extract_number(response)
            if score is not None:
                scores.append(min(10, max(1, score)))
            else:
                scores.append(5)  # neutral if can't parse
        except Exception:
            scores.append(5)
    
    # Check if the model gets the ordering right (good > bad)
    if len(scores) >= 3:
        ordering_correct = (scores[0] > scores[1]) and (scores[2] > scores[1])
        return 1.0 if ordering_correct else 0.5
    
    return 0.5


# ============================================================================
# Step 3: LoRA Training
# ============================================================================

def run_training(config: Config):
    """
    Fine-tune Qwen with LoRA on spatial QA data.
    """
    print("\n" + "=" * 60)
    print("STEP 3: LoRA Training")
    print("=" * 60)
    
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import TrainingArguments, Trainer
    
    # Load model
    model, processor = _load_qwen(config)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Load training data
    with open(config.qa_file) as f:
        train_data = json.load(f)
    
    print(f"Training on {len(train_data)} QA pairs")
    
    # Create dataset
    dataset = SpatialQADataset(train_data, processor, config.max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "checkpoints"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    
    # Custom data collator
    def collate_fn(examples):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [ex["input_ids"] for ex in examples],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [ex["attention_mask"] for ex in examples],
            batch_first=True,
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [ex["labels"] for ex in examples],
            batch_first=True,
            padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    
    print("\nStarting training...")
    start_time = time.time()
    
    train_result = trainer.train()
    
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    
    # Save the LoRA adapter
    adapter_path = os.path.join(config.output_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    print(f"  LoRA adapter saved to: {adapter_path}")
    
    # Save training metadata
    meta = {
        "training_loss": train_result.training_loss,
        "training_time_minutes": elapsed / 60,
        "num_samples": len(train_data),
        "epochs": config.num_epochs,
        "lora_rank": config.lora_rank,
        "learning_rate": config.learning_rate,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(config.output_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    _cleanup_model(model, processor)
    return meta


class SpatialQADataset(torch.utils.data.Dataset):
    """Dataset for spatial QA fine-tuning."""
    
    def __init__(self, qa_pairs, processor, max_length):
        self.qa_pairs = qa_pairs
        self.processor = processor
        self.max_length = max_length
        self.tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        qa = self.qa_pairs[idx]
        
        # Format as instruction
        text = (
            f"<|im_start|>system\n"
            f"You are a spatial reasoning assistant. Answer questions about "
            f"distances and positions precisely in meters.<|im_end|>\n"
            f"<|im_start|>user\n{qa['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n{qa['answer']}<|im_end|>"
        )
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding=False,
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Labels: mask the prompt, only compute loss on the answer
        labels = input_ids.clone()
        
        # Find where the assistant answer starts
        answer_text = f"<|im_start|>assistant\n{qa['answer']}"
        answer_tokens = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        
        # Mask everything before the answer with -100
        answer_start = len(input_ids) - len(answer_tokens)
        if answer_start > 0:
            labels[:answer_start] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ============================================================================
# Step 4: Post-Training Evaluation
# ============================================================================

def run_evaluation(config: Config):
    """
    Evaluate the trained model on the same holdout set.
    Compare against baseline.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Post-Training Evaluation")
    print("=" * 60)
    
    from peft import PeftModel
    
    # Load baseline
    baseline_path = os.path.join(config.output_dir, "baseline.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    # Load trained model
    model, processor = _load_qwen(config)
    adapter_path = os.path.join(config.output_dir, "lora_adapter")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("Loaded LoRA adapter")
    
    # Load holdout
    with open(config.holdout_file) as f:
        holdout = json.load(f)
    
    numeric_qs = [q for q in holdout if q["numeric_answer"] is not None]
    eval_qs = numeric_qs[:config.num_eval_samples]
    
    print(f"Evaluating on {len(eval_qs)} numeric spatial questions...")
    
    results = _evaluate_spatial(model, processor, eval_qs)
    
    # Compare
    improvement = (baseline["spatial_error"] - results["mean_error"]) / baseline["spatial_error"]
    
    print(f"\n{'=' * 50}")
    print(f"COMPARISON:")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<30} {'Baseline':>10} {'Trained':>10} {'Change':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Mean spatial error':<30} {baseline['spatial_error']:>9.1%} {results['mean_error']:>9.1%} {improvement:>+9.1%}")
    print(f"  {'Median spatial error':<30} {baseline['spatial_median_error']:>9.1%} {results['median_error']:>9.1%}")
    print(f"  {'Within 25% accuracy':<30} {baseline['correct_within_25pct']:>9.1%} {results['correct_within_25pct']:>9.1%}")
    print(f"  {'Within 50% accuracy':<30} {baseline['correct_within_50pct']:>9.1%} {results['correct_within_50pct']:>9.1%}")
    
    # Verdict
    passed = results["mean_error"] < config.spatial_error_threshold
    improved = results["mean_error"] < baseline["spatial_error"]
    
    print(f"\n  SPATIAL TEST: {'PASS' if passed else 'FAIL'} (threshold: {config.spatial_error_threshold:.0%})")
    print(f"  IMPROVED:     {'YES' if improved else 'NO'} ({improvement:+.1%} change)")
    
    # Save results
    eval_results = {
        "baseline": baseline,
        "trained": {
            "spatial_error": results["mean_error"],
            "spatial_median_error": results["median_error"],
            "correct_within_25pct": results["correct_within_25pct"],
            "correct_within_50pct": results["correct_within_50pct"],
        },
        "improvement": improvement,
        "passed_threshold": passed,
        "improved_over_baseline": improved,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(config.results_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    
    _cleanup_model(model, processor)
    return eval_results


# ============================================================================
# Step 5: Canary Checks
# ============================================================================

def run_canary(config: Config):
    """
    Verify that training didn't destroy existing capabilities.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Canary Checks")
    print("=" * 60)
    
    from peft import PeftModel
    
    # Load baseline
    baseline_path = os.path.join(config.output_dir, "baseline.json")
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    # Load trained model
    model, processor = _load_qwen(config)
    adapter_path = os.path.join(config.output_dir, "lora_adapter")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    # Aesthetic canary
    print("\nAesthetic canary...")
    aesthetic_score = _evaluate_aesthetic_canary(model, processor)
    aesthetic_degradation = 1.0 - (aesthetic_score / max(baseline["aesthetic_score"], 0.01))
    aesthetic_pass = aesthetic_degradation < config.aesthetic_degradation_threshold
    
    print(f"  Baseline aesthetic: {baseline['aesthetic_score']:.2f}")
    print(f"  Trained aesthetic:  {aesthetic_score:.2f}")
    print(f"  Degradation:        {aesthetic_degradation:.1%}")
    print(f"  AESTHETIC CANARY:   {'PASS' if aesthetic_pass else 'FAIL'}")
    
    # Language canary — generate explanations and check coherence
    print("\nLanguage canary...")
    language_prompts = [
        "Explain why a floor lamp should be placed near a sofa in a living room.",
        "Describe three reasons why a bookshelf works well against a wall.",
        "Why is it important to keep pathways clear in a room?"
    ]
    
    language_ok = True
    for prompt in language_prompts:
        response = _generate(model, processor, prompt, max_tokens=100)
        # Basic coherence checks
        if len(response) < 20:
            print(f"  WARNING: Short response ({len(response)} chars): {response[:50]}")
            language_ok = False
        elif response.count("the") < 1 and response.count("a ") < 1:
            print(f"  WARNING: Possibly garbled: {response[:50]}")
            language_ok = False
        else:
            print(f"  OK: {response[:80]}...")
    
    print(f"  LANGUAGE CANARY:    {'PASS' if language_ok else 'FAIL'}")
    
    # Summary
    print(f"\n{'=' * 50}")
    print(f"CANARY SUMMARY:")
    print(f"  Aesthetic: {'PASS' if aesthetic_pass else 'FAIL'}")
    print(f"  Language:  {'PASS' if language_ok else 'FAIL'}")
    print(f"  Overall:   {'PASS' if (aesthetic_pass and language_ok) else 'FAIL'}")
    
    canary_results = {
        "aesthetic_baseline": baseline["aesthetic_score"],
        "aesthetic_trained": aesthetic_score,
        "aesthetic_degradation": aesthetic_degradation,
        "aesthetic_pass": aesthetic_pass,
        "language_pass": language_ok,
        "overall_pass": aesthetic_pass and language_ok,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    canary_path = os.path.join(config.output_dir, "canary_results.json")
    with open(canary_path, "w") as f:
        json.dump(canary_results, f, indent=2)
    
    _cleanup_model(model, processor)
    return canary_results


# ============================================================================
# Model Utilities
# ============================================================================

def _load_qwen(config: Config):
    """Load Qwen2.5-VL model and processor."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    
    print(f"Loading {config.model_name}...")
    
    dtype = getattr(torch, config.torch_dtype)
    
    # Try loading as VL model — supports both Qwen2.5-VL and Qwen3-VL
    try:
        # Qwen3-VL: try the latest class first
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            config.model_name, 
            trust_remote_code=True
        )
        print(f"  Loaded via AutoModelForImageTextToText")
    except Exception as e1:
        try:
            # Fallback: Qwen2.5-VL specific class
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            processor = AutoProcessor.from_pretrained(
                config.model_name, 
                trust_remote_code=True
            )
            print(f"  Loaded via Qwen2_5_VLForConditionalGeneration")
        except Exception as e2:
            print(f"  VL models not available, loading as causal LM...")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            processor = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
    
    model.eval()
    
    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Loaded. GPU memory: {mem_gb:.1f} GB")
    
    return model, processor


def _generate(model, processor, prompt: str, max_tokens: int = 50) -> str:
    """Generate text from the model."""
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    
    messages = [
        {"role": "system", "content": "You are a spatial reasoning assistant. Answer precisely."},
        {"role": "user", "content": prompt}
    ]
    
    # Try chat template, fall back to raw
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"System: You are a spatial reasoning assistant.\nUser: {prompt}\nAssistant:"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response


def _extract_number(text: str) -> Optional[float]:
    """Extract the first number from a text response."""
    if not text:
        return None
    
    # Try to find a decimal number
    matches = re.findall(r'[-+]?\d*\.?\d+', text)
    if matches:
        try:
            val = float(matches[0])
            if 0 < val < 100:  # sanity check for meters
                return val
        except ValueError:
            pass
    
    return None


def _cleanup_model(model, processor):
    """Free GPU memory."""
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# ============================================================================
# Final Report
# ============================================================================

def print_final_report(config: Config):
    """Print a summary of all results."""
    print("\n" + "=" * 60)
    print("FINAL REPORT — Small-Scale Distillation Test")
    print("=" * 60)
    
    # Load all results
    results = {}
    for name in ["baseline", "results", "training_meta", "canary_results"]:
        path = os.path.join(config.output_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    
    if "baseline" in results:
        b = results["baseline"]
        print(f"\nBaseline (before training):")
        print(f"  Spatial error:  {b['spatial_error']:.1%}")
        print(f"  Aesthetic:      {b['aesthetic_score']:.2f}")
    
    if "training_meta" in results:
        t = results["training_meta"]
        print(f"\nTraining:")
        print(f"  Final loss:     {t['training_loss']:.4f}")
        print(f"  Time:           {t['training_time_minutes']:.1f} minutes")
        print(f"  Samples:        {t['num_samples']}")
    
    if "results" in results:
        r = results["results"]
        print(f"\nSpatial Accuracy:")
        print(f"  Baseline error: {r['baseline']['spatial_error']:.1%}")
        print(f"  Trained error:  {r['trained']['spatial_error']:.1%}")
        print(f"  Improvement:    {r['improvement']:+.1%}")
        print(f"  Passed:         {'YES' if r['passed_threshold'] else 'NO'}")
    
    if "canary_results" in results:
        c = results["canary_results"]
        print(f"\nCanary Checks:")
        print(f"  Aesthetic:      {'PASS' if c['aesthetic_pass'] else 'FAIL'}")
        print(f"  Language:       {'PASS' if c['language_pass'] else 'FAIL'}")
    
    # Overall verdict
    all_pass = True
    if "results" in results:
        all_pass = all_pass and results["results"].get("improved_over_baseline", False)
    if "canary_results" in results:
        all_pass = all_pass and results["canary_results"].get("overall_pass", False)
    
    print(f"\n{'=' * 50}")
    print(f"VERDICT: {'DISTILLATION WORKS — proceed to full training' if all_pass else 'NEEDS INVESTIGATION — check errors above'}")
    print(f"{'=' * 50}")
    
    if all_pass:
        print(f"\nNext steps:")
        print(f"  1. Scale up to Qwen2.5-VL-32B")
        print(f"  2. Generate 50-100K QA pairs with SpatialVLM")
        print(f"  3. Train with LoRA rank 32 for 3-5 days on 4×A100")
        print(f"  4. Run full Stage 4 test suite (29 tests)")
    else:
        print(f"\nDebug steps:")
        print(f"  1. Check training loss curve — did it converge?")
        print(f"  2. Check QA data quality — are the ground truth answers correct?")
        print(f"  3. Try lower learning rate (1e-4) or higher LoRA rank (16)")
        print(f"  4. Try more training data (increase room variants)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trivima Distillation Test")
    parser.add_argument("--image_dir", type=str, default="data/sample_images/",
                        help="Directory with room images")
    parser.add_argument("--output_dir", type=str, default="distillation_test/",
                        help="Output directory for results")
    parser.add_argument("--run", type=str, default="all",
                        choices=["all", "generate", "baseline", "train", "evaluate", "canary", "report"],
                        help="Which step to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Model name/path (default: Qwen3-VL-8B-Instruct)")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    config = Config(
        model_name=args.model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        qa_file=os.path.join(args.output_dir, "spatial_qa.json"),
        holdout_file=os.path.join(args.output_dir, "spatial_qa_holdout.json"),
        results_file=os.path.join(args.output_dir, "results.json"),
    )
    
    os.makedirs(config.output_dir, exist_ok=True)
    config.save(os.path.join(config.output_dir, "config.json"))
    
    print(f"Trivima Distillation Test")
    print(f"  Model:      {config.model_name}")
    print(f"  Output:     {config.output_dir}")
    print(f"  LoRA rank:  {config.lora_rank}")
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  LR:         {config.learning_rate}")
    
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name()}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
    else:
        print("  WARNING: No GPU detected — training will be very slow")
    
    steps = {
        "generate": generate_spatial_qa,
        "baseline": run_baseline,
        "train": run_training,
        "evaluate": run_evaluation,
        "canary": run_canary,
        "report": lambda c: print_final_report(c),
    }
    
    if args.run == "all":
        for step_name, step_fn in steps.items():
            step_fn(config)
    else:
        steps[args.run](config)


if __name__ == "__main__":
    main()
