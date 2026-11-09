"""
finetune.py — LoRA fine-tune on winning Frankl proof traces.

Loads dataset/train.jsonl produced by build_dataset.py and trains a small
causal LM to predict (element, rule, reason) from family descriptions.

Default base model: Qwen/Qwen2.5-1.5B-Instruct (small, free, no auth needed)
Adapter saved to:   checkpoints/frankl-lora/

Usage:
    python3 finetune.py
    python3 finetune.py --model Qwen/Qwen2.5-0.5B-Instruct   # even smaller
    python3 finetune.py --epochs 5 --lr 2e-4

Requirements:
    pip install transformers peft datasets torch accelerate
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH  = "dataset/train.jsonl"
OUTPUT_DIR    = "checkpoints/frankl-lora"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sft_dataset(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return Dataset.from_list(records)


def tokenize_example(example, tokenizer, max_length):
    """
    Concatenate prompt + completion as one sequence.
    Labels are -100 on the prompt tokens (we only supervise the completion).
    """
    prompt_text     = example["prompt"]
    completion_text = example["completion"] + tokenizer.eos_token

    prompt_ids     = tokenizer(prompt_text,     add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels    = ([-100] * len(prompt_ids) + completion_ids)[:max_length]

    # Pad to max_length
    pad_len   = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
    labels    = labels    + [-100]                   * pad_len
    attn_mask = [1] * (max_length - pad_len) + [0] * pad_len

    return {
        "input_ids":      input_ids,
        "attention_mask": attn_mask,
        "labels":         labels,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL,
                        help="HuggingFace model ID (causal LM)")
    parser.add_argument("--epochs",  type=int,   default=3)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--batch",   type=int,   default=4,
                        help="Per-device train batch size")
    parser.add_argument("--max-len", type=int,   default=512,
                        help="Max token length per example")
    parser.add_argument("--lora-r",  type=int,   default=8,
                        help="LoRA rank")
    args = parser.parse_args()

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run build_dataset.py first.")
        return

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float32   # MPS has limited fp16 op support
    else:
        device = "cpu"
        dtype  = torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model + LoRA ──────────────────────────────────────────────────────────
    print(f"Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 4,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw = load_sft_dataset(DATASET_PATH)
    print(f"\nDataset: {len(raw)} examples")

    tokenized = raw.map(
        lambda ex: tokenize_example(ex, tokenizer, args.max_len),
        remove_columns=raw.column_names,
        desc="Tokenising",
    )

    n_eval = min(50, max(1, len(tokenized) // 10))
    split  = tokenized.train_test_split(test_size=n_eval, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # ── Training args ─────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=max(1, 16 // args.batch),
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,          # keep off for MPS/CPU compatibility
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nFine-tuning with LoRA (r={args.lora_r}, α={args.lora_r*4})...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print("\nTo use the fine-tuned model:")
    print("  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
