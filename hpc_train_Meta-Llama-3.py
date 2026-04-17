#QLoRA fine-tuning + evaluation

import json
import math
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
import os
from datetime import datetime
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import sacrebleu


# CONFIG
BASE_MODEL = "NousResearch/Meta-Llama-3-8B"
OUTPUT_DIR = "model/Llama3_qa_model"
DATA_PATH  = "all_dataset.json"
MAX_LEN    = 256


# LOAD DATA
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()
        if not q or not a:
            continue
        prompt = f"### Question:\n{q}\n\n### Answer:\n"
        full   = prompt + a
        records.append({"prompt": prompt, "full": full, "answer": a})
    return records


records = load_data(DATA_PATH)
train_data, val_data = train_test_split(records, test_size=0.1, random_state=42)
train_ds = Dataset.from_list(train_data)
val_ds   = Dataset.from_list(val_data)


# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    tokens = tokenizer(
        example["full"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

    # mask prompt tokens with -100 so loss is only computed on the answer
    prompt_tokens = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=MAX_LEN,
    )
    prompt_len = len(prompt_tokens["input_ids"])

    labels = tokens["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    labels = [
        -100 if token_id == tokenizer.pad_token_id else label
        for token_id, label in zip(tokens["input_ids"], labels)
    ]
    tokens["labels"] = labels
    return tokens


train_ds = train_ds.map(tokenize)
val_ds   = val_ds.map(tokenize)


# MODEL
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


# TRAINING
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("Training started...")
trainer.train()


# EVALUATION
print("\nRunning evaluation...")

eval_loss  = trainer.evaluate()["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"\nPerplexity: {perplexity:.2f}")


# GENERATION-BASED METRICS
def generate_answer(prompt):
    inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    text    = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("### Answer:")[-1].strip()


def compute_f1(pred_tokens, true_tokens):
    """Token-overlap F1 (same as SQuAD metric)."""
    pred_set = set(pred_tokens)
    true_set = set(true_tokens)
    common   = pred_set & true_set
    if not common:
        return 0.0
    precision = len(common) / len(pred_set)
    recall    = len(common) / len(true_set)
    return 2 * precision * recall / (precision + recall)


exact_match = 0
token_accs  = []
f1_scores   = []
all_preds   = []
all_refs    = []

for sample in val_data[:50]:
    pred        = generate_answer(sample["prompt"])
    true        = sample["answer"]
    pred_tokens = pred.split()
    true_tokens = true.split()

    # Exact match
    if pred.strip() == true.strip():
        exact_match += 1

    # Token accuracy (positional)
    match = sum(p == t for p, t in zip(pred_tokens, true_tokens))
    token_accs.append(match / max(len(true_tokens), 1))

    # F1
    f1_scores.append(compute_f1(pred_tokens, true_tokens))

    # Collect for corpus-level BLEU
    all_preds.append(pred)
    all_refs.append(true)


em_score  = exact_match / len(val_data[:50])
token_acc = float(np.mean(token_accs))
f1_avg    = float(np.mean(f1_scores))

# Corpus-level BLEU via sacrebleu (normalized to 0-1)
bleu_avg = sacrebleu.corpus_bleu(all_preds, [all_refs]).score / 100

print(f"\n✅ Exact Match Accuracy : {em_score:.3f}")
print(f"✅ Token Accuracy        : {token_acc:.3f}")
print(f"✅ F1 Score              : {f1_avg:.3f}")
print(f"✅ BLEU Score            : {bleu_avg:.3f}")


# SAVE
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

results = {
    "model_name": BASE_MODEL,
    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "perplexity":     float(perplexity),
        "exact_match":    float(em_score),
        "token_accuracy": token_acc,
        "f1_score":       f1_avg,
        "bleu_score":     bleu_avg,
    },
}

metrics_file = os.path.join(metrics_dir, "evaluation_results.json")
with open(metrics_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nMetrics saved at: {metrics_file}")
print("Model saved!")