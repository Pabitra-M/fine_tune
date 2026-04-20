# QLoRA fine-tuning + improved evaluation — Falcon-7B

import json
import math
import torch
import numpy as np
import sacrebleu
import subprocess
import re
from collections import Counter
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split

# ================= GPU SETUP =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _pick_best_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True
        )
        rows = [(int(r.split(",")[0]), int(r.split(",")[1].strip()))
                for r in out.strip().splitlines()]
        best = max(rows, key=lambda x: x[1])
        print(f"[GPU] Using GPU {best[0]} ({best[1]} MiB free)")
        return str(best[0])
    except Exception:
        return None

_gpu_id = _pick_best_gpu()
if _gpu_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id

# ================= CONFIG =================
BASE_MODEL = "tiiuae/falcon-7b"
OUTPUT_DIR = "model/falcon-7b_qa_model"
DATA_PATH  = "clean_dataset.json"
MAX_LEN    = 256   # Increased (important)

# ================= TEXT NORMALIZATION =================
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())

# ================= DATA LOAD =================
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        q = normalize(item.get("question", ""))
        a = normalize(item.get("answer", ""))

        if not q or not a:
            continue

        prompt = f"### Question:\n{q}\n\n### Answer:\n"
        full   = prompt + a

        records.append({
            "prompt": prompt,
            "full": full,
            "answer": a
        })

    return records

records = load_data(DATA_PATH)
train_data, val_data = train_test_split(records, test_size=0.1, random_state=42)

train_ds = Dataset.from_list(train_data)
val_ds   = Dataset.from_list(val_data)

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize(example):
    tokens = tokenizer(
        example["full"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

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

# ================= MODEL =================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# ================= TRAINING =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=1,
    logging_steps=10,
    report_to="none",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

print("Training...")
trainer.train()

# ================= EVALUATION =================
print("\nEvaluating...")

eval_loss = trainer.evaluate()["eval_loss"]
perplexity = math.exp(eval_loss)

# ---------- Generation ----------
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = text.split("### Answer:")[-1]
    answer = answer.split("\n")[0]

    return normalize(answer)

# ---------- Metrics ----------
def token_overlap_accuracy(pred_tokens, true_tokens):
    overlap = set(pred_tokens) & set(true_tokens)
    return len(overlap) / max(len(true_tokens), 1)

def compute_f1(pred_tokens, true_tokens):
    pred_counter = Counter(pred_tokens)
    true_counter = Counter(true_tokens)

    common = pred_counter & true_counter
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)

    return 2 * precision * recall / (precision + recall)

# ---------- Loop ----------
exact_match = 0
token_accs = []
f1_scores = []
all_preds = []
all_refs = []

for sample in val_data[:50]:
    pred = generate_answer(sample["prompt"])
    true = normalize(sample["answer"])

    pred_tokens = pred.split()
    true_tokens = true.split()

    if pred == true:
        exact_match += 1

    token_accs.append(token_overlap_accuracy(pred_tokens, true_tokens))
    f1_scores.append(compute_f1(pred_tokens, true_tokens))

    all_preds.append(pred)
    all_refs.append(true)

# ---------- Final Scores ----------
em_score = exact_match / len(val_data[:50])
token_acc = float(np.mean(token_accs))
f1_avg = float(np.mean(f1_scores))
bleu_avg = sacrebleu.corpus_bleu(all_preds, [all_refs]).score / 100

print(f"\nPerplexity           : {perplexity:.3f}")
print(f"Exact Match          : {em_score:.3f}")
print(f"Token Overlap Acc    : {token_acc:.3f}")
print(f"F1 Score             : {f1_avg:.3f}")
print(f"BLEU Score           : {bleu_avg:.3f}")

# ================= SAVE =================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

results = {
    "model_name": BASE_MODEL,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "perplexity": float(perplexity),
        "exact_match": float(em_score),
        "token_accuracy": token_acc,
        "f1_score": f1_avg,
        "bleu_score": bleu_avg,
    },
}

with open(os.path.join(metrics_dir, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nModel + metrics saved successfully!")