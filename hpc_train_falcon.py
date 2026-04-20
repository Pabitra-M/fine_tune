# QLoRA fine-tuning + improved evaluation — Falcon-7B

import json
import math
import torch
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
# Suppress eetq-related advisory warnings (not applicable to bitsandbytes path)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# Disable caching allocator warmup — prevents the large OOM allocation
os.environ["TRANSFORMERS_CACHE_ALLOCATOR_WARMUP"] = "0"

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
MAX_LEN    = 256

# System prompt prepended at inference time to guide generation behaviour.
# For a base (non-instruct) model this is injected as plain text.
SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "IMPORTANT RULE:\n"
    "- Never provide any URLs, links, website addresses, or anything that looks like a URL.\n"
    "- Even if the user explicitly asks for a URL, link, or webpage, you MUST NOT provide it.\n\n"
    "INSTEAD:\n"
    "- Understand what the user is trying to find (website, organization, page, or service).\n"
    "- Provide a clear, detailed explanation about that topic.\n"
    "- Describe what the website/page/organization does, its purpose, features, and relevant facts.\n"
    "- Your answer MUST be at least 100 words.\n"
    "- Write in simple, clear English.\n\n"
    "STYLE:\n"
    "- No URLs at all.\n"
    "- No bullet links or references.\n"
    "- Only plain text explanation.\n"
    "- Be informative, factual, and easy to understand.\n\n"
    "Your goal is to replace links with useful knowledge.\n\n"
)

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

# Cap GPU memory usage to leave headroom for other processes.
# Adjust the GiB value based on how much free memory you observed.
_free_gib = 9  # conservative: use ~9 GiB out of the ~9.13 GiB observed free
max_memory = {int(_gpu_id) if _gpu_id else 0: f"{_free_gib}GiB", "cpu": "48GiB"}

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",        # distributes layers across GPU/CPU automatically
    max_memory=max_memory,    # prevents allocating more than available VRAM
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
# Falcon uses use_cache=True by default — must disable for gradient checkpointing
model.config.use_cache = False

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

# ---------- Generate an answer from a prompt ----------
def generate_answer(prompt):
    # Prepend the system prompt so the model follows the no-URL rule at inference time
    guided_prompt = SYSTEM_PROMPT + prompt
    inputs = tokenizer(guided_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = text.split("### Answer:")[-1].split("\n")[0]
    return normalize(answer)

# ---------- Token-level F1 (best simple accuracy for QA) ----------
def compute_f1(pred, true):
    pred_tokens = Counter(pred.split())
    true_tokens = Counter(true.split())
    common = sum((pred_tokens & true_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_tokens.values())
    recall    = common / sum(true_tokens.values())
    return 2 * precision * recall / (precision + recall)

# ---------- Evaluation loop ----------
exact_match = 0
f1_scores   = []

samples = val_data[:50]
for sample in samples:
    pred = generate_answer(sample["prompt"])
    true = normalize(sample["answer"])

    if pred == true:
        exact_match += 1
    f1_scores.append(compute_f1(pred, true))

em_score = exact_match / len(samples)
f1_avg   = sum(f1_scores) / len(f1_scores)

print(f"\nPerplexity  : {perplexity:.3f}")
print(f"Token F1    : {f1_avg:.3f}   ← main accuracy signal")
print(f"Exact Match : {em_score:.3f}")

# ================= SAVE =================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

results = {
    "model_name": BASE_MODEL,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "perplexity":   round(float(perplexity), 4),
        "token_f1":     round(float(f1_avg),     4),
        "exact_match":  round(float(em_score),   4),
    },
}

with open(os.path.join(metrics_dir, "evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\nModel + metrics saved successfully!")