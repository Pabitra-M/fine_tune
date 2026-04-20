#QLoRA fine-tuning + evaluation

import json
import math
import torch
import numpy as np
import sacrebleu
import subprocess
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

# ── Memory environment setup (must happen before any CUDA calls) ──────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Auto-select the GPU with the most free memory so we avoid saturated GPUs
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
        print(f"[GPU] Auto-selected GPU {best[0]} ({best[1]} MiB free)")
        return str(best[0])
    except Exception:
        return None  # fall back to default

_gpu_id = _pick_best_gpu()
if _gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_id
# ─────────────────────────────────────────────────────────────────────────────





# CONFIG
BASE_MODEL = "NousResearch/Llama-2-7b-chat-hf"
OUTPUT_DIR = "model/Llama-2_qa_model"
DATA_PATH  = "clean_dataset.json"
MAX_LEN    = 128   # Reduced from 256 → halves per-sample activation memory


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
tokenizer.padding_side = "right"   # required for causal LM fine-tuning


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
    bnb_4bit_use_double_quant=True,
)

# Do NOT pass device_map here — bitsandbytes manages GPU placement internally.
# Passing device_map triggers dispatch_model → .to() which is illegal for 4/8-bit models.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
)

# Prepare model for k-bit training (freezes base weights, enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)

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
    per_device_train_batch_size=1,        # Reduced from 2 to 1 to save memory
    gradient_accumulation_steps=4,        # Increased from 2 to 4 to maintain effective batch size
    learning_rate=2e-4,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,  
    per_device_eval_batch_size=1,         # Reduced to match train
    logging_steps=10,
    report_to="none",
    gradient_checkpointing=True,          # MASSIVE memory saving during training
    optim="paged_adamw_8bit",             # Uses 8-bit optimizer states to save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Flush any cached allocations before training begins
torch.cuda.empty_cache()
print(f"[GPU] Memory before training: "
      f"{torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
      f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

print("Training started...")
trainer.train()


# EVALUATION
print("\nRunning evaluation...")

eval_loss  = trainer.evaluate()["eval_loss"]
perplexity = math.exp(eval_loss)
print(f"\nPerplexity: {perplexity:.2f}")


# GENERATION-BASED METRICS
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

def generate_answer(prompt):
    guided_prompt = SYSTEM_PROMPT + prompt          # ← only change: prepend system prompt
    inputs  = tokenizer(guided_prompt, return_tensors="pt").to(model.device)
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