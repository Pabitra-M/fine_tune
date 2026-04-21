#QLoRA fine-tuning + evaluation

import json
import math
import re
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

# Minimum free VRAM required before the script is allowed to start.
# Llama-2-7B in 4-bit needs ~5 GiB; 8 GiB gives comfortable headroom.
MIN_FREE_MIB   = 8_000   # MiB  — raise/lower based on your cluster
POLL_INTERVAL  = 60      # seconds between each GPU-memory check

def _wait_for_gpu(min_free_mib: int = MIN_FREE_MIB,
                  poll_interval: int = POLL_INTERVAL) -> str:
    """
    Block until a GPU with at least `min_free_mib` MiB free is available.
    Returns the GPU index as a string.  Never raises — retries forever.
    """
    import time
    attempt = 0
    while True:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                text=True
            )
            rows = [
                (int(r.split(",")[0]), int(r.split(",")[1].strip()))
                for r in out.strip().splitlines()
            ]
            # Pick the GPU with the most free memory
            best_idx, best_free = max(rows, key=lambda x: x[1])

            if best_free >= min_free_mib:
                print(f"[GPU] ✅ GPU {best_idx} ready — "
                      f"{best_free} MiB free (need {min_free_mib} MiB). Starting now.")
                return str(best_idx)
            else:
                attempt += 1
                print(f"[GPU] ⏳ Attempt {attempt}: best GPU {best_idx} only has "
                      f"{best_free} MiB free (need {min_free_mib} MiB). "
                      f"Waiting {poll_interval}s ...")
                time.sleep(poll_interval)

        except Exception as e:
            print(f"[GPU] ⚠️  nvidia-smi failed ({e}). Retrying in {poll_interval}s ...")
            import time
            time.sleep(poll_interval)

_gpu_id = _wait_for_gpu()
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

# Simple token-level accuracy: how many predicted tokens match true tokens
# (ignores padding / prompt tokens marked with -100)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)   # shape: (batch, seq_len)
    mask    = labels != -100                   # ignore padding & prompt tokens
    correct = (predictions == labels) & mask
    accuracy = correct.sum() / mask.sum() if mask.sum() > 0 else 0.0
    return {"accuracy": float(accuracy)}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,           # ← simple match accuracy
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


def normalize(text):
    """Lowercase and remove punctuation for fair comparison."""
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


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


norm_match    = 0   # normalized match  (ignores case + punctuation)
contains_match = 0  # true answer found inside prediction
f1_scores     = []
all_preds     = []
all_refs      = []

for sample in val_data[:50]:
    pred        = generate_answer(sample["prompt"])
    true        = sample["answer"]
    pred_tokens = normalize(pred).split()
    true_tokens = normalize(true).split()

    # Normalized match — ignores case and punctuation
    if normalize(pred) == normalize(true):
        norm_match += 1

    # Contains match — true answer words appear inside prediction
    # (good for generative models that paraphrase)
    overlap = sum(1 for t in true_tokens if t in set(pred_tokens))
    if len(true_tokens) > 0 and overlap / len(true_tokens) >= 0.8:
        contains_match += 1

    # F1
    f1_scores.append(compute_f1(pred_tokens, true_tokens))

    # Collect for corpus-level BLEU
    all_preds.append(pred)
    all_refs.append(true)


n = len(val_data[:50])
norm_acc     = norm_match    / n
contains_acc = contains_match / n
f1_avg       = float(np.mean(f1_scores))

# Corpus-level BLEU via sacrebleu (normalized to 0-1)
bleu_avg = sacrebleu.corpus_bleu(all_preds, [all_refs]).score / 100

# Overall accuracy = average of contains_acc + f1 (most meaningful for generative QA)
overall_acc = (contains_acc + f1_avg) / 2

print(f"\n✅ Normalized Match Acc  : {norm_acc:.3f}")
print(f"✅ Contains Match Acc    : {contains_acc:.3f}  ← best accuracy for generative models")
print(f"✅ F1 Score              : {f1_avg:.3f}")
print(f"✅ BLEU Score            : {bleu_avg:.3f}")
print(f"✅ Overall Accuracy      : {overall_acc:.3f}  ← (Contains + F1) / 2")


# SAVE
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(metrics_dir, exist_ok=True)

results = {
    "model_name": BASE_MODEL,
    "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": {
        "perplexity":          float(perplexity),
        "normalized_match":    float(norm_acc),
        "contains_match":      float(contains_acc),
        "f1_score":            f1_avg,
        "bleu_score":          bleu_avg,
        "overall_accuracy":    float(overall_acc),
    },
}

metrics_file = os.path.join(metrics_dir, "evaluation_results.json")
with open(metrics_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nMetrics saved at: {metrics_file}")
print("Model saved!")