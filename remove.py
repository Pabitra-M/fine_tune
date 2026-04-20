import json

INPUT_FILE = "all_dataset.json"
OUTPUT_FILE = "clean_dataset.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

seen_questions = set()
clean_data = []

for item in data:
    q = item.get("question", "").strip().lower()

    if q not in seen_questions:
        seen_questions.add(q)
        clean_data.append(item)

print(f"Original size: {len(data)}")
print(f"After removing duplicates: {len(clean_data)}")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean_data, f, indent=4, ensure_ascii=False)