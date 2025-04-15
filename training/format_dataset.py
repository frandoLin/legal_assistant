import json
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from tqdm import tqdm  # Import tqdm for progress bar
import nltk

nltk.download("punkt")

# --- Config ---
use_explanation_ratio = 0.5  # Ratio of examples using explanation output
random.seed(42)

# --- Dynamic Instructions ---
label_instructions = [
    "Classify the clause type based on the sentence.",
    "What type of legal clause is described here?",
    "Identify the legal clause category for this sentence.",
    "Tag the following sentence with its clause type.",
    "Which clause type best matches this sentence?"
]

explanation_instructions = [
    "Explain what kind of legal clause this sentence represents.",
    "Describe the legal purpose of this sentence.",
    "What is the function of this clause in the contract?",
    "Explain the meaning of this clause.",
    "Summarize what this clause is about."
]

# --- Load CUAD JSON ---
with open("../data/CUAD_v1.json", "r") as f:
    cuad = json.load(f)

samples = []

# --- Process CUAD structure ---
for doc in tqdm(cuad["data"], desc="Processing documents"):  # Add progress bar for documents
    for para in doc["paragraphs"]:
        context = para["context"]
        sentences = sent_tokenize(context)

        # Build clause span → clause type mapping
        span_to_label = {}
        for qa in para["qas"]:
            clause_type = qa["id"].split("__")[-1].replace("_", " ")
            for ans in qa["answers"]:
                s, e = ans["answer_start"], ans["answer_start"] + len(ans["text"])
                span_to_label[(s, e)] = clause_type

        # Match sentences to clause spans
        for sent in sentences:
            s_start = context.find(sent)
            s_end = s_start + len(sent)

            matched = [
                clause for (start, end), clause in span_to_label.items()
                if start < s_end and end > s_start
            ]

            label = matched[0] if matched else "No relevant clause"
            use_expl = random.random() < use_explanation_ratio

            # Instruction and output
            if use_expl:
                instruction = random.choice(explanation_instructions)
                if label == "No relevant clause":
                    output = "This sentence does not describe a specific legal clause."
                else:
                    output = f"This is a {label} clause because it defines the {label.lower()} aspect of the contract."
            else:
                instruction = random.choice(label_instructions)
                output = label

            samples.append({
                "instruction": instruction,
                "input": sent.strip(),
                "output": output.strip()
            })

# --- Split into train/val/test ---
train_data, temp_data = train_test_split(samples, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# --- Save as JSONL ---
def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

save_jsonl("../data/cuad_train.jsonl", train_data)
save_jsonl("../data/cuad_val.jsonl", val_data)
save_jsonl("../data/cuad_test.jsonl", test_data)

print(f"Saved {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test examples.")
