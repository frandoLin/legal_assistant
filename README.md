# Legal assistant

This repo shows how to fine-tune a LLaMa 3.2 model using **lora** on **legal contract clauses** .
---

## Project Structure

```
project/
├── data/                         # Raw data and soft label outputs
├── training/                     # Format & prefix-tune model
├── inference/                    # Generate explanations using tuned model
├── models/                       # Prefix-tuned model (PEFT format)
├── requirements.txt
└── README.md
```

---

## Setup

### Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Pipeline Overview

### Step 1: Format dataset to instruction–input–output format
**Script**: `training/format_dataset.py`  
**What it does**: Extracts sentence + label pairs and structures them like:
```json
{
  "instruction": "Explain the legal concept in this clause.",
  "input": "The contract may be terminated at any time...",
  "output": "Termination"
}
```
**Run it**:
```bash
python training/format_dataset.py
```
Output → `data/xx_train.jsonl`

---

### Step 2: Train a llama3 model with lora
**Script**: `training/train_lora_tuning.py`  
**What it does**: Loads `xx_train.jsonl`, applies lora tuning using Hugging Face + PEFT.

**Run it**:
```bash
python training/train_lora_tuning.py
```
Output → `models/llama3_lora_tuned/`

---

### Step 3 Evaluate explanation quality with GPT 4
**Script**: `inference/evaluate_llm_as_judge.py`  
**What it does**: Compares generated soft labels to GPT-4 with the references in the test set.

🏃 **Run it**:
```bash
python inference/evaluate_llm_as_judge.py
```
Input → `data/eval_set.jsonl` with:
```json
{ "prediction": "...", "reference": "..." }
```



## Reference
- CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review https://arxiv.org/abs/2103.06268
