# Legal assistant

This repo shows how to fine-tune a LLaMa 3.2 model using **lora** on **legal contract clauses** , then apply it to a **RAG system** for improved semantic retrieval using **soft explanations**.

---

## Project Structure

```
project/
├── data/                         # Raw data and soft label outputs
├── training/                     # Format & prefix-tune model
├── inference/                    # Generate explanations using tuned model
├── retrieval/                    # Embed & evaluate RAG performance
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

### Step 2: Train a prefix-tuned GPT-2 model on soft labels
**Script**: `training/train_lora_tuning.py`  
**What it does**: Loads `xx_train.jsonl`, applies lora tuning using Hugging Face + PEFT.

**Run it**:
```bash
python training/train_lora_tuning.py
```
Output → `models/llama3_lora_tuned/`

---

### Step 3 Evaluate explanation quality with GPT 4
**Script**: `inference/evaluate_bleu_rouge.py`  
**What it does**: Compares generated soft labels to GPT-4 with the references in the test set.

🏃 **Run it**:
```bash
python inference/evaluate_llm_as_judge.py
```
Input → `data/eval_set.jsonl` with:
```json
{ "prediction": "...", "reference": "..." }
```
## Todo:
---

### Step 4(Optional): Build RAG Retrieval Index
Suggested: `retrieval/build_faiss_index.py`  
**What it should do**:
- Embed either raw or soft-labeled clauses
- Store vectors in a FAISS index

---

### Step 5 (Optional): Compare retrieval accuracy
Suggested: `retrieval/compare_rag_metrics.py`  
**What it should do**:
- Evaluate RAG using **Recall@k** and **MRR**
- Compare baseline (raw text) vs. prefix-tuned soft explanations

---

### Step 6 (ptional): Visualize metric distributions
Suggested: `retrieval/visualize_results.py`  
**What it should do**:
- Plot histograms of recall hit/miss
- Plot MRR distributions

---

## What's Unique About This Project?

- Prefix-tuning instead of full model fine-tuning
- Explains contracts via **natural language soft labels**
- Integrates into a retrieval pipeline for **semantic RAG**

---

## Example Use Case

You want to build a system that:
- Ingests new contract clauses
- Explains them like a legal expert
- Finds similar clauses in a database using meaning, not keywords

This repo gives you a **trainable and explainable LLM-based backend** to do exactly that.

## Reference
- CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review https://arxiv.org/abs/2103.06268
