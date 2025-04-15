from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI  # Updated import

# Set OpenAI API key via client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paths and configurations
base_model_path = "meta-llama/Meta-Llama-3-8B"
lora_adapter_path = "../models/llama3_lora_tuned"
test_data_path = "../data/cuad_test.jsonl"

# Load tokenizer and models
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
model.eval()

# Load test set
print("Loading test data...")
test_data = load_dataset("json", data_files=test_data_path)["train"]

# Sample subset of test data for evaluation (to manage API costs)
n_samples = 10  # Adjust based on your OpenAI API budget
test_samples = test_data.select(range(min(n_samples, len(test_data))))
print(f"Selected {len(test_samples)} test samples for evaluation")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_gpt4_evaluation(reference, prediction, instruction, input_text):
    """Use GPT-4 to evaluate the quality of the prediction compared to the reference."""
    try:
        # Updated API call syntax for OpenAI API 1.0+
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # or "gpt-4" depending on availability
            messages=[
                {"role": "system", "content": """You are an expert in legal contract analysis.
                Your task is to evaluate the quality of a model's response for legal clause classification and explanation.
                Focus on the legal substance and meaning rather than exact wording."""},
                {"role": "user", "content": f"""
                Task instruction: {instruction}
                
                Contract clause: {input_text}
                
                Reference answer: {reference}
                
                Model prediction: {prediction}
                
                Please evaluate the model's response compared to the reference on the following dimensions:
                
                1. Factual accuracy (0-10): Is the legal classification or explanation legally correct?
                
                2. Semantic similarity (0-10): How similar is the meaning of the prediction to the reference, 
                   regardless of specific wording? Consider key legal concepts, principles, and interpretations.
                
                3. Practical utility (0-10): How useful would this explanation be to a legal professional 
                   trying to understand the clause's implications?
                
                4. Overall score (0-10)
                
                Format your response with numerical scores for each dimension, followed by a brief 
                justification of your evaluation.
                """
                }
            ],
            temperature=0.2,
            max_tokens=500
        )
        # Updated access to response content
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise e

# Generate predictions and evaluate
results = []

print("Generating predictions and evaluating...")
for idx, item in enumerate(tqdm(test_samples)):
    # Format prompt as used during training
    prompt = f"{item['instruction']}\n{item['input']}\nExplanation:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract generated text
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Explanation:")[-1].strip()
    reference = item["output"]
    
    # Get GPT-4 evaluation
    print(f"\nEvaluating sample {idx+1}/{len(test_samples)}...")
    try:
        evaluation = get_gpt4_evaluation(
            reference=reference,
            prediction=prediction,
            instruction=item["instruction"],
            input_text=item["input"]
        )
        time.sleep(1)  # Rate limiting
    except Exception as e:
        evaluation = f"Evaluation failed: {str(e)}"
    
    # Save results
    results.append({
        "instruction": item["instruction"],
        "input": item["input"],
        "reference": reference,
        "prediction": prediction,
        "evaluation": evaluation
    })
    
    # Periodically save results in case of interruption
    if (idx + 1) % 10 == 0:
        with open("../models/llm_judge_results_partial.json", "w") as f:
            json.dump(results, f, indent=2)

# Parse scores from evaluations
total_scores = {"factual": 0, "similarity": 0, "utility": 0, "overall": 0}
count = 0

for result in results:
    eval_text = result["evaluation"].lower()
    try:
        if "factual accuracy" in eval_text:
            factual = float([line for line in eval_text.split("\n") if "factual accuracy" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            similarity = float([line for line in eval_text.split("\n") if "semantic similarity" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            utility = float([line for line in eval_text.split("\n") if "practical utility" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            overall = float([line for line in eval_text.split("\n") if "overall score" in line.lower() or "overall:" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            
            total_scores["factual"] += factual
            total_scores["similarity"] += similarity
            total_scores["utility"] += utility
            total_scores["overall"] += overall
            count += 1
    except:
        print(f"Could not parse scores from evaluation: {eval_text}")

# Calculate average scores
avg_scores = {k: v/count if count > 0 else 0 for k, v in total_scores.items()}

# Save final results
with open("../models/llm_judge_results.json", "w") as f:
    json.dump({
        "samples": results,
        "average_scores": avg_scores,
        "metadata": {
            "base_model": base_model_path,
            "adapter": lora_adapter_path,
            "samples_evaluated": len(results),
            "successful_evaluations": count
        }
    }, f, indent=2)


# After processing all samples, save a detailed log
detailed_log = []
for idx, result in enumerate(results):
    log_entry = {
        "sample_id": idx + 1,
        "instruction": result["instruction"],
        "input_text": result["input"],
        "reference": result["reference"],
        "prediction": result["prediction"],
        "evaluation": result["evaluation"],
    }
    
    # Try to extract scores from evaluation text
    eval_text = result["evaluation"].lower()
    try:
        if "factual accuracy" in eval_text:
            log_entry["factual_score"] = float([line for line in eval_text.split("\n") 
                                        if "factual accuracy" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            log_entry["similarity_score"] = float([line for line in eval_text.split("\n") 
                                        if "semantic similarity" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            log_entry["utility_score"] = float([line for line in eval_text.split("\n") 
                                        if "practical utility" in line.lower()][0].split(":")[-1].strip().split("/")[0])
            log_entry["overall_score"] = float([line for line in eval_text.split("\n") 
                                        if "overall score" in line.lower() or "overall:" in line.lower()][0].split(":")[-1].strip().split("/")[0])
    except:
        log_entry["score_extraction_error"] = True
    
    detailed_log.append(log_entry)

# Save detailed log with focus on references, predictions and evaluations
with open("../models/llm_judge_detailed_log.json", "w") as f:
    json.dump(detailed_log, f, indent=2)

print(f"Detailed log saved to ../models/llm_judge_detailed_log.json")


# Print summary
print("\n===== LLM-as-Judge Evaluation Summary =====")
print(f"Samples evaluated: {len(results)}")
print(f"Successful evaluations: {count}")
print(f"Average factual accuracy: {avg_scores['factual']:.2f}/10")
print(f"Average semantic similarity: {avg_scores['similarity']:.2f}/10")  # Updated to match metrics
print(f"Average practical utility: {avg_scores['utility']:.2f}/10")       # Updated to match metrics
print(f"Average overall score: {avg_scores['overall']:.2f}/10")
print(f"Full results saved to ../models/llm_judge_results.json")