from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm
import json
import os
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI  
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import random
from datetime import datetime
import gc

# Set OpenAI API key via client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Keep the original get_gpt4_evaluation function for consistency
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

def clear_gpu_memory():
    """Aggressively clear GPU memory between model evaluations"""
    gc.collect()
    torch.cuda.empty_cache()
    
    # Force garbage collection
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)  # Give system time to release memory

def setup_model(model_name, adapter_path=None, adapter_type="lora"):
    """Set up a model for evaluation with support for different adapter types"""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model {model_name}...")
    # Add CPU offloading options to fix memory issues
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    if adapter_path:
        print(f"Loading {adapter_type} adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Set model label based on adapter type
        if adapter_type.lower() == "lora":
            model_label = f"{model_name}_LoRA"
        elif adapter_type.lower() == "ptuning":
            model_label = f"{model_name}_PTuning"
        elif adapter_type.lower() == "prefix":
            model_label = f"{model_name}_Prefix"
        else:
            model_label = f"{model_name}_{adapter_type}"
    else:
        model = base_model
        model_label = f"{model_name}_base"
        
    model.eval()
    return model, tokenizer, model_label

def generate_prediction(model, tokenizer, instruction, input_text):
    """Generate a prediction using the model"""
    prompt = f"{instruction}\n{input_text}\nExplanation:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Explanation:")[-1].strip()
    return prediction

def parse_scores(eval_text):
    """Parse scores from evaluation text - using the same approach as original script"""
    try:
        eval_text = eval_text.lower()
        factual = float([line for line in eval_text.split("\n") if "factual accuracy" in line.lower()][0].split(":")[-1].strip().split("/")[0])
        similarity = float([line for line in eval_text.split("\n") if "semantic similarity" in line.lower()][0].split(":")[-1].strip().split("/")[0])
        utility = float([line for line in eval_text.split("\n") if "practical utility" in line.lower()][0].split(":")[-1].strip().split("/")[0])
        overall = float([line for line in eval_text.split("\n") if "overall score" in line.lower() or "overall:" in line.lower()][0].split(":")[-1].strip().split("/")[0])
        
        return {
            "factual": factual,
            "similarity": similarity,
            "utility": utility,
            "overall": overall,
            "score_extraction_error": False
        }
    except Exception as e:
        print(f"Error parsing scores: {e}")
        return {
            "factual": 0,
            "similarity": 0,
            "utility": 0,
            "overall": 0,
            "score_extraction_error": True,
            "raw_evaluation": eval_text
        }

def create_visualizations(results_dict, output_dir):
    """Create visualizations comparing the models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    models = list(results_dict.keys())
    metrics = ["factual", "similarity", "utility", "overall"]
    
    # Create DataFrame for easier plotting
    plot_data = []
    for model in models:
        avg_scores = results_dict[model]["average_scores"]
        for metric in metrics:
            plot_data.append({
                "Model": model,
                "Metric": metric.capitalize(),
                "Score": avg_scores[metric]
            })
    
    df = pd.DataFrame(plot_data)
    
    # 1. Bar chart comparison
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    ax = sns.barplot(x="Metric", y="Score", hue="Model", data=df)
    plt.title("Model Performance Comparison", fontsize=16)
    plt.xlabel("Evaluation Metric", fontsize=14)
    plt.ylabel("Score (0-10)", fontsize=14)
    plt.ylim(0, 10)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_bar.png"))
    
    # 2. Radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Prepare data for radar chart
    metrics_radar = ["Factual", "Similarity", "Utility", "Overall"]
    num_metrics = len(metrics_radar)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    for model in models:
        avg_scores = results_dict[model]["average_scores"]
        values = [avg_scores["factual"], avg_scores["similarity"], 
                 avg_scores["utility"], avg_scores["overall"]]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar)
    ax.set_yticks(range(0, 11, 2))
    ax.set_yticklabels([str(i) for i in range(0, 11, 2)])
    ax.set_ylim(0, 10)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.title("Model Performance Radar Chart", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_radar.png"))
    
    # 3. Create a summary table
    plt.figure(figsize=(10, 6))
    summary_data = {}
    
    for model in models:
        avg_scores = results_dict[model]["average_scores"]
        summary_data[model] = {
            "Factual": f"{avg_scores['factual']:.2f}",
            "Similarity": f"{avg_scores['similarity']:.2f}",
            "Utility": f"{avg_scores['utility']:.2f}",
            "Overall": f"{avg_scores['overall']:.2f}"
        }
    
    summary_df = pd.DataFrame(summary_data).T
    
    # Plot the table
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=summary_df.values,
        rowLabels=summary_df.index,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title("Performance Summary Table", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_table.png"))
    
    # 4. Create a differences bar chart comparing to base model
    if "meta-llama/Meta-Llama-3-8B_base" in results_dict and len(models) > 1:
        plt.figure(figsize=(12, 8))
        
        # Calculate differences vs base model
        base_model = "meta-llama/Meta-Llama-3-8B_base"
        diff_data = []
        
        for model in models:
            if model != base_model:
                for metric in metrics:
                    diff = results_dict[model]["average_scores"][metric] - results_dict[base_model]["average_scores"][metric]
                    diff_data.append({
                        "Model": model,
                        "Metric": metric.capitalize(),
                        "Improvement": diff
                    })
        
        diff_df = pd.DataFrame(diff_data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x="Metric", y="Improvement", hue="Model", data=diff_df)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f"Score Improvement vs Base Model", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Score Improvement", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_improvements.png"))
    
    print(f"Visualizations saved to {output_dir}")

def evaluate_model(model, tokenizer, model_label, test_data):
    """Evaluate a model on test data using the original evaluation function"""
    results = []
    total_scores = {"factual": 0, "similarity": 0, "utility": 0, "overall": 0}
    count = 0
    detailed_log = []
    
    for idx, item in enumerate(tqdm(test_data, desc=f"Evaluating {model_label}")):
        try:
            instruction = item.get("instruction", "Explain the legal concept in this clause:")
            input_text = item.get("input", item.get("text", ""))
            reference = item.get("output", item.get("reference", ""))
            
            # Generate prediction
            prediction = generate_prediction(model, tokenizer, instruction, input_text)
            
            # Get GPT-4 evaluation
            print(f"\nEvaluating {model_label} sample {idx+1}/{len(test_data)}...")
            evaluation = get_gpt4_evaluation(
                reference=reference,
                prediction=prediction,
                instruction=instruction,
                input_text=input_text
            )
            
            # Parse scores
            scores = parse_scores(evaluation)
            
            if not scores["score_extraction_error"]:
                total_scores["factual"] += scores["factual"]
                total_scores["similarity"] += scores["similarity"]
                total_scores["utility"] += scores["utility"]
                total_scores["overall"] += scores["overall"]
                count += 1
            
            # Save result
            result = {
                "instruction": instruction,
                "input": input_text,
                "prediction": prediction,
                "reference": reference,
                "evaluation": evaluation
            }
            results.append(result)
            
            # Create detailed log entry
            log_entry = {
                "sample_id": idx + 1,
                "model": model_label,
                "instruction": instruction,
                "input_text": input_text,
                "reference": reference,
                "prediction": prediction,
                "evaluation": evaluation,
                "factual_score": scores["factual"],
                "similarity_score": scores["similarity"],
                "utility_score": scores["utility"],
                "overall_score": scores["overall"],
                "score_extraction_error": scores["score_extraction_error"]
            }
            detailed_log.append(log_entry)
            
            # Rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error evaluating item {idx} with {model_label}: {e}")
    
    # Calculate average scores
    avg_scores = {
        "factual": total_scores["factual"] / count if count > 0 else 0,
        "similarity": total_scores["similarity"] / count if count > 0 else 0,
        "utility": total_scores["utility"] / count if count > 0 else 0,
        "overall": total_scores["overall"] / count if count > 0 else 0
    }
    
    print(f"\n{model_label} Evaluation Complete!")
    print(f"Average Scores:")
    print(f"Factual Accuracy: {avg_scores['factual']:.2f}/10")
    print(f"Semantic Similarity: {avg_scores['similarity']:.2f}/10")
    print(f"Practical Utility: {avg_scores['utility']:.2f}/10")
    print(f"Overall Score: {avg_scores['overall']:.2f}/10")
    
    return {
        "results": results,
        "average_scores": avg_scores,
        "detailed_log": detailed_log,
        "metadata": {
            "model": model_label,
            "samples_evaluated": len(results),
            "successful_evaluations": count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }


def class_balanced_sample(data, n_samples):
    """Sample a balanced subset of positive and negative examples"""
    test_data = []
    positive_samples, negative_samples = [], []
    negative_label = ["This sentence does not describe a specific legal clause.", "No relevant clause"]

    for item in data:
        if isinstance(item, dict) and "output" in item:
            if item["output"] not in negative_label:
                positive_samples.append(item)
            else:
                negative_samples.append(item)

    # Get counts of available examples
    available = min(len(positive_samples),len(negative_samples)) 

    if available < n_samples // 2:
        test_data.extend(random.sample(positive_samples, available))
        test_data.extend(random.sample(negative_samples, available))
    else:
        test_data.extend(random.sample(positive_samples, n_samples // 2))
        test_data.extend(random.sample(negative_samples, n_samples // 2))

    return test_data

def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuning methods on legal tasks")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B", help="Base model name")
    parser.add_argument("--lora_adapter", default="../models/llama3_lora_tuned", help="Path to LoRA adapter")
    parser.add_argument("--ptuning_adapter", default="../models/llama3_ptuningv2_tuned", help="Path to P-Tuning adapter")
    parser.add_argument("--prefix_adapter", default="../models/llama3_prefix_tuned", help="Path to Prefix Tuning adapter")
    parser.add_argument("--test_data", default="../data/cuad_test.jsonl", help="Path to test data")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", default="../evaluation_results", help="Output directory for results")
    parser.add_argument("--skip_base", action="store_true", help="Skip evaluation of base model")
    parser.add_argument("--skip_lora", action="store_true", help="Skip evaluation of LoRA model")
    parser.add_argument("--skip_ptuning", action="store_true", help="Skip evaluation of P-Tuning model")
    parser.add_argument("--skip_prefix", action="store_true", help="Skip evaluation of Prefix Tuning model")
    args = parser.parse_args()
    
    # Create output directory
    model_parts = []
    if not args.skip_base:
        model_parts.append("base")
    if not args.skip_lora:
        model_parts.append("lora")
    if not args.skip_ptuning:
        model_parts.append("ptuning")
    if not args.skip_prefix:
        model_parts.append("prefix")

    model_string = "_".join(model_parts)
    output_dir = os.path.join(args.output_dir, f"compare_{model_string}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    if args.test_data:
        test_data = load_dataset("json", data_files=args.test_data)["train"]
        # Sample data if needed
        if args.samples > 0 and args.samples < len(test_data):
            test_data = class_balanced_sample(test_data, args.samples)
    
    print(f"Evaluating {len(test_data)} test samples")
    
    # Initialize results dictionary
    results_dict = {}
    
    # Setup and evaluate base model (if not skipped)
    if not args.skip_base:
        base_model, base_tokenizer, base_label = setup_model(args.base_model)
        base_results = evaluate_model(base_model, base_tokenizer, base_label, test_data)
        results_dict[base_label] = base_results
        del base_model, base_tokenizer
        clear_gpu_memory()
    
    # Setup and evaluate LoRA model (if not skipped)
    if not args.skip_lora:
        lora_model, lora_tokenizer, lora_label = setup_model(args.base_model, args.lora_adapter, "lora")
        lora_results = evaluate_model(lora_model, lora_tokenizer, lora_label, test_data)
        results_dict[lora_label] = lora_results
        del lora_model, lora_tokenizer
        clear_gpu_memory()
    
    # Setup and evaluate P-Tuning model (if not skipped)
    if not args.skip_ptuning and args.ptuning_adapter:
        ptuning_model, ptuning_tokenizer, ptuning_label = setup_model(
            args.base_model, args.ptuning_adapter, "ptuning"
        )
        ptuning_results = evaluate_model(ptuning_model, ptuning_tokenizer, ptuning_label, test_data)
        results_dict[ptuning_label] = ptuning_results
        del ptuning_model, ptuning_tokenizer
        clear_gpu_memory()
    
    # Setup and evaluate Prefix Tuning model (if not skipped)
    if not args.skip_prefix and args.prefix_adapter:
        prefix_model, prefix_tokenizer, prefix_label = setup_model(
            args.base_model, args.prefix_adapter, "prefix"
        )
        prefix_results = evaluate_model(prefix_model, prefix_tokenizer, prefix_label, test_data)
        results_dict[prefix_label] = prefix_results
        del prefix_model, prefix_tokenizer
        clear_gpu_memory()
    
    # Save detailed logs
    all_detailed_logs = []
    for model_results in results_dict.values():
        all_detailed_logs.extend(model_results["detailed_log"])
    
    with open(os.path.join(output_dir, "detailed_evaluation_log.json"), "w") as f:
        json.dump(all_detailed_logs, f, indent=2)
    
    # Convert to DataFrame for CSV export
    detailed_df = pd.DataFrame(all_detailed_logs)
    detailed_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    # Save results for each model
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({
            model: {k: v for k, v in results.items() if k != "detailed_log"} 
            for model, results in results_dict.items()
        }, f, indent=2)
    
    # Create visualizations
    create_visualizations(results_dict, output_dir)
    
    print(f"Evaluation complete! Results and visualizations saved to {output_dir}")
    
    # Compare model performances
    print("\n=== Model Comparison ===")
    model_names = list(results_dict.keys())
    
    if "meta-llama/Meta-Llama-3-8B_base" in results_dict:
        base_overall = results_dict["meta-llama/Meta-Llama-3-8B_base"]["average_scores"]["overall"]
        print(f"Base Model Score: {base_overall:.2f}/10")
        
        for model in model_names:
            if model != "meta-llama/Meta-Llama-3-8B_base":
                model_score = results_dict[model]["average_scores"]["overall"]
                improvement = model_score - base_overall
                print(f"{model} Score: {model_score:.2f}/10 ({'+' if improvement > 0 else ''}{improvement:.2f})")
    
    # Generate comparison summary
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "models_evaluated": model_names,
        "scores": {model: results_dict[model]["average_scores"] for model in model_names},
    }
    
    # Find best model overall
    if len(model_names) > 1:
        best_model = max(model_names, key=lambda m: results_dict[m]["average_scores"]["overall"])
        best_score = results_dict[best_model]["average_scores"]["overall"]
        comparison["best_model"] = {
            "name": best_model,
            "score": best_score
        }
        print(f"\nBest overall model: {best_model} (Score: {best_score:.2f}/10)")
    
    with open(os.path.join(output_dir, "model_comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=2)

if __name__ == "__main__":
    main()