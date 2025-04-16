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

def setup_model(model_name, adapter_path=None):
    """Set up a model for evaluation"""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model_label = f"{model_name}_LoRA"
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
    
    # 4. Create a differences bar chart
    if len(models) >= 2:
        plt.figure(figsize=(12, 8))
        
        # Calculate differences
        model1 = models[0]
        model2 = models[1]
        diff_data = []
        
        for metric in metrics:
            diff = results_dict[model2]["average_scores"][metric] - results_dict[model1]["average_scores"][metric]
            diff_data.append({
                "Metric": metric.capitalize(),
                "Difference": diff
            })
        
        diff_df = pd.DataFrame(diff_data)
        
        # Plot
        ax = sns.barplot(x="Metric", y="Difference", data=diff_df)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title(f"Score Difference: {model2} vs {model1}", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Score Difference", fontsize=14)
        
        # Add value labels
        for i, v in enumerate(diff_df["Difference"]):
            color = "green" if v > 0 else "red"
            plt.text(i, v + 0.1 if v > 0 else v - 0.3, f"{v:.2f}", 
                    color=color, fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "model_improvement.png"))
    
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
    # positive_count, negative_count = 0, 0
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

    # for i in range(len(data)):
    #     while positive_count + negative_count < n_samples:
            
    #         if data[i]["output"] not in negative_label and positive_count < n_samples // 2:
    #             positive_count += 1
    #             test_data.append(data[i])
    #         elif data[i]["output"] in negative_label and negative_count < n_samples // 2:
    #             negative_count += 1
    #             test_data.append(data[i])
    #     break

    return test_data

def main():
    parser = argparse.ArgumentParser(description="Compare base and LoRA models on legal tasks")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B", help="Base model name")
    parser.add_argument("--lora_adapter", default="../models/llama3_lora_tuned", help="Path to LoRA adapter")
    parser.add_argument("--test_data", default="../data/cuad_test.jsonl", help="Path to test data")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_dir", default="../evaluation_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    if args.test_data:
        test_data = load_dataset("json", data_files=args.test_data)["train"]
        # Sample data if needed
        if args.samples > 0 and args.samples < len(test_data):
            test_data = class_balanced_sample(test_data, args.samples)
    
    print(f"Evaluating {len(test_data)} test samples")
    
    # Setup and evaluate base model
    base_model, base_tokenizer, base_label = setup_model(args.base_model)
    base_results = evaluate_model(base_model, base_tokenizer, base_label, test_data)
    del base_model
    torch.cuda.empty_cache()
    time.sleep(5)  # Allow time for GPU memory to clear
    
    # Setup and evaluate LoRA model
    lora_model, lora_tokenizer, lora_label = setup_model(args.base_model, args.lora_adapter)
    lora_results = evaluate_model(lora_model, lora_tokenizer, lora_label, test_data)
    
    # Save detailed logs
    all_detailed_logs = base_results["detailed_log"] + lora_results["detailed_log"]
    with open(os.path.join(output_dir, "detailed_evaluation_log.json"), "w") as f:
        json.dump(all_detailed_logs, f, indent=2)
    
    # Convert to DataFrame for CSV export
    detailed_df = pd.DataFrame(all_detailed_logs)
    detailed_df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
    
    # Save results for each model
    results_dict = {
        base_label: base_results,
        lora_label: lora_results
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump({
            model: {k: v for k, v in results.items() if k != "detailed_log"} 
            for model, results in results_dict.items()
        }, f, indent=2)
    
    # Create visualizations
    create_visualizations(results_dict, output_dir)
    
    print(f"Evaluation complete! Results and visualizations saved to {output_dir}")
    
    # Check for significant improvement
    base_overall = base_results["average_scores"]["overall"]
    lora_overall = lora_results["average_scores"]["overall"]
    improvement = lora_overall - base_overall
    
    print("\n=== Deployment Decision ===")
    if improvement > 0.2:
        print(f"SIGNIFICANT IMPROVEMENT: +{improvement:.2f} points")
        print(f"Recommendation: Deploy the LoRA model")
    elif improvement > 0:
        print(f"SLIGHT IMPROVEMENT: +{improvement:.2f} points")
        print(f"Recommendation: Consider deploying if other factors warrant it")
    else:
        print(f"NO IMPROVEMENT: {improvement:.2f} points")
        print(f"Recommendation: Keep the base model")
        
    # Generate deployment decision file
    decision = {
        "timestamp": datetime.now().isoformat(),
        "base_model": base_label,
        "lora_model": lora_label,
        "base_score": base_overall,
        "lora_score": lora_overall,
        "improvement": improvement,
        "should_deploy": improvement > 0.2,
        "recommendation": "deploy" if improvement > 0.2 else "consider" if improvement > 0 else "keep_base"
    }
    
    with open(os.path.join(output_dir, "deployment_decision.json"), "w") as f:
        json.dump(decision, f, indent=2)

if __name__ == "__main__":
    main()