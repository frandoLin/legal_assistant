import torch
from vllm import LLM, SamplingParams
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Function to merge models
def merge_models():
    print("Merged model not found. Creating merged model...")
    
    # 2. Merge base model with LoRA weights
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load the LoRA model
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Merge weights
    print("Merging LoRA weights with base model...")
    merged_model = model.merge_and_unload()

    # Save the merged model
    print(f"Saving merged model to {merged_model_path}...")
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    
    # Clean up memory
    del base_model, model, merged_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

# Test prompts for benchmarking
def get_test_prompts():
    return [
        "Identify the legal clause category for this sentence: The Client will make timely payments of amounts earned by the Developer under this Agreement.",
        "Summarize what this clause is about: In the event of the expiration of this Agreement or termination of this Agreement for any reason whatsoever, each party shall promptly return all Confidential Information of the other party.",
        "Explain the purpose of this legal clause: The parties agree to submit to the exclusive jurisdiction of the courts located within the county of Los Angeles, California.",
        "What type of legal clause is this: Neither party shall assign or transfer any rights or obligations under this Agreement without the prior written consent of the other party.",
        "Categorize this legal text: Licensor reserves the right to modify the terms of this Agreement at any time by providing notice to Licensee.",
        "Identify the legal concept in this clause: All intellectual property rights, including copyrights, patents, patent disclosures and inventions, will remain the sole property of the Company.",
        "What is the primary focus of this clause: The Company may terminate this Agreement immediately upon written notice if the Contractor breaches this Agreement.",
        "Analyze this legal statement: Each party shall bear its own costs and expenses incurred in connection with the negotiation and preparation of this Agreement.",
        "What is this legal provision about: This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
        "Explain the legal implications of: The Contractor agrees to indemnify, defend, and hold harmless the Company from and against any claims arising from the Contractor's breach of this Agreement."
    ]

# Function to run inference with vLLM and measure metrics
def run_vllm_inference(num_runs=3):
    print("\n=== Running vLLM Inference Benchmarks ===")
    
    # 3. Load the merged model with vLLM
    print("Loading merged model with vLLM...")
    start_time = time.time()
    llm = LLM(model=merged_model_path, tensor_parallel_size=1)
    model_load_time = time.time() - start_time
    print(f"vLLM model loaded in {model_load_time:.2f} seconds")
    
    prompts = get_test_prompts()
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    
    metrics = {
        "first_token_latency": [],
        "tokens_per_second": [],
        "end_to_end_latency": [],
        "output_length": [],
        "examples": []
    }
    
    # Run multiple times to get averages
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}:")
        
        # Process each prompt
        for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
            # Measure end-to-end latency
            start_time = time.time()
            
            # Since streaming is not supported, we'll use a different approach
            # Generate with a single token to measure first token latency
            first_token_params = SamplingParams(temperature=0.7, max_tokens=1)
            first_token_start = time.time()
            llm.generate([prompt], first_token_params)
            first_token_time = time.time()
            first_token_latency = first_token_time - first_token_start
            
            # Now generate the full response to measure throughput
            full_generation_start = time.time()
            output = llm.generate([prompt], sampling_params)[0]
            end_time = time.time()
            
            # Get token count
            num_tokens = len(output.outputs[0].token_ids)
            
            # Calculate throughput (tokens per second)
            generation_time = end_time - full_generation_start
            tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
            
            # Calculate total end-to-end latency
            end_to_end_latency = end_time - start_time
            
            # Add to metrics
            metrics["first_token_latency"].append(first_token_latency)
            metrics["tokens_per_second"].append(tokens_per_second)
            metrics["end_to_end_latency"].append(end_to_end_latency)
            metrics["output_length"].append(num_tokens)
            
            # Save complete example data (only for the first run)
            if run == 0:
                metrics["examples"].append({
                    "prompt": prompt,
                    "output": output.outputs[0].text,
                    "metrics": {
                        "first_token_latency": first_token_latency,
                        "tokens_per_second": tokens_per_second,
                        "end_to_end_latency": end_to_end_latency,
                        "num_tokens": num_tokens
                    }
                })
                
    # Calculate average metrics
    avg_metrics = {
        "avg_first_token_latency": sum(metrics["first_token_latency"]) / len(metrics["first_token_latency"]),
        "avg_tokens_per_second": sum(metrics["tokens_per_second"]) / len(metrics["tokens_per_second"]),
        "avg_end_to_end_latency": sum(metrics["end_to_end_latency"]) / len(metrics["end_to_end_latency"]),
        "avg_output_length": sum(metrics["output_length"]) / len(metrics["output_length"]),
        "model_load_time": model_load_time
    }
    
    # Print summary
    print("\nvLLM Inference Metrics:")
    print(f"Average First Token Latency: {avg_metrics['avg_first_token_latency']:.4f} seconds")
    print(f"Average Throughput: {avg_metrics['avg_tokens_per_second']:.2f} tokens/second")
    print(f"Average End-to-End Latency: {avg_metrics['avg_end_to_end_latency']:.4f} seconds")
    print(f"Average Output Length: {avg_metrics['avg_output_length']:.2f} tokens")
    
    # Clean up
    del llm
    torch.cuda.empty_cache()
    
    return metrics, avg_metrics

# Function to run inference with standard HF and measure metrics
def run_hf_inference(num_runs=3):
    print("\n=== Running Standard HuggingFace Inference Benchmarks ===")
    
    # Load model with HuggingFace
    print("Loading merged model with HuggingFace...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    
    # Remove the device parameter here - model is already on proper device(s) with device_map="auto"
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    model_load_time = time.time() - start_time
    print(f"HuggingFace model loaded in {model_load_time:.2f} seconds")
    
    prompts = get_test_prompts()
    
    metrics = {
        "first_token_latency": [],
        "tokens_per_second": [],
        "end_to_end_latency": [],
        "output_length": [],
        "examples": []
    }
    
    # Run multiple times to get averages
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}:")
        
        # Process each prompt
        for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
            # Measure end-to-end latency
            start_time = time.time()
            
            # Generate first token only to measure first token latency
            first_token_result = generator(
                prompt, 
                max_new_tokens=1, 
                temperature=0.7,
                do_sample=True,
            )
            first_token_time = time.time()
            first_token_latency = first_token_time - start_time
            
            # Now generate the full response to measure throughput
            full_result = generator(
                prompt, 
                max_new_tokens=256, 
                temperature=0.7,
                do_sample=True,
            )
            end_time = time.time()
            
            # Get output length (tokens)
            output_text = full_result[0]["generated_text"][len(prompt):]
            token_ids = tokenizer.encode(output_text)
            num_tokens = len(token_ids)
            
            # Calculate metrics
            end_to_end_latency = end_time - start_time
            tokens_per_second = num_tokens / (end_time - first_token_time) if num_tokens > 0 else 0
            
            # Add to metrics
            metrics["first_token_latency"].append(first_token_latency)
            metrics["tokens_per_second"].append(tokens_per_second)
            metrics["end_to_end_latency"].append(end_to_end_latency)
            metrics["output_length"].append(num_tokens)
            
            # Save complete example data (only for the first run)
            if run == 0:
                metrics["examples"].append({
                    "prompt": prompt,
                    "output": output_text,
                    "metrics": {
                        "first_token_latency": first_token_latency,
                        "tokens_per_second": tokens_per_second,
                        "end_to_end_latency": end_to_end_latency,
                        "num_tokens": num_tokens
                    }
                })
            
    # Calculate average metrics
    avg_metrics = {
        "avg_first_token_latency": sum(metrics["first_token_latency"]) / len(metrics["first_token_latency"]),
        "avg_tokens_per_second": sum(metrics["tokens_per_second"]) / len(metrics["tokens_per_second"]),
        "avg_end_to_end_latency": sum(metrics["end_to_end_latency"]) / len(metrics["end_to_end_latency"]),
        "avg_output_length": sum(metrics["output_length"]) / len(metrics["output_length"]),
        "model_load_time": model_load_time
    }
    
    # Print summary
    print("\nHuggingFace Inference Metrics:")
    print(f"Average First Token Latency: {avg_metrics['avg_first_token_latency']:.4f} seconds")
    print(f"Average Throughput: {avg_metrics['avg_tokens_per_second']:.2f} tokens/second")
    print(f"Average End-to-End Latency: {avg_metrics['avg_end_to_end_latency']:.4f} seconds")
    print(f"Average Output Length: {avg_metrics['avg_output_length']:.2f} tokens")
    
    # Clean up
    del model, tokenizer, generator
    torch.cuda.empty_cache()
    
    return metrics, avg_metrics

# Function to create visualizations comparing vLLM and HuggingFace
def create_visualizations(vllm_metrics, hf_metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up comparison data
    metrics = {
        "First Token Latency (s)": {
            "vLLM": vllm_metrics["avg_first_token_latency"],
            "HuggingFace": hf_metrics["avg_first_token_latency"]
        },
        "Throughput (tokens/s)": {
            "vLLM": vllm_metrics["avg_tokens_per_second"],
            "HuggingFace": hf_metrics["avg_tokens_per_second"]
        },
        "End-to-End Latency (s)": {
            "vLLM": vllm_metrics["avg_end_to_end_latency"],
            "HuggingFace": hf_metrics["avg_end_to_end_latency"]
        },
        "Model Load Time (s)": {
            "vLLM": vllm_metrics["model_load_time"],
            "HuggingFace": hf_metrics["model_load_time"]
        }
    }
    
    # Create bar charts for each metric
    plt.figure(figsize=(15, 10))
    
    # Set up the colors
    colors = {
        "vLLM": "#1f77b4",  # Blue
        "HuggingFace": "#ff7f0e"  # Orange
    }
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # Plot each metric
    for i, (metric_name, values) in enumerate(metrics.items()):
        ax = axs[i]
        frameworks = list(values.keys())
        metric_values = list(values.values())
        
        bars = ax.bar(
            frameworks, 
            metric_values,
            color=[colors[framework] for framework in frameworks],
            width=0.6
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )
        
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # If this is the throughput metric (higher is better), add an arrow
        if "Throughput" in metric_name:
            max_val = max(metric_values)
            max_idx = metric_values.index(max_val)
            ax.annotate(
                "Higher is better",
                xy=(max_idx, max_val),
                xytext=(max_idx, max_val * 1.1),
                ha='center',
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=10
            )
        # For latency metrics (lower is better), add an arrow
        else:
            min_val = min(metric_values)
            min_idx = metric_values.index(min_val)
            ax.annotate(
                "Lower is better",
                xy=(min_idx, min_val),
                xytext=(min_idx, min_val * 0.5 if min_val > 0 else 0.1),
                ha='center',
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=10
            )
    
    # Add a title for the entire figure
    plt.suptitle(f"Inference Performance: vLLM vs HuggingFace\nModel: {base_model_id} with LoRA", 
                 fontsize=16, 
                 fontweight='bold')
    
    # Add some spacing between plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "inference_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    
    # Create a summary table image
    plt.figure(figsize=(12, 6))
    improvement_percentage = {
        "First Token Latency": (1 - vllm_metrics["avg_first_token_latency"] / hf_metrics["avg_first_token_latency"]) * 100,
        "Throughput": (vllm_metrics["avg_tokens_per_second"] / hf_metrics["avg_tokens_per_second"] - 1) * 100,
        "End-to-End Latency": (1 - vllm_metrics["avg_end_to_end_latency"] / hf_metrics["avg_end_to_end_latency"]) * 100,
        "Model Load Time": (1 - vllm_metrics["model_load_time"] / hf_metrics["model_load_time"]) * 100
    }
    
    cell_text = [
        [f"{hf_metrics['avg_first_token_latency']:.4f}s", f"{vllm_metrics['avg_first_token_latency']:.4f}s", f"{improvement_percentage['First Token Latency']:.1f}%"],
        [f"{hf_metrics['avg_tokens_per_second']:.2f}", f"{vllm_metrics['avg_tokens_per_second']:.2f}", f"{improvement_percentage['Throughput']:.1f}%"],
        [f"{hf_metrics['avg_end_to_end_latency']:.4f}s", f"{vllm_metrics['avg_end_to_end_latency']:.4f}s", f"{improvement_percentage['End-to-End Latency']:.1f}%"],
        [f"{hf_metrics['model_load_time']:.2f}s", f"{vllm_metrics['model_load_time']:.2f}s", f"{improvement_percentage['Model Load Time']:.1f}%"],
    ]
    
    plt.axis('off')
    table = plt.table(
        cellText=cell_text,
        rowLabels=["First Token Latency", "Throughput (tokens/s)", "End-to-End Latency", "Model Load Time"],
        colLabels=["HuggingFace", "vLLM", "Improvement"],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.1, 0.8, 0.7]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color improvement cells based on whether improvement is positive or negative
    # Create row label list for reference
    row_labels = ["First Token Latency", "Throughput (tokens/s)", "End-to-End Latency", "Model Load Time"]
    
    # Color improvement cells based on whether improvement is positive or negative
    for i in range(4):
        cell = table[(i+1, 2)]
        if "Throughput" in row_labels[i]:
            # For throughput, higher is better
            if improvement_percentage["Throughput"] > 0:
                cell.set_facecolor('#d8f3dc')  # Light green
            else:
                cell.set_facecolor('#ffccd5')  # Light red
        else:
            # Create a mapping between row labels and improvement keys
            metric_mapping = {
                "First Token Latency": "First Token Latency",
                "Throughput (tokens/s)": "Throughput",
                "End-to-End Latency": "End-to-End Latency",
                "Model Load Time": "Model Load Time"
            }
            
            # Get the appropriate key for improvement percentage
            metric_key = metric_mapping[row_labels[i]]
            
            if improvement_percentage[metric_key] > 0:
                cell.set_facecolor('#d8f3dc')  # Light green
            else:
                cell.set_facecolor('#ffccd5')  # Light red
    
    plt.title(f"Inference Performance Comparison Summary\nModel: {base_model_id} with LoRA", 
              fontsize=14, 
              fontweight='bold',
              pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_summary_table.png"), dpi=300, bbox_inches='tight')
    
    # Save raw metrics to JSON
    metrics_data = {
        "vLLM": vllm_metrics,
        "HuggingFace": hf_metrics,
        "improvement_percentage": improvement_percentage,
        "test_info": {
            "model": base_model_id,
            "adapter": lora_path,
            "merged_model": merged_model_path,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    with open(os.path.join(output_dir, "inference_metrics.json"), 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\nVisualization and metrics saved to {output_dir}")
    print("\n=== Performance Improvement Summary ===")
    for metric, improvement in improvement_percentage.items():
        if (metric == "Throughput" and improvement > 0) or (metric != "Throughput" and improvement > 0):
            print(f"{metric}: {abs(improvement):.1f}% {'better' if improvement > 0 else 'worse'} with vLLM")
        else:
            print(f"{metric}: {abs(improvement):.1f}% {'better' if (metric == 'Throughput' and improvement > 0) or (metric != 'Throughput' and improvement < 0) else 'worse'} with vLLM")

if __name__ == "__main__":
    # Required for multiprocessing in vLLM
    multiprocessing.set_start_method('spawn', force=True)
    
    # 1. Load base model and LoRA adapter
    base_model_id = "meta-llama/Meta-Llama-3-8B"
    lora_path = "../models/llama3_lora_tuned"

    # Create merged model path
    merged_model_path = "../models/llama3_merged_model"

    # Create output directory for results
    output_dir = f"../evaluation_results/inference_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Check if merged model already exists to avoid remerging
    if not os.path.exists(merged_model_path):
        merge_models()
    else:
        print(f"Using existing merged model at {merged_model_path}")

    # Run benchmarks
    num_runs = 3  # Number of runs for each framework to get more reliable metrics
    
    # Run vLLM benchmarks
    vllm_metrics_raw, vllm_avg_metrics = run_vllm_inference(num_runs)
    
    # Run HuggingFace benchmarks
    hf_metrics_raw, hf_avg_metrics = run_hf_inference(num_runs)
    
    # Create visualizations comparing the two
    create_visualizations(vllm_avg_metrics, hf_avg_metrics, output_dir)