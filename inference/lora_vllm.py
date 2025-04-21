import torch
from vllm import LLM, SamplingParams
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import multiprocessing

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

# Function to run inference
def run_inference():
    # 3. Load the merged model with vLLM
    print("Loading merged model with vLLM...")
    llm = LLM(model=merged_model_path, tensor_parallel_size=1)

    # 4. Run inference with optimized performance
    prompts = [
        "Identify the legal clause category for this sentence: The Client will make timely payments of amounts earned by the Developer under this Agreement.",
        "Summarize what this clause is about: In the event of the expiration of this Agreement or termination..."
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated text: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    # Required for multiprocessing in vLLM
    multiprocessing.set_start_method('spawn', force=True)
    
    # 1. Load base model and LoRA adapter
    base_model_id = "meta-llama/Meta-Llama-3-8B"
    lora_path = "../models/llama3_lora_tuned"

    # Create merged model path
    merged_model_path = "../models/llama3_merged_model"

    # Check if merged model already exists to avoid remerging
    if not os.path.exists(merged_model_path):
        merge_models()
    else:
        print(f"Using existing merged model at {merged_model_path}")

    # Run inference
    run_inference()
