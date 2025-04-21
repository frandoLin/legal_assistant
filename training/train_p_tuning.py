from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit, prepare_model_for_kbit_training
import torch
import numpy as np
from transformers.trainer_callback import TrainerCallback
from datetime import datetime
import matplotlib.pyplot as plt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Try to use a non-gradient checkpointing approach for p-tuning
# First, let's disable using prepare_model_for_kbit_training since it might be enabling gradient checkpointing
USE_KBIT_TRAINING = False

# Load dataset
train_dataset = Dataset.from_json("../data/cuad_train.jsonl")
val_dataset = Dataset.from_json("../data/cuad_val.jsonl")

print(f"Original dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Reduce dataset size with stratified sampling to maintain class distribution
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)

# Limit to 20,000 for training and 2,000 for validation
train_dataset = train_dataset.select(range(min(20000, len(train_dataset))))
val_dataset = val_dataset.select(range(min(2000, len(val_dataset))))

print(f"Reduced dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Load tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Format inputs for causal language modeling
def tokenize_fn(example):
    full_text = f"{example['instruction']}\n{example['input']}\nExplanation: {example['output']}"
    tokenized = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Process datasets
print("Processing training data...")
train_data = train_dataset.map(
    lambda example: tokenize_fn(example), 
    remove_columns=train_dataset.column_names
)
print("Processing validation data...")
val_data = val_dataset.map(
    lambda example: tokenize_fn(example),
    remove_columns=val_dataset.column_names
)

# Delete original datasets to free up memory
del train_dataset
del val_dataset
torch.cuda.empty_cache()

# Load model without gradient checkpointing
print("Loading model with gradient checkpointing explicitly disabled...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_cache=True,  # Setting use_cache=True prevents gradient checkpointing
)

# Make absolutely sure gradient checkpointing is disabled
if hasattr(model, "config"):
    model.config.use_cache = True
    setattr(model.config, "gradient_checkpointing", False)

# Only use prepare_model_for_kbit_training if safe (it might enable gradient checkpointing)
if USE_KBIT_TRAINING:
    model = prepare_model_for_kbit_training(model)
else:
    # Instead, set modules to training mode manually
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # For backward compatibility
    model.is_loaded_in_8bit = True
    model.model_parallel = True

# Define P-Tuning v2 config
# The key difference from prefix tuning is using PromptTuningConfig with num_transformer_submodules=3
print("Setting up P-Tuning v2 configuration...")
ptuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.RANDOM,
    num_virtual_tokens=16,
    token_dim=model.config.hidden_size,
    # Use num_transformer_submodules=1 instead of 3 to avoid dimension mismatch
    num_transformer_submodules=1,  
    num_attention_heads=model.config.num_attention_heads,
    num_layers=model.config.num_hidden_layers,
    prompt_tuning_init_text=None,
)

# Create metrics callback
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.last_logged_step = -1
        
    def on_init_end(self, args, state, control, **kwargs):
        pass
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.last_logged_step = -1
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
            
        if "loss" in logs and state.global_step > self.last_logged_step:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)
            self.last_logged_step = state.global_step
            
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            
    def plot_metrics(self, save_path="../models/ptuningv2_training_metrics.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.steps, self.train_losses, label="Training Loss")
        
        if self.eval_losses:
            if len(self.steps) >= len(self.eval_losses):
                eval_indices = np.linspace(0, len(self.steps)-1, len(self.eval_losses), dtype=int)
                eval_steps = [self.steps[i] for i in eval_indices]
                plt.plot(eval_steps, self.eval_losses, label="Validation Loss", marker="o")
            
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title(f"P-Tuning v2 - Training and Validation Loss - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path)
        print(f"Metrics plot saved to {save_path}")

metrics_callback = MetricsCallback()

# Training arguments - WITHOUT gradient_checkpointing
args = TrainingArguments(
    output_dir="../models/llama3_ptuningv2_tuned",  # Change output directory
    per_device_train_batch_size=2,  # Smaller batch size
    gradient_accumulation_steps=8,  # More gradient accumulation steps
    num_train_epochs=2,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    # NO gradient_checkpointing here
    optim="paged_adamw_8bit",
    learning_rate=5e-4,
    report_to="none",
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
)

# Now create the P-Tuning v2 model
print("Creating P-Tuning v2 model...")
model = get_peft_model(model, ptuning_config)
print(f"Trainable parameters: {model.print_trainable_parameters()}")

# Create trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[metrics_callback]
)

# Clean up memory before training
torch.cuda.empty_cache()
print("Starting training...")
train_result = trainer.train()

# Print and save metrics
print("\nTraining complete! Final metrics:")
print(f"Training loss: {train_result.training_loss:.4f}")

# Run final evaluation
eval_result = trainer.evaluate()
print(f"Validation loss: {eval_result['eval_loss']:.4f}")

# Plot training metrics
metrics_callback.plot_metrics()

# Save the P-Tuning v2 model
model.save_pretrained("../models/llama3_ptuningv2_tuned")
print("Model saved successfully!")

# Save training summary
with open("../models/ptuningv2_training_summary.txt", "w") as f:
    f.write(f"P-Tuning v2 completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {model_id}\n")
    f.write(f"Number of virtual tokens: {ptuning_config.num_virtual_tokens}\n")
    f.write(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}\n")
    f.write(f"Final training loss: {train_result.training_loss:.4f}\n")
    f.write(f"Final validation loss: {eval_result['eval_loss']:.4f}\n")
    f.write(f"Total training steps: {trainer.state.global_step}\n")
    f.write(f"Note: Using P-Tuning v2 with num_transformer_submodules=3 for deep prompting\n")

print("Training summary saved to ../models/ptuningv2_training_summary.txt")