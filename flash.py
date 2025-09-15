# Training Llama3 with Differential Privacy using FlashDP and HuggingFace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flashdp'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.utils import logging
from datasets import load_dataset
from flashdp.api.wrap_model import wrap_with_flashdp_layers
from utils import evaluate_exact_match, evaluate_f1  # Importing the eval functions from utils.py
from peft import LoraConfig, get_peft_model, TaskType  # Added for LoRA
import subprocess  # Add this import


logging.set_verbosity_error()

class FlashDPModel:
    def __init__(
        self,
        model_name,
        train_batch_size,
        eval_batch_size,
        num_epochs,
        learning_rate,
        max_length,
        dp_c,
        dp_noise=None,
        target_epsilon=8.0,
        target_delta=1e-5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        lora_bias="none",
    ):
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.dp_c = dp_c
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.dp_noise = dp_noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        # LoRA configs
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_bias = lora_bias

    def preprocess_dataset(self, subsample_size=5000, seed=101):
        dataset = load_dataset("squad")
        # Subsample for speed
        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(subsample_size))
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(subsample_size // 10))

        def preprocess(example):
            example["input_text"] = "Question: " + example["question"] + " Context: " + example["context"]
            example["target_text"] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
            return example

        dataset = dataset.map(preprocess)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        def tokenize(example):
            # Concatenate input and target for input_ids
            input_text = example["input_text"]
            target_text = example["target_text"]
            full_text = input_text + self.tokenizer.eos_token + target_text + self.tokenizer.eos_token
            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
            # Find split point
            input_ids_input = self.tokenizer(
                input_text + self.tokenizer.eos_token,
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )["input_ids"]
            input_len = sum([1 for t in input_ids_input if t != self.tokenizer.pad_token_id])
            # Mask out input tokens in labels
            labels = [-100] * input_len + tokenized["input_ids"][input_len:]
            # For padding after full_text, also mask as -100
            pad_start = len(labels)
            if pad_start < self.max_length:
                labels += [-100] * (self.max_length - pad_start)
            tokenized["labels"] = labels[:self.max_length]
            return {
                "input_ids": tokenized["input_ids"],
                "labels": tokenized["labels"],
                "attention_mask": tokenized["attention_mask"]
            }

        dataset = dataset.map(tokenize, batched=False, remove_columns=dataset["train"].column_names)

        self.train_loader = DataLoader(
            dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        self.val_loader = DataLoader(
            dataset["validation"],
            batch_size=self.eval_batch_size,
            collate_fn=default_data_collator
        )

    def estimate_noise_multiplier(self, target_epsilon, target_delta, epochs, batch_size, dataset_size):
        """
        Basic privacy accountant for the Gaussian mechanism (approximate, not tight):
        Returns the noise multiplier (sigma) for a given epsilon, delta, epochs, batch_size, dataset_size.
        """
        # This is a very rough approximation for demonstration purposes only!
        # For real applications, use a library like Opacus or the official accountant from FlashDP.
        import math
        steps = int(epochs * (dataset_size // batch_size))
        q = batch_size / dataset_size
        # Analytical Gaussian Mechanism (simplified):
        # sigma >= q * sqrt(2 * steps * log(1/delta)) / epsilon
        if target_epsilon is None or target_delta is None:
            return 1.0
        sigma = (q * math.sqrt(2 * steps * math.log(1/target_delta))) / target_epsilon
        return sigma

    def init_model(self):
        # HuggingFace authentication check
        from transformers import pipeline
        try:
            _ = pipeline("text-generation", model=self.model_name)
        except Exception as e:
            print(f"\nERROR: Unable to access model '{self.model_name}'.\n"
                  f"Make sure you have accepted the license at https://huggingface.co/{self.model_name} and are logged in.\n"
                  f"Original error: {e}\n")
            sys.exit(1)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.model = self.model.to(torch.float32)
        # Print model device map for verification
        if hasattr(self.model, 'hf_device_map'):
            print("[Device Map] Model loaded with device map:")
            print(self.model.hf_device_map)
        self.print_gpu_utilization()  # Print GPU utilization after model load
        # DP wrapping (before LoRA)
        # If dp_noise is not set, estimate it from target_epsilon
        if self.dp_noise is None and self.target_epsilon is not None:
            self.dp_noise = self.estimate_noise_multiplier(
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.num_epochs,
                batch_size=self.train_batch_size,
                dataset_size=len(self.train_loader.dataset)
            )
            print(f"[FlashDP] Estimated noise_multiplier for target_epsilon={self.target_epsilon}: {self.dp_noise}")
        self.model = wrap_with_flashdp_layers(
            self.model,
            target_modules=[torch.nn.Linear],
            skip_layers=[],
            C=self.dp_c,
            noise_multiplier=self.dp_noise
        )
        # LoRA config (after FlashDP)
        target_modules = self.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Try to modify this, remove last 3??
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.lora_bias,
            target_modules=target_modules,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        # Optimizer over only trainable (LoRA) params
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    def train(self):
        self.model.train()
        model_device = next(self.model.parameters()).device
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(model_device)
                labels = batch["labels"].to(model_device)
                attention_mask = batch["attention_mask"].to(model_device) if "attention_mask" in batch else None
                outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            self.print_gpu_utilization()  # Print GPU utilization after each epoch
        print(f"DP Training completed with noise_multiplier={self.dp_noise}, C={self.dp_c}, epochs={self.num_epochs}, batch_size={self.train_batch_size}")
        # Save LoRA adapters only
        adapter_dir = "./llama3-8b-flashdp-lora"
        self.model.save_pretrained(adapter_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(adapter_dir)

    def evaluate(self):
        # Use the same validation loader as in preprocess_dataset
        if self.val_loader is None:
            print("Validation loader not initialized. Run preprocess_dataset() first.")
            return
        model_device = next(self.model.parameters()).device
        print("Evaluating with Exact Match metric from utils.py...")
        evaluate_exact_match(
            self.model,
            self.val_loader,
            model_device,
            self.tokenizer,
            max_gen_length=30
        )
        print("Evaluating with F1 Score metric from utils.py...")
        evaluate_f1(
            self.model,
            self.val_loader,
            model_device,
            self.tokenizer,
            max_gen_length=30
        )

    def print_gpu_utilization(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
            )
            print("\n[GPU UTILIZATION]")
            for line in result.stdout.strip().split('\n'):
                idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
                print(f"GPU {idx} ({name}): Utilization {util}% | Memory {mem_used} MiB / {mem_total} MiB")
            print()
        except Exception as e:
            print(f"Could not query GPU utilization: {e}")

if __name__ == "__main__":

    # Configs
    target_epsilon = 8.0  # Set desired epsilon 
    target_delta = 1e-5   # Set desired delta 
    model_name = "mlabonne/Meta-Llama-3-8B"
    train_batch_size = 2
    eval_batch_size = 2
    num_epochs = 3
    learning_rate = 1e-4
    max_length = 512
    dp_c = 1.0
    dp_noise = None  # Let the code compute noise_multiplier from target_epsilon
    # LoRA configs                      
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = None
    lora_bias = "none"

    if torch.cuda.device_count() == 0:
        print("ERROR: FlashDP requires a CUDA-enabled GPU to run (Triton kernel error: 0 active drivers).")
        print("Please run this script on a machine with a CUDA GPU and the correct CUDA drivers installed.")
        sys.exit(1)
    print(f"Using {torch.cuda.device_count()} GPU(s).")

    flashdp = FlashDPModel(
        model_name=model_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        dp_c=dp_c,
        dp_noise=dp_noise,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_bias=lora_bias,
    )
    flashdp.print_gpu_utilization()  # Print initial GPU utilization

    flashdp.preprocess_dataset()
    flashdp.init_model()
    flashdp.train()
    flashdp.evaluate()