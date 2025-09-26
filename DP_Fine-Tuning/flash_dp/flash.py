# Training Llama3 with Differential Privacy using FlashDP and HuggingFace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flashdp'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.utils import logging
from datasets import load_dataset
from flashdp.api.wrap_model import wrap_with_flashdp_layers
from peft import LoraConfig, get_peft_model, TaskType  # Added for LoRA
from utils.model_utils import *
from utils.gpu_usage import *

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
        lora_alpha=16,
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

    def preprocess_dataset(self, train_size, eval_size, seed=101):
        dataset = load_dataset(self.dataset_name)

        dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(train_size))
        dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(eval_size))

        # Initialize tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        def preprocess_and_tokenize_train(example):
            input_text = "Context: " + example["context"] + " Question: " + example["question"] + " Answer: "
            target_text = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
            full_text = input_text + target_text
            
            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_input_length + self.max_target_length,
                truncation=True,
                padding="max_length",
            )

            input_tokens = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
                add_special_tokens=False
            )
            input_length = len(input_tokens["input_ids"])

            labels = tokenized["input_ids"].copy()
            labels[:input_length] = [-100] * input_length  # mask input
            labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

            tokenized["labels"] = labels
            return tokenized
        
        def preprocess_and_tokenize_eval(example):
            # Prompt only (no gold answer appended to input_ids)
            input_text = "Context: " + example["context"] + " Question: " + example["question"] + " Answer: "
            target_text = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""

            # Tokenize prompt only for inputs
            tokenized_inputs = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length"
            )

            # Tokenize full_text (with answer) just to build labels
            tokenized_full = self.tokenizer(
                input_text + target_text,
                max_length=self.max_input_length + self.max_target_length,
                truncation=True,
                padding="max_length"
            )

            # Mask out the input part, keep only the answer portion for labels
            labels = tokenized_full["input_ids"].copy()
            input_length = len(self.tokenizer(input_text, add_special_tokens=False)["input_ids"])
            labels[:input_length] = [-100] * input_length
            labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

            tokenized_inputs["labels"] = labels
            
            return tokenized_inputs

        train_dataset = dataset["train"].map(
            preprocess_and_tokenize_train,
            batched=False,
            remove_columns=dataset["train"].column_names,
        )

        eval_dataset = dataset["validation"].map(
            preprocess_and_tokenize_eval, 
            batched=False,
            remove_columns=dataset["validation"].column_names
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        
        self.val_loader = DataLoader(
            eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=default_data_collator,
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
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.model = self.model.to(torch.float32)
        
        # # Print model device map for verification
        # if hasattr(self.model, 'hf_device_map'):
        #     print("[Device Map] Model loaded with device map:")
        #     print(self.model.hf_device_map)
            
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
        target_modules = self.lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
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
        print("Evaluating...")
        evaluate_model(
            self.model,
            self.val_loader,
            model_device,
            self.tokenizer,
            max_gen_length=10,
            show_samples=10,
        )

if __name__ == "__main__":

    # Configs
    target_epsilon = 8.0  # Set desired epsilon 
    target_delta = 1e-5   # Set desired delta 
    model_name = "mlabonne/Meta-Llama-3-8B"
    train_batch_size = 1
    eval_batch_size = 1
    num_epochs = 5
    learning_rate = 2e-4
    max_length = 512
    train_size = 5000
    eval_size = 500
    dp_c = 1.0
    dp_noise = None  # Let the code compute noise_multiplier from target_epsilon
    # LoRA configs                      
    lora_r = 16
    lora_alpha = 16
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
    # Start GPU utilization logging using utils
    gpu_util_thread, gpu_util_stop_event, gpu_util_data = start_gpu_utilization_logging(interval=1.0)

    flashdp.preprocess_dataset(train_size=train_size, eval_size=eval_size)
    flashdp.init_model()
    flashdp.train()
    
    print(f"Model: {model_name}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max input length: {max_length}")
    print(f"Max target length: {max_length}")
    print(f"Traing size: {train_size}")
    print(f"Eval size: {eval_size}")

    flashdp.evaluate()
    
    stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
    print_gpu_utilization_summary(gpu_util_data)