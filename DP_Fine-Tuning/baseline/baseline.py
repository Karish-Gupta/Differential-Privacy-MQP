import numpy as np
import torch
import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from utils import *
from huggingface_hub import login
import os

# Login to HF CLI
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
# transformers.utils.logging.set_verbosity_debug()


class Baseline:
    def __init__(
        self,
        model_name,
        dataset_name,
        train_batch_size,
        eval_batch_size,
        gradient_accumulation_steps,
        num_epochs,
        learning_rate,
        max_input_length,
        max_target_length,
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=None,  # if None, good defaults for LLaMA
        lora_bias="none",          # "none" | "lora_only" | "all"
    ):
        # Configs
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LoRA configs
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.lora_bias = lora_bias

        # Setup
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None

    def preprocess_dataset(self, subsample_size, seed=101):
        dataset = load_dataset(self.dataset_name)

        if subsample_size is not None:
            dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(subsample_size))
            dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(max(subsample_size // 10, 1)))

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

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="cuda:0")
        self.model = self.model.to(torch.float16)
        self.model.gradient_checkpointing_enable()

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
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for step, batch in enumerate(self.train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                running_loss += loss.item()
                if step % 500 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss {running_loss / (step+1):.4f}")


        # Save LoRA adapters only
        adapter_dir = "./llama3-8b-instruct-squad-dp-lora"
        self.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

    def evaluate(self):
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
    # Model Configs
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name = "squad"
    train_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 8
    num_epochs = 5
    learning_rate = 2e-4
    max_input_length = 512
    max_target_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_size = 5000

    baseline_model = Baseline(
        model_name=model_name,
        dataset_name=dataset_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    # Start GPU utilization logging using utils
    gpu_util_thread, gpu_util_stop_event, gpu_util_data = start_gpu_utilization_logging(interval=1.0)

    baseline_model.preprocess_dataset(subsample_size=sample_size, seed=101)
    baseline_model.init_model()
    baseline_model.train()

    print(f"Model: {model_name}")
    print(f"On device: {device}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Eval batch size: {eval_batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max input length: {max_input_length}")
    print(f"Max target length: {max_target_length}")
    print(f"Traing size: {sample_size}")

    baseline_model.evaluate()

    # Output GPU logging
    stop_gpu_utilization_logging(gpu_util_thread, gpu_util_stop_event)
    print_gpu_utilization_summary(gpu_util_data)
