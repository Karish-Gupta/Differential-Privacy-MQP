import sys
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.utils import logging
from datasets import load_dataset
from utils import evaluate_exact_match, evaluate_f1
from peft import LoraConfig, get_peft_model, TaskType

logging.set_verbosity_error()

class BasicLoRAModel:
    def __init__(
        self,
        model_name,
        dataset_name,
        train_batch_size,
        eval_batch_size,
        num_epochs,
        learning_rate,
        max_length,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
        lora_bias="none",
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
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

    def preprocess_dataset(self, subsample_size, seed=42):
        dataset = load_dataset(self.dataset_name)

        if subsample_size is not None:
            dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(subsample_size))
            dataset["validation"] = dataset["validation"].shuffle(seed=seed).select(range(subsample_size // 10)) 

        def preprocess(example):
            example["input_text"] = "Question: " + example["question"] + " Context: " + example["context"]
            example["target_text"] = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
            return example

        dataset = dataset.map(preprocess)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        def tokenize(example):
            model_inputs = self.tokenizer(
                example["input_text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
            labels = self.tokenizer(
                example["target_text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length"
            )
            labels["input_ids"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_train = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_val = dataset["validation"].map(tokenize, batched=True, remove_columns=dataset["validation"].column_names)

        self.train_loader = DataLoader(tokenized_train, batch_size=self.train_batch_size, shuffle=True, collate_fn=default_data_collator)
        self.val_loader = DataLoader(tokenized_val, batch_size=self.eval_batch_size, collate_fn=default_data_collator)

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.model = self.model.to(torch.float32)

        target_modules = self.lora_target_modules or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
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
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                running_loss += loss.item()
                if step % 50 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss {running_loss / (step+1):.4f}")

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

        adapter_dir = "./llama3-8b-lora"
        self.model.save_pretrained(adapter_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(adapter_dir)

    def evaluate(self):
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
            max_gen_length=30,
        )
        evaluate_f1(
            self.model,
            self.val_loader,
            model_device,
            self.tokenizer,
            max_gen_length=30,
        )

if __name__ == "__main__":
    model_name = "mlabonne/Meta-Llama-3-8B"
    dataset_name = "rajpurkar/squad"
    train_batch_size = 2
    eval_batch_size = 2
    num_epochs = 3
    learning_rate = 2e-4
    max_length = 512

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = None
    lora_bias = "none"

    if torch.cuda.device_count() == 0:
        print("ERROR: CUDA GPU required.")
        sys.exit(1)
    print(f"Using {torch.cuda.device_count()} GPU(s).")

    trainer = BasicLoRAModel(
        model_name=model_name,
        dataset_name=dataset_name,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        lora_bias=lora_bias,
    )
    trainer.preprocess_dataset(subsample_size=5000, seed=101)
    trainer.init_model()
    trainer.train()
    trainer.evaluate()