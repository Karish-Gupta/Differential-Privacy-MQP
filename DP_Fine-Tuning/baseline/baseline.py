# Training Llama3 with HuggingFace + LoRA (no FlashDP)
import sys
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from datasets import load_dataset
from utils import evaluate_exact_match  # Importing eval function from utils.py
from peft import LoraConfig, get_peft_model, TaskType

logging.set_verbosity_error()


class BasicLoRAModel:
    def __init__(
        self,
        model_name,
        train_batch_size,
        eval_batch_size,
        num_epochs,
        learning_rate,
        max_length,
        lora_r=8,
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

    def preprocess_dataset(self):
        class SquadTextDataset(Dataset):
            def __init__(self, tokenizer, split="train", max_length=128):
                squad = load_dataset("squad")
                if split == "train":
                    self.data = squad["train"].select(range(2500))
                else:
                    self.data = squad["validation"].select(range(50))
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.samples = self._preprocess()

            def _preprocess(self):
                samples = []
                for item in self.data:
                    text = f"question: {item['question']} context: {item['context']} answer: {item['answers']['text'][0] if item['answers']['text'] else ''}"
                    tokens = self.tokenizer.encode(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                    )
                    samples.append(tokens)
                return samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                x = torch.tensor(self.samples[idx][:-1], dtype=torch.long)
                y = torch.tensor(self.samples[idx][1:], dtype=torch.long)
                return x, y

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Train loader
        train_dataset = SquadTextDataset(
            self.tokenizer, split="train", max_length=self.max_length
        )
        self.train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)

        # Validation loader
        val_dataset = SquadTextDataset(
            self.tokenizer, split="validation", max_length=self.max_length
        )
        self.val_loader = DataLoader(val_dataset, batch_size=self.eval_batch_size)

    def init_model(self):
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        self.model = self.model.to(torch.float32)

        # LoRA config
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

        # Optimizer
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    def train(self):
        self.model.train()
        model_device = next(self.model.parameters()).device
        for epoch in range(self.num_epochs):
            total_loss = 0
            for x, y in self.train_loader:
                x, y = x.to(model_device), y.to(model_device)
                outputs = self.model(input_ids=x, labels=y)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Save LoRA adapters only
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


if __name__ == "__main__":
    # Configs
    model_name = "mlabonne/Meta-Llama-3-8B"
    train_batch_size = 2
    eval_batch_size = 2
    num_epochs = 1
    learning_rate = 1e-5
    max_length = 128

    # LoRA configs
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
    trainer.preprocess_dataset()
    trainer.init_model()
    trainer.train()
    trainer.evaluate()
