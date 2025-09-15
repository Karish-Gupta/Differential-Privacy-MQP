# Training Llama3 with HuggingFace + LoRA + TEM (pypantera)
import sys
import os
import urllib.request
import zipfile

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from datasets import load_dataset
from utils import *  # Import eval functions
from peft import LoraConfig, get_peft_model, TaskType

# pyPANTERA imports
from pypantera.embeddings import EmbeddingLoader
from pypantera.mechanisms import TEM

logging.set_verbosity_error()


class TEMModel:
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
        do_privacy=False,
        privacy_epsilon=1.0,
        privacy_beta=0.5,
        embedding_path=None,
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
        # TEM configs
        self.do_privacy = do_privacy
        self.privacy_epsilon = privacy_epsilon
        self.privacy_beta = privacy_beta
        self.embedding_path = embedding_path
        self.tem = None
        self.emb_loader = None

    def download_glove_if_missing(self):
        """Download GloVe embeddings if not present."""
        target_file = self.embedding_path
        if os.path.exists(target_file):
            print(f"[INFO] GloVe embeddings already available at {target_file}")
            return

        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        zip_path = os.path.join(os.path.dirname(target_file), "glove.6B.zip")

        if not os.path.exists(zip_path):
            url = "https://nlp.stanford.edu/data/glove.6B.zip"
            print(f"[INFO] Downloading GloVe embeddings from {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            print("[INFO] Download complete.")

        print("[INFO] Extracting GloVe embeddings...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(target_file))
        print("[INFO] Extraction complete.")

    def preprocess_dataset(self, subsample_size=None, seed=42):
        dataset = load_dataset(self.dataset_name)

        if subsample_size is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=seed)
                .select(range(subsample_size))
            )
            dataset["validation"] = (
                dataset["validation"]
                .shuffle(seed=seed)
                .select(range(max(1, subsample_size // 10)))
            )

        # Initialize TEM if needed
        if self.do_privacy:
            self.download_glove_if_missing()
            print("Initializing TEM privacy mechanism...")
            self.emb_loader = EmbeddingLoader("glove", dim=300, path=self.embedding_path)
            self.tem = TEM(
                self.emb_loader,
                epsilon=self.privacy_epsilon,
                beta=self.privacy_beta,
                distance_metric="cosine",
            )

        def obfuscate_text(text):
            words = text.split()
            obf_words = []
            for w in words:
                if self.emb_loader and self.emb_loader.has_embedding(w):
                    obf_words.append(self.tem.obfuscate(w))
                else:
                    obf_words.append(w)
            return " ".join(obf_words)

        def preprocess(example):
            inp = "Question: " + example["question"] + " Context: " + example["context"]
            if self.do_privacy and self.tem is not None:
                inp = obfuscate_text(inp)
            tgt = example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else ""
            example["input_text"] = inp
            example["target_text"] = tgt
            return example

        dataset = dataset.map(preprocess)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        def tokenize(batch):
            inputs = self.tokenizer(
                batch["input_text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = self.tokenizer(
                batch["target_text"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels["input_ids"],
            }

        tokenized_train = dataset["train"].map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
        tokenized_val = dataset["validation"].map(tokenize, batched=True, remove_columns=dataset["validation"].column_names)

        self.train_loader = DataLoader(tokenized_train, batch_size=self.train_batch_size, shuffle=True)
        self.val_loader = DataLoader(tokenized_val, batch_size=self.eval_batch_size)

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
            for batch in self.train_loader:
                x = batch["input_ids"].to(model_device)
                y = batch["labels"].to(model_device)
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
        print("Evaluating with Exact Match and F1 metrics from utils.py...")
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
    # Configs
    model_name = "mlabonne/Meta-Llama-3-8B"
    dataset_name = "rajpurkar/squad"
    train_batch_size = 2
    eval_batch_size = 2
    num_epochs = 3
    learning_rate = 2e-4
    max_length = 512

    # LoRA configs
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = None
    lora_bias = "none"

    # TEM configs
    do_privacy = True
    privacy_epsilon = 1.0
    privacy_beta = 0.5
    embedding_path = "/glove/glove.6B.300d.txt"  # update this path

    if torch.cuda.device_count() == 0:
        print("ERROR: CUDA GPU required.")
        sys.exit(1)
    print(f"Using {torch.cuda.device_count()} GPU(s).")

    trainer = TEMModel(
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
        do_privacy=do_privacy,
        privacy_epsilon=privacy_epsilon,
        privacy_beta=privacy_beta,
        embedding_path=embedding_path,
    )
    trainer.preprocess_dataset(subsample_size=5000, seed=101)
    trainer.init_model()
    trainer.train()
    trainer.evaluate()
