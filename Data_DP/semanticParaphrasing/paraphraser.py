import numpy as np
import torch
import tqdm
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset
from huggingface_hub import login
import os

# Login to HF CLI if token is available
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
transformers.utils.logging.set_verbosity_debug()

class StandardModel:
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

        # Setup
        self.dataset = None
        self.tokenizer = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None

    def preprocess_dataset(self):
        dataset = load_dataset(self.dataset_name)

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
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length"
            )
            labels = self.tokenizer(
                example["target_text"],
                max_length=self.max_target_length,
                truncation=True,
                padding="max_length"
            )
            labels["input_ids"] = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        self.dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)

        self.train_loader = DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=default_data_collator
        )
        self.val_loader = DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
            collate_fn=default_data_collator
        )

    def init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            offload_folder="offload"
        )
        self.model.gradient_checkpointing_enable()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

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
                if step % 50 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss {running_loss / (step+1):.4f}")

        # Save model + tokenizer once at the end
        self.model.save_pretrained("./llama3-8b-instruct-squad-model")
        self.tokenizer.save_pretrained("./llama3-8b-instruct-squad-tokenizer")

    def evaluate_exact_match(self, max_gen_length=50):
        self.model.eval()
        em_scores = []

        for batch in tqdm.tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Generate predictions
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_gen_length,
                )

            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels = batch["labels"]

            # Convert labels back to text (ignore -100 padding)
            decoded_labels = []
            for label_ids in labels:
                label_ids = [l for l in label_ids.tolist() if l != -100]
                text = self.tokenizer.decode(label_ids, skip_special_tokens=True)
                decoded_labels.append(text)

            # Compare predictions to references
            for pred, ref in zip(preds, decoded_labels):
                pred_norm = pred.strip().lower()
                ref_norm = ref.strip().lower()
                em = 1 if pred_norm == ref_norm else 0
                em_scores.append(em)

        exact_match = np.mean(em_scores)
        print(f"Exact Match Accuracy: {exact_match:.4f}")
        return exact_match




if __name__ == "__main__":
    # Model Configs
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_name = "rajpurkar/squad"
    train_batch_size = 2
    eval_batch_size = 2
    gradient_accumulation_steps = 8
    num_epochs = 1
    learning_rate = 2e-4
    max_input_length = 512
    max_target_length = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Model: {model_name}")
    print(f"On device: {device}")

    trainer = StandardModel(
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

    trainer.preprocess_dataset()
    trainer.init_model()
    trainer.train()
    trainer.evaluate_exact_match()
