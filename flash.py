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
from sklearn.metrics import accuracy_score
from flashdp.api.wrap_model import wrap_with_flashdp_layers

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
        dp_noise,
    ):
        self.model_name = model_name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.dp_c = dp_c
        self.dp_noise = dp_noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def preprocess_dataset(self):
        class SquadTextDataset(Dataset):
            def __init__(self, tokenizer, split="train", max_length=128):
                self.data = load_dataset("squad", split=f"{split}[:20]")
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.samples = self._preprocess()

            def _preprocess(self):
                samples = []
                for item in self.data:
                    text = f"question: {item['question']} context: {item['context']} answer: {item['answers']['text'][0] if item['answers']['text'] else ''}"
                    tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True, padding="max_length")
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
        train_dataset = SquadTextDataset(self.tokenizer, split="train", max_length=128)
        self.train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

        # Validation loader
        val_dataset = SquadTextDataset(self.tokenizer, split="validation", max_length=128)
        self.val_loader = DataLoader(val_dataset, batch_size=2)

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
        self.model = wrap_with_flashdp_layers(
            self.model,
            target_modules=[torch.nn.Linear],  # Removed torch.nn.LayerNorm
            skip_layers=[],
            C=self.dp_c,
            noise_multiplier=self.dp_noise
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

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
        print(f"DP Training completed with noise_multiplier={self.dp_noise}, C={self.dp_c}, epochs={self.num_epochs}, batch_size={self.train_batch_size}")

    def evaluate(self):
        squad_dataset = load_dataset("squad", split="validation[:20]")

        def preprocess_function(examples):
            prompts = []
            for question, context in zip(examples["question"], examples["context"]):
                prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
                prompts.append(prompt)
            return {"prompt": prompts}

        processed_dataset = squad_dataset.map(preprocess_function, batched=True)

        self.model.eval()
        losses = []
        predictions = []
        references = []
        contains_correct = []
        for example in processed_dataset:
            inputs = self.tokenizer(example["prompt"], return_tensors="pt")
            # Move input_ids to model device before generate
            model_device = next(self.model.parameters()).device
            for k in inputs:
                inputs[k] = inputs[k].to(model_device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
                lm_inputs = self.tokenizer(
                    example["prompt"], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=inputs["input_ids"].shape[1]
                )
                for k in lm_inputs:
                    lm_inputs[k] = lm_inputs[k].to(model_device)
                labels = lm_inputs["input_ids"]
                lm_out = self.model(input_ids=labels, labels=labels)
                loss = lm_out.loss.item()
                losses.append(loss)
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Answer:" in full_output:
                answer = full_output.split("Answer:")[-1].strip()
            else:
                answer = full_output.strip()
            predictions.append(answer.lower())
            ref = example.get("answers", {}).get("text", [""])[0].lower() if "answers" in example else ""
            references.append(ref)
            contains_correct.append(ref in answer.lower())

        avg_loss = sum(losses) / len(losses) if losses else float('nan')
        accuracy = accuracy_score(references, predictions)
        contains_accuracy = sum(contains_correct) / len(contains_correct) if contains_correct else float('nan')
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Exact Match Accuracy: {accuracy:.4f}")
        print(f"Validation Contains Accuracy: {contains_accuracy:.4f}")
        for i in range(min(3, len(predictions))):
            print(f"Q: {processed_dataset[i]['question']}")
            print(f"True: {references[i]}")
            print(f"Pred: {predictions[i]}\n")

if __name__ == "__main__":
    # Configs
    model_name = "mlabonne/Meta-Llama-3-8B"
    train_batch_size = 2
    eval_batch_size = 2
    num_epochs = 1
    learning_rate = 1e-5
    max_length = 128
    dp_c = 1.0
    dp_noise = 1.0

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
    )
    flashdp.preprocess_dataset()
    flashdp.init_model()
    flashdp.train()
    flashdp.evaluate()