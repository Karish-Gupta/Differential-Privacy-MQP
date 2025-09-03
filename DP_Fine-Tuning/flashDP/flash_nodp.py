# Training Llama3 without Differential Privacy using HuggingFace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flashdp'))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# --- Config Section ---
MODEL_NAME = "mlabonne/Meta-Llama-3-8B"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-5

# --- Data Section ---
class SquadTextDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=256):
        self.data = load_dataset("squad", split=f"{split}[:200]")
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

# --- Model Loading ---
def load_llama3(model_name, device):
    from transformers import pipeline
    try:
        _ = pipeline("text-generation", model=model_name)   
    except Exception as e:
        print(f"\nERROR: Unable to access model '{model_name}'.\n"
              f"Make sure you have accepted the license at https://huggingface.co/{model_name} and are logged in.\n"
              f"Original error: {e}\n"
              )
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use HuggingFace device_map="auto" for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = model.to(torch.float32)
    tokenizer.pad_token = tokenizer.eos_token
    # Do NOT move model to device or wrap with DataParallel
    return model, tokenizer

# --- Training Loop ---
def train(model, dataloader, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            # Do NOT move x, y to device; let HF handle device placement
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    print(f"Training completed (no Differential Privacy).")

# --- Evaluation/Inference Section (SQuAD QA integration) ---
def evaluate(model, tokenizer, device):
    squad_dataset = load_dataset("squad", split="validation[:100]")

    def preprocess_function(examples):
        prompts = []
        for question, context in zip(examples["question"], examples["context"]):
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            prompts.append(prompt)
        return {"prompt": prompts}

    processed_dataset = squad_dataset.map(preprocess_function, batched=True)

    model.eval()
    losses = []
    predictions = []
    references = []
    for example in processed_dataset:
        # Do NOT move inputs to device; let HF handle device placement
        inputs = tokenizer(example["prompt"], return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            lm_inputs = tokenizer(example["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=inputs["input_ids"].shape[1])
            labels = lm_inputs["input_ids"]
            lm_out = model(input_ids=labels, labels=labels)
            loss = lm_out.loss.item()
            losses.append(loss)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        predictions.append(answer.lower())
        references.append(example.get("answers", {}).get("text", [""])[0].lower() if "answers" in example else "")

    avg_loss = sum(losses) / len(losses) if losses else float('nan')
    accuracy = accuracy_score(references, predictions)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Exact Match Accuracy: {accuracy:.4f}")
    for i in range(min(3, len(predictions))):
        print(f"Q: {processed_dataset[i]['question']}")
        print(f"True: {references[i]}")
        print(f"Pred: {predictions[i]}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: This script requires a CUDA-enabled GPU to run.")
        sys.exit(1)
    print(f"Using {torch.cuda.device_count()} GPU(s).")
    model, tokenizer = load_llama3(MODEL_NAME, device)
    dataset = SquadTextDataset(tokenizer, split="train", max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train(model, dataloader, optimizer, device, epochs=EPOCHS)
    evaluate(model, tokenizer, device)

if __name__ == "__main__":
    main()
    evaluate(model, tokenizer, device)

if __name__ == "__main__":
    main()
