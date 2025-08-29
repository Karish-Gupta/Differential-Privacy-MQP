# Training Llama2 with Differential Privacy using FlashDP and HuggingFace
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'flashdp'))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashdp.api.wrap_model import wrap_with_flashdp_layers
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# --- Config Section ---
MODEL_NAME = "mlabonne/Meta-Llama-3-8B"
MAX_LENGTH = 256        # Increase sequence length for more realistic training
BATCH_SIZE = 4          # Increase batch size for better throughput
EPOCHS = 3              # Train for more epochs
LR = 1e-5
DP_C = 1.0
DP_NOISE = 1.0

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
            # Concatenate question and context for LM
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

# --- Model Loading and Wrapping ---
def load_llama2_with_flashdp(model_name, device):
    # Ensure HuggingFace authentication for gated models like Llama2
    # Run `huggingface-cli login` in your shell or use the following in code:
    # from huggingface_hub import login
    # login(token="YOUR_HF_TOKEN")
    # Quick access check using pipeline (will raise if not authenticated)
    from transformers import pipeline
    try:
        _ = pipeline("text-generation", model=model_name)
    except Exception as e:
        print(f"\nERROR: Unable to access model '{model_name}'.\n"
              f"Make sure you have accepted the license at https://huggingface.co/{model_name} and are logged in.\n"
              f"Original error: {e}\n"
              )
        sys.exit(1)
    # If access is all ok, proceed to load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:0")  # Force all weights to cuda:0
    model = model.to(torch.float32)
    tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model = wrap_with_flashdp_layers(
        model,
        target_modules=[torch.nn.Linear, torch.nn.LayerNorm],
        skip_layers=[],
        C=DP_C,
        noise_multiplier=DP_NOISE
    )
    return model, tokenizer

# --- Training Loop ---
def train(model, dataloader, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        # Optionally: evaluate on validation set here for utility

    # --- Privacy accounting (example, adjust for your DP engine) ---
    # If FlashDP provides a privacy accountant, use it here.
    # Otherwise, print the DP parameters used:
    print(f"DP Training completed with noise_multiplier={DP_NOISE}, C={DP_C}, epochs={epochs}, batch_size={BATCH_SIZE}")
    # If you have a function to compute epsilon, print it:
    # epsilon = compute_epsilon(...)
    # print(f"Achieved (ε, δ)-DP with ε={epsilon:.2f}, δ=1e-5")

# --- Evaluation/Inference Section (SQuAD QA integration) ---
def evaluate(model, tokenizer, device):
    # Use a larger validation set for better metrics
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
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            # For loss calculation, get model output logits
            lm_inputs = tokenizer(example["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=inputs["input_ids"].shape[1]).to(device)
            labels = lm_inputs["input_ids"]
            lm_out = model(input_ids=labels, labels=labels)
            loss = lm_out.loss.item()
            losses.append(loss)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
        predictions.append(answer.lower())
        # Use the first reference answer from SQuAD
        references.append(example.get("answers", {}).get("text", [""])[0].lower() if "answers" in example else "")

    avg_loss = sum(losses) / len(losses) if losses else float('nan')
    # Simple exact match accuracy
    accuracy = accuracy_score(references, predictions)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Exact Match Accuracy: {accuracy:.4f}")
    # Optionally print a few predictions for qualitative analysis
    for i in range(min(3, len(predictions))):
        print(f"Q: {processed_dataset[i]['question']}")
        print(f"True: {references[i]}")
        print(f"Pred: {predictions[i]}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: FlashDP requires a CUDA-enabled GPU to run (Triton kernel error: 0 active drivers).")
        print("Please run this script on a machine with a CUDA GPU and the correct CUDA drivers installed.")
        sys.exit(1)
    model, tokenizer = load_llama2_with_flashdp(MODEL_NAME, device)
    dataset = SquadTextDataset(tokenizer, split="train", max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    train(model, dataloader, optimizer, device, epochs=EPOCHS)
    evaluate(model, tokenizer, device)

if __name__ == "__main__":
    main()
    main()
    main()
