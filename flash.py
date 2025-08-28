# Training Llama2 with Differential Privacy using FlashDP and HuggingFace

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from flashdp import wrap_with_flashdp_layers
from datasets import load_dataset

# SQuAD v1.1 dataset loader and preprocessor for causal LM
class SquadTextDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=256):
        self.data = load_dataset("squad", split=split)
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
        x = torch.tensor(self.samples[idx][:-1])
        y = torch.tensor(self.samples[idx][1:])
        return x, y

def main():
    model_name = "meta-llama/Llama-2-7b-hf"  # Change to your preferred Llama2 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)

    # Wrap model with FlashDP
    model = wrap_with_flashdp_layers(
        model,
        target_modules=[torch.nn.Linear, torch.nn.LayerNorm],
        skip_layers=[],
        C=1.0,
        noise_multiplier=1.0
    )

    # Prepare SQuAD v1.1 data
    dataset = SquadTextDataset(tokenizer, split="train", max_length=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    for epoch in range(1):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

if __name__ == "__main__":
    main()
