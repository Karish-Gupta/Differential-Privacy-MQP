import numpy as np
import torch
import tqdm

def evaluate_exact_match(model, val_loader, device, tokenizer, max_gen_length=50):
    model.eval()
    em_scores = []
    losses = []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Compute loss
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss.item()
            losses.append(loss)

        # Generate predictions
        with torch.no_grad():
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
            )

        preds = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        # Convert labels back to text (ignore -100 padding)
        decoded_labels = []
        for label_ids in labels:
            label_ids = [l.item() for l in label_ids if l.item() != -100]
            text = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(text)

        # Exact match
        for pred, ref in zip(preds, decoded_labels):
            pred_norm = pred.strip().lower()
            ref_norm = ref.strip().lower()
            em = 1 if contains_token(pred_norm, ref_norm) else 0
            em_scores.append(em)

    # Aggregate metrics
    avg_loss = np.mean(losses) if losses else float("nan")
    exact_match = np.mean(em_scores) if em_scores else float("nan")

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Exact Match Accuracy (token-level containment): {exact_match:.4f}")

    return {"loss": avg_loss, "exact_match": exact_match}


def contains_token(pred, ref):
    """
    Returns True if the reference tokens appear in order inside prediction tokens.
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    if not ref_tokens:
        return False

    # Sliding window check
    for i in range(len(pred_tokens) - len(ref_tokens) + 1):
        if pred_tokens[i:i+len(ref_tokens)] == ref_tokens:
            return True
    return False