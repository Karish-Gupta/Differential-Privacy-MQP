import numpy as np
import torch
import tqdm

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


def evaluate_exact_match(model, val_loader, device, tokenizer, max_gen_length=50):
    model.eval()
    em_scores = []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels = batch["labels"]

        # Convert labels back to text (ignore -100 padding)
        decoded_labels = []
        for label_ids in labels:
            label_ids = [l for l in label_ids.tolist() if l != -100]
            text = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(text)

        # Compare predictions to references
        for pred, ref in zip(preds, decoded_labels):
            pred_norm = pred.strip().lower()
            ref_norm = ref.strip().lower()

            em = 1 if contains_token(pred_norm, ref_norm) else 0
            em_scores.append(em)

    exact_match = np.mean(em_scores)
    print(f"Exact Match Accuracy: {exact_match:.4f}")
    return exact_match
