import numpy as np
import torch
from tqdm import tqdm
import random
from sklearn.metrics import f1_score
import threading
import time
import csv
import subprocess

def exact_match(pred, ref):
    """
    Returns True if prediction exactly matches reference
    """
    return pred.strip().lower() == ref.strip().lower()

def contains_token(pred, ref):
    """
    Returns True if the reference tokens appear in order inside prediction tokens
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


def evaluate_model(model, val_loader, device, tokenizer, max_gen_length=50, show_samples=20, seed=42):
    model.eval()
    contains_scores = []
    exact_match_scores = []
    all_preds, all_refs = [], []

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            # For decoder-only models like Llama, we need to generate differently
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.eos_token_id,  # Important for Llama
                eos_token_id=tokenizer.eos_token_id,  # Important for Llama
                do_sample=False,  # Use greedy decoding for consistency
                temperature=0,
            )
        
        # For decoder models, the output includes the input + generated text
        # We need to extract only the generated part
        generated_ids = outputs[:, input_ids.shape[1]:]  # Take only the new tokens
        
        # Decode and clean predictions
        preds = [p for p in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]

        # Decode and clean gold labels
        decoded_labels = []
        for label_ids in labels:
            label_ids = [l for l in label_ids.tolist() if l != -100]
            text = tokenizer.decode(label_ids, skip_special_tokens=True)
            decoded_labels.append(text)

        # Collect for F1
        all_preds.extend(preds)
        all_refs.extend(decoded_labels)

        # Compute EM and contains acc
        for pred, ref in zip(preds, decoded_labels):
            pred_norm = pred.lower()
            ref_norm = ref.lower()
            
            # Contains accuracy
            contains = 1 if contains_token(pred_norm, ref_norm) else 0
            contains_scores.append(contains)
            
            # Exact match
            em = 1 if exact_match(pred, ref) else 0
            exact_match_scores.append(em)
            
        # print(f"Input shape: {input_ids.shape}")
        # print(f"Output shape: {outputs.shape}")
        # print(f"Input: {input_ids}")
        # print(f"Output: {outputs}")
        # print(f"Generated tokens shape: {generated_ids.shape}")
        # print(f"Generated tokens: {generated_ids}")

    # Aggregate metrics
    contains_accuracy = np.mean(contains_scores)
    exact_match_accuracy = np.mean(exact_match_scores)

    pred_tokens = [p.lower().split() for p in all_preds]
    ref_tokens = [r.lower().split() for r in all_refs]

    y_true, y_pred = [], []
    for ref, pred in zip(ref_tokens, pred_tokens):
        ref_set = set(ref)
        pred_set = set(pred)
        all_tokens = list(ref_set | pred_set)
        for token in all_tokens:
            y_true.append(token in ref_set)
            y_pred.append(token in pred_set)

    f1 = f1_score(y_true, y_pred)

    print(f"\nContains Accuracy: {contains_accuracy:.4f}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Token-level F1 Score: {f1:.4f}")

    # Show sample predictions
    if show_samples > 0:
        print("\nSample predictions:\n")
        random.seed(seed)
        indices = random.sample(range(len(all_preds)), min(show_samples, len(all_preds)))
        for i in indices:
            print("=" * 80)
            print(f"Gold Answer: {all_refs[i]}")
            print(f"Predicted Answer: {all_preds[i]}")

    return {
        "contains_accuracy": contains_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "f1": f1
    }
    
def start_gpu_utilization_logging(logfile="gpu_utilization_debug.csv", interval=1.0, util_data=None, stop_event=None):
    """
    Start a background thread to log GPU utilization every `interval` seconds.
    Returns (thread, stop_event, util_data).
    """
    if stop_event is None:
        stop_event = threading.Event()
    if util_data is None:
        util_data = []
    def _gpu_utilization_logger():
        # Write CSV header
        with open(logfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_index", "gpu_name", "utilization", "mem_used", "mem_total"])
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
                )
                timestamp = time.time()
                lines = result.stdout.strip().split('\n')
                with open(logfile, "a", newline="") as f:
                    writer = csv.writer(f)
                    for line in lines:
                        idx, name, util, mem_used, mem_total = [x.strip() for x in line.split(',')]
                        writer.writerow([timestamp, idx, name, util, mem_used, mem_total])
                        util_data.append((float(idx), float(util)))
            except Exception:
                pass
            time.sleep(interval)
    thread = threading.Thread(target=_gpu_utilization_logger, daemon=True)
    thread.start()
    return thread, stop_event, util_data

def stop_gpu_utilization_logging(thread, stop_event):
    """
    Stop the background GPU utilization logging thread.
    """
    stop_event.set()
    if thread is not None:
        thread.join()

def print_gpu_utilization_summary(util_data):
    """
    Compute and print the average GPU utilization for each GPU and overall.
    """
    if not util_data:
        print("No GPU utilization data collected.")
        return
    from collections import defaultdict
    util_per_gpu = defaultdict(list)
    for idx, util in util_data:
        util_per_gpu[idx].append(util)
    print("\n[GPU UTILIZATION SUMMARY]")
    total_sum = 0
    total_count = 0
    for idx in sorted(util_per_gpu.keys()):
        vals = util_per_gpu[idx]
        avg = sum(vals) / len(vals)
        print(f"GPU {int(idx)}: Average Utilization {avg:.2f}% over {len(vals)} samples")
        total_sum += sum(vals)
        total_count += len(vals)
    if total_count > 0:
        overall_avg = total_sum / total_count
        print(f"Overall Average Utilization: {overall_avg:.2f}%")
    print()