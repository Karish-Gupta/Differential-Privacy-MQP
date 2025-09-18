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
    f1_scores = []
    all_preds, all_refs = [], []
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        with torch.no_grad():
            # Find where each input actually ends (excluding padding)
            input_lengths = attention_mask.sum(dim=1).cpu()
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_length,
                pad_token_id=tokenizer.pad_token_id,  # Use pad_token_id
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,  # Temperature=0 is not valid, use 1.0 for greedy
            )
        
        # Extract generated tokens and gold answers properly
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            # Get actual input length for this sample
            input_len = input_lengths[i].item()
            
            # Extract only generated tokens (after the input)
            generated_ids = outputs[i, input_len:]
            
            # Remove padding and EOS tokens from generation
            generated_ids = generated_ids[generated_ids != tokenizer.pad_token_id]
            if len(generated_ids) > 0 and generated_ids[-1] == tokenizer.eos_token_id:
                generated_ids = generated_ids[:-1]
            
            pred = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Extract gold answer from labels
            # Get non-masked labels after the input
            label_seq = labels[i].tolist()
            
            # Find where the answer starts (first non -100 after some -100s)
            answer_start = 0
            for idx, l in enumerate(label_seq):
                if l == -100:
                    answer_start = idx + 1
                else:
                    break
            
            # Extract answer tokens (non-padding, non-masked)
            answer_tokens = []
            for l in label_seq[answer_start:]:
                if l != -100 and l != tokenizer.pad_token_id:
                    answer_tokens.append(l)
                elif l == tokenizer.eos_token_id:
                    break  # Stop at EOS
            
            # Decode the gold answer
            ref = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            all_preds.append(pred)
            all_refs.append(ref)
            
            # Calculate metrics
            pred_norm = pred.lower().strip()
            ref_norm = ref.lower().strip()
            
            # Exact match
            exact_match_scores.append(1 if pred_norm == ref_norm else 0)
            
            # Contains accuracy
            contains_scores.append(1 if ref_norm in pred_norm or pred_norm in ref_norm else 0)
            
            # Calculate F1 for this example (word-level)
            pred_tokens = pred_norm.split()
            ref_tokens = ref_norm.split()
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                f1_scores.append(0.0)
            else:
                common_tokens = set(pred_tokens) & set(ref_tokens)
                num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common_tokens)
                
                if num_common == 0:
                    f1_scores.append(0.0)
                else:
                    precision = num_common / len(pred_tokens)
                    recall = num_common / len(ref_tokens)
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_scores.append(f1)
    
    # Aggregate metrics
    contains_accuracy = np.mean(contains_scores)
    exact_match_accuracy = np.mean(exact_match_scores)
    f1 = np.mean(f1_scores)
    
    print(f"\nContains Accuracy: {contains_accuracy:.4f}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
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