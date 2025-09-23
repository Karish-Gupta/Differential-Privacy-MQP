import threading
import time
import csv
import subprocess

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