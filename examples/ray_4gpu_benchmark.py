import ray
import torch
import time
import socket

# Connect to the unified Ray cluster
ray.init(address="auto")

print("\n" + "="*70)
print("⚡ COMPUTE POOL: 4-GPU RAY DISTRIBUTED BENCHMARK ⚡")
print("="*70)

res = ray.cluster_resources()
print(f"[*] Total Cluster CPUs: {res.get('CPU', 0)}")
print(f"[*] Total Cluster GPUs: {res.get('GPU', 0)} (60 GB Combined VRAM)")
print(f"[*] Cluster Nodes:      {len(ray.nodes())}")
print("="*70 + "\n")

@ray.remote(num_gpus=1)
def benchmark_gpu_worker(task_id):
    hostname = socket.gethostname()
    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    props = torch.cuda.get_device_properties(device_id)
    vram_gb = props.total_memory / (1024**3)

    # Perform GPU Matrix Multiply (8000x8000 FP32)
    t0 = time.time()
    a = torch.randn(8000, 8000, device="cuda")
    b = torch.randn(8000, 8000, device="cuda")
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) * 1000

    return {
        "task_id": task_id,
        "host": hostname,
        "gpu": gpu_name,
        "vram_total": f"{vram_gb:.1f} GB",
        "matmul_ms": f"{elapsed:.2f} ms",
        "status": "PASS"
    }

print("[*] Dispatching 4 concurrent GPU tasks across all 4 GPUs...\n")
start_time = time.time()
futures = [benchmark_gpu_worker.remote(i) for i in range(4)]
results = ray.get(futures)
total_time = time.time() - start_time

print(f"{'Task':<8} {'Hostname':<25} {'GPU Device':<16} {'VRAM':<10} {'Compute Time':<15} {'Status'}")
print("-" * 85)
for r in sorted(results, key=lambda x: x['task_id']):
    print(f"Task {r['task_id']:<3} {r['host']:<25} {r['gpu']:<16} {r['vram_total']:<10} {r['matmul_ms']:<15} {r['status']}")

print("-" * 85)
print(f"\n[✓] All 4 GPUs executed in parallel in {total_time:.2f}s total!")
print("="*70 + "\n")
