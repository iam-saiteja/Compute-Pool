"""Example PyTorch GPU Workload for Compute Pool.

Submit to a single GPU slot with:
    compute-pool jobs submit --script examples/train_pytorch.py --name train-mlp --gpu
Or dispatch 4 independent shards across the 2-node cluster (from the `shell --slot cluster` terminal):
    cp-dispatch "python train_pytorch.py"
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

rank = int(os.environ.get("CP_TASK_INDEX", "0"))
world_size = int(os.environ.get("CP_TASK_COUNT", "1"))
user = os.environ.get("KAGGLE_USERNAME", "local")

print("==================================================")
print(f" [Node {rank}/{world_size}] Worker on {user}")
print(f" CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f" GPU Device     : {gpu_name} (Devices on Node: {gpu_count})")
    print(f" VRAM per GPU   : {vram_gb:.1f} GB")
print("==================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Simple MLP
model = nn.Sequential(
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

N_SAMPLES = 20000 // world_size
BATCH_SIZE = 128
EPOCHS = 5

torch.manual_seed(42 + rank)
X = torch.randn(N_SAMPLES, 512, device=device)
y = torch.randint(0, 10, (N_SAMPLES,), device=device)

print(f"\nTraining on local shard of {N_SAMPLES} samples on {device}...")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    steps = 0
    perm = torch.randperm(X.size(0))
    for i in range(0, X.size(0), BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        bx, by = X[idx], y[idx]
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        steps += 1
    print(f"  Node {rank} | Epoch {epoch}/{EPOCHS} -> Loss: {epoch_loss / steps:.4f}")

elapsed = time.time() - t0
throughput = (N_SAMPLES * EPOCHS) / elapsed

print("\n==================================================")
print(f" Node {rank} Finished in {elapsed:.2f}s ({throughput:.1f} samples/sec)")
print("==================================================")
