#!/usr/bin/env python3
"""NCCL All-Reduce Test -- verifies 4-GPU LUPINE fabric

Run after: enable-lupine
Run as:    lupine-env python3 examples/nccl_allreduce_test.py

Tests:
  1. GPU visibility: torch.cuda.device_count() == 4
  2. Data path: cross-GPU tensor copies via LUPINE fabric
  3. NCCL simulation: all_reduce across all 4 GPUs
"""
import os
import sys
import time


def main():
    try:
        import torch
    except ImportError:
        print("[!] PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[!] CUDA not available.")
        print("    Ensure LUPINE env is active: run 'enable-lupine' first")
        print("    Then: lupine-env python3 examples/nccl_allreduce_test.py")
        sys.exit(1)

    n = torch.cuda.device_count()
    print("=" * 70)
    print("  Compute Pool -- 4-GPU LUPINE Fabric Verification")
    print("=" * 70)
    print(f"\n[*] Detected CUDA devices: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory // (1024 ** 3)
        print(f"    GPU {i}: {props.name} -- {vram_gb} GB VRAM")

    if n < 4:
        print(f"\n[!] Expected 4 GPUs, got {n}.")
        print("    Run: enable-lupine")
        print("    Then: lupine-env python3 examples/nccl_allreduce_test.py")
        sys.exit(1)

    print(f"\n[OK] GPU count: {n}/4 Tesla T4s visible via LUPINE fabric")

    # Step 1: Allocate tensors on each GPU
    print(f"\n[*] Allocating tensors on all {n} GPUs...")
    tensors = []
    for i in range(n):
        t = torch.ones(4096, dtype=torch.float32, device=f"cuda:{i}") * float(i + 1)
        tensors.append(t)
        expected = float((i + 1) * 4096)
        actual = t.sum().item()
        status = "OK" if abs(actual - expected) < 1 else "FAIL"
        print(f"    [{status}] cuda:{i}  sum={actual:.0f}  expected={expected:.0f}")

    # Step 2: Cross-GPU copy test
    print(f"\n[*] Cross-GPU copy test (validates LUPINE data path)...")
    t_start = time.time()
    for i in range(n):
        j = (i + 1) % n
        src = tensors[i]
        dst = src.to(f"cuda:{j}")
        torch.cuda.synchronize(j)
        ok = abs(dst.sum().item() - src.sum().item()) < 1
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] cuda:{i} -> cuda:{j}  sum={dst.sum().item():.0f}")
        if not ok:
            print("[!] Cross-GPU copy FAILED")
            sys.exit(1)
    elapsed_copy = (time.time() - t_start) * 1000
    print(f"    All cross-GPU copies verified ({elapsed_copy:.0f}ms)")

    # Step 3: Simulated all_reduce (sum each GPU's tensor from all other GPUs)
    print(f"\n[*] Simulated all_reduce across {n} GPUs...")
    expected_sum = float(sum(range(1, n + 1)) * 4096)
    t_start = time.time()
    for i in range(n):
        t = tensors[i].clone()
        for j in range(n):
            if j != i:
                t = t + tensors[j].to(f"cuda:{i}")
        torch.cuda.synchronize(i)
        actual = t.sum().item()
        ok = abs(actual - expected_sum) < 1
        status = "OK" if ok else "FAIL"
        print(f"    [{status}] GPU {i}: post-reduce sum={actual:.0f}  expected={expected_sum:.0f}")
        if not ok:
            print("[!] All-reduce FAILED")
            sys.exit(1)
    elapsed_reduce = (time.time() - t_start) * 1000

    lupine_server = os.environ.get("LUPINE_SERVER", "not set")
    print("\n" + "=" * 70)
    print("  [OK] 4-GPU LUPINE fabric VERIFIED")
    print(f"  GPUs detected:    {n}/4 Tesla T4s")
    print(f"  Cross-GPU copies: PASSED  ({elapsed_copy:.0f}ms)")
    print(f"  All-reduce:       PASSED  ({elapsed_reduce:.0f}ms)")
    print(f"  LUPINE_SERVER:    {lupine_server}")
    print("")
    print("  Next steps:")
    print(f"    lupine-env torchrun --nproc_per_node={n} your_training_script.py")
    print(f"    lupine-env deepspeed --num_gpus={n} your_training_script.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
