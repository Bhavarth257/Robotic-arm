"""M0 smoke test: run PickCube-v1 on the GPU sim with random actions, report FPS and peak VRAM."""
import argparse
import os
import subprocess
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401  registers envs
import torch


def gpu_mem_mib():
    # nvidia-smi sees PhysX memory too; torch.cuda.max_memory_allocated would not.
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    for line in out.strip().splitlines():
        pid, mem = (x.strip() for x in line.split(","))
        if int(pid) == os.getpid():
            return int(mem)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print("GPU:", torch.cuda.get_device_name(0))

    env = gym.make("PickCube-v1", num_envs=args.num_envs, obs_mode="state", sim_backend="gpu")
    env.reset(seed=args.seed)
    low = torch.as_tensor(env.action_space.low, device="cuda")
    high = torch.as_tensor(env.action_space.high, device="cuda")
    torch.manual_seed(args.seed)

    peak = gpu_mem_mib()
    start = time.perf_counter()
    for i in range(args.steps):
        # Sample on GPU: a CPU sample + copy every step would dominate the timing.
        action = low + (high - low) * torch.rand_like(low)
        env.step(action)
        if i % 100 == 0:
            peak = max(peak, gpu_mem_mib())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = max(peak, gpu_mem_mib())
    env.close()

    print(f"envs={args.num_envs} steps={args.steps} time={elapsed:.1f}s "
          f"FPS={args.num_envs * args.steps / elapsed:.0f} peak_VRAM={peak} MiB")


if __name__ == "__main__":
    main()
