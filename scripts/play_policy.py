"""Watch a trained policy in the SAPIEN viewer: one env on the CPU sim, so it can run next to training.

Usage: python scripts/play_policy.py --ckpt runs/<name>/best.pt [--episodes 5] [--seed 0]
"""
import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch

import so101_rl.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from train_ppo import Agent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = gym.make("SO101PickPlace-v1", num_envs=1, obs_mode="state", sim_backend="cpu",
                   control_mode="pd_joint_delta_pos", render_mode="human")
    ck = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    agent = Agent(ck["n_obs"], ck["n_act"])
    agent.load_state_dict(ck["agent"])
    agent.eval()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        success = False
        for _ in range(200):
            with torch.no_grad():
                action = agent.get_action(obs.float(), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            success |= bool(info["success"].any())
            env.render()
            time.sleep(0.05)  # 20 Hz control, so it plays at real speed
            if terminated.any() or truncated.any():
                break
        print(f"episode {ep}: success={success} grasped_at_end={bool(info['is_grasped'].any())}")
    env.close()


if __name__ == "__main__":
    main()
