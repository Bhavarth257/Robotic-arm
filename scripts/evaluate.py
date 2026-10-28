"""Evaluate a PPO checkpoint on SO101PickPlace-v1: success rate, grasp rate, episode length, videos.

Episodes end at first success (as they would on the real robot); otherwise they run 200 steps.
Usage: python scripts/evaluate.py --ckpt runs/<name>/final.pt [--episodes 500] [--seed 0] [--dr all]
       python scripts/evaluate.py --onnx models/so101_pickplace.onnx --dr all   (exported policy)
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch

import so101_rl.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent))
from train_ppo import Agent  # noqa: E402

STEPS = 200


class OnnxPolicy:
    """Runs an exported actor with ONNX Runtime on the CPU, like the robot will."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def get_action(self, obs, deterministic=True):
        out = self.sess.run(None, {"obs": obs.float().cpu().numpy()})[0]
        return torch.from_numpy(out).to(obs.device)


def rollout(env, agent, seed, frames=False, n_obs=None):
    """Run one batch of episodes; return per-env success, grasped, length (and frames if asked)."""
    n = env.unwrapped.num_envs
    obs, _ = env.reset(seed=seed)
    success = torch.zeros(n, dtype=torch.bool, device="cuda")
    grasped = torch.zeros_like(success)
    length = torch.full((n,), STEPS, device="cuda")
    video = []
    for t in range(STEPS):
        with torch.no_grad():
            # Older checkpoints saw fewer observations; new ones were appended at the end.
            obs, _, _, _, info = env.step(agent.get_action(obs[:, :n_obs], deterministic=True))
        new = info["success"] & ~success
        length[new] = t + 1
        success |= info["success"]
        grasped |= info["is_grasped"]
        if frames:
            video.append(env.render().cpu().numpy())
        if success.all():
            break
    return success.cpu(), grasped.cpu(), length.cpu(), (np.stack(video, 1) if frames else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", help="PyTorch checkpoint")
    p.add_argument("--onnx", help="exported policy instead of --ckpt")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--videos", type=int, default=5)
    p.add_argument("--dr", nargs="*", default=[], help="domain randomisations to switch on, or 'all'")
    args = p.parse_args()

    assert bool(args.ckpt) != bool(args.onnx), "give exactly one of --ckpt / --onnx"
    if args.onnx:
        agent = OnnxPolicy(args.onnx)
        ck = {"n_obs": json.loads(Path(args.onnx).with_suffix(".json").read_text())["n_obs"]}
    else:
        ck = torch.load(args.ckpt, weights_only=False)
        agent = Agent(ck["n_obs"], ck["n_act"]).cuda()
        agent.load_state_dict(ck["agent"])
        agent.eval()
    name = args.onnx or args.ckpt

    env = gym.make("SO101PickPlace-v1", num_envs=args.episodes, obs_mode="state", sim_backend="gpu",
                   domain_rand=args.dr)
    success, grasped, length, _ = rollout(env, agent, args.seed, n_obs=ck["n_obs"])
    env.close()
    print(f"{name}: {args.episodes} episodes, seed {args.seed}, domain_rand {args.dr or 'off'}")
    print(f"  success rate      {success.float().mean() * 100:.1f}% ({success.sum()}/{args.episodes})")
    print(f"  grasp rate        {grasped.float().mean() * 100:.1f}%")
    print(f"  mean ep. length   {length.float().mean():.1f} steps (successes: {length[success].float().mean():.1f})")

    # Videos from small separate batches (rendering 500 envs would not fit in memory):
    # keep up to 3 successes and the rest failures.
    out = Path(name).parent / "eval_videos"
    out.mkdir(exist_ok=True)
    want_fail = args.videos - min(3, args.videos)
    env = gym.make("SO101PickPlace-v1", num_envs=16, obs_mode="state", sim_backend="gpu", render_mode="rgb_array",
                   domain_rand=args.dr,
                   human_render_camera_configs=dict(width=320, height=320))
    saved = {True: 0, False: 0}
    for batch in range(8):
        s, _, _, video = rollout(env, agent, args.seed + 1000 + batch, frames=True, n_obs=ck["n_obs"])
        for i in range(16):
            ok = bool(s[i])
            if saved[ok] < (args.videos - want_fail if ok else want_fail):
                name = out / f"{'success' if ok else 'fail'}_{saved[ok]}.mp4"
                imageio.mimsave(name, video[i], fps=20)
                saved[ok] += 1
        if saved[True] + saved[False] >= args.videos:
            break
    env.close()
    print(f"  videos: {saved[True]} success, {saved[False]} fail -> {out}")


if __name__ == "__main__":
    main()
