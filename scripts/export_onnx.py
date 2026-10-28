"""Export a PPO checkpoint's actor (deterministic mean action) to ONNX, plus a JSON describing its I/O.

Writes <out>.onnx and <out>.json, then checks parity with PyTorch and CPU latency.
Usage: python scripts/export_onnx.py --ckpt runs/<name>/best.pt --out models/so101_pickplace
"""
import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

import so101_rl.envs  # noqa: F401
from so101_rl.envs import pick_place

sys.path.insert(0, str(Path(__file__).parent))
from train_ppo import Agent  # noqa: E402


def obs_layout(n_obs):
    """Names and sizes of the flat observation, in order, read from the env itself."""
    env = gym.make("SO101PickPlace-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu")
    obs, _ = env.reset(seed=0)
    env.close()
    layout = [{"name": f"{group}.{k}", "size": int(v.shape[-1])}
              for group in ("agent", "extra") for k, v in obs[group].items()]
    total = sum(x["size"] for x in layout)
    assert total >= n_obs, (total, n_obs)
    # Older checkpoints use only the first n_obs entries (new ones were appended at the end).
    kept, n = [], 0
    for x in layout:
        if n >= n_obs:
            break
        kept.append(x)
        n += x["size"]
    assert n == n_obs, "checkpoint input size does not end on an observation boundary"
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="models/so101_pickplace")
    args = p.parse_args()

    ck = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    agent = Agent(ck["n_obs"], ck["n_act"])
    agent.load_state_dict(ck["agent"])
    actor = agent.actor_mean.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(actor, torch.zeros(1, ck["n_obs"]), out.with_suffix(".onnx"),
                      input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, dynamo=False)

    meta = {
        "source_checkpoint": args.ckpt,
        "n_obs": ck["n_obs"],
        "n_act": ck["n_act"],
        "obs_layout": obs_layout(ck["n_obs"]),
        "control_hz": 20,
        "action": {
            "mode": "pd_joint_delta_pos",
            "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"],
            # Policy output is clipped to [-1, 1] and scaled to this many radians per step.
            "max_delta_rad": [0.05] * 5 + [0.2],
        },
        "home_qpos": pick_place.HOME_QPOS,
        "zone_center": list(pick_place.ZONE_CENTER),
        "obs_normalisation": None,  # the PPO baseline trains on raw observations
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    # Parity: 1000 observations drawn around realistic magnitudes.
    sess = ort.InferenceSession(str(out.with_suffix(".onnx")), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(0, 0.5, (1000, ck["n_obs"])).astype(np.float32)
    with torch.no_grad():
        ref = actor(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"obs": x})[0]
    diff = float(np.abs(ref - got).max())

    # Latency: one observation at a time, as on the robot.
    one = x[:1]
    for _ in range(50):
        sess.run(None, {"obs": one})
    times = []
    for _ in range(1000):
        t = time.perf_counter()
        sess.run(None, {"obs": one})
        times.append(time.perf_counter() - t)

    print(f"wrote {out.with_suffix('.onnx')} and {out.with_suffix('.json')}")
    print(f"  parity: max |onnx - torch| = {diff:.2e} ({'pass' if diff < 1e-4 else 'FAIL'}, < 1e-4)")
    print(f"  latency: median {np.median(times) * 1e3:.3f} ms, p99 {np.percentile(times, 99) * 1e3:.3f} ms "
          f"({'pass' if np.median(times) < 2e-3 else 'FAIL'}, < 2 ms)")


if __name__ == "__main__":
    main()
