"""Record sim episodes driven by the exported ONNX policy, for the C++ bridge's replay test.

Each row: episode start flag, the raw inputs the robot side has (joint pos/vel, TCP pose, cube
position and yaw), the 45-value observation the sim built from them, and the policy's raw action.
No domain randomisation, so observed values equal true values.
Usage: python scripts/record_replay.py --onnx models/so101_pickplace.onnx --out <csv>
"""
import argparse
import csv

import gymnasium as gym
import numpy as np
import onnxruntime as ort

import so101_rl.envs  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="models/so101_pickplace.onnx")
    p.add_argument("--out", default="ros2_ws/src/so101_policy_bridge/test/data/replay.csv")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    n_obs = sess.get_inputs()[0].shape[1]
    env = gym.make("SO101PickPlace-v1", num_envs=1, obs_mode="state_dict", sim_backend="cpu")
    u = env.unwrapped

    def flat(o):
        return np.concatenate([v[0].numpy() for g in ("agent", "extra") for v in o[g].values()])[:n_obs]

    header = (["episode_start"] + [f"qpos{i}" for i in range(6)] + [f"qvel{i}" for i in range(6)]
              + [f"tcp{i}" for i in range(7)] + ["cube_x", "cube_y", "cube_z", "cube_yaw"]
              + [f"obs{i}" for i in range(n_obs)] + [f"action{i}" for i in range(6)])
    rows = []
    for ep in range(args.episodes):
        o, _ = env.reset(seed=100 + ep)
        for t in range(200):
            obs = flat(o).astype(np.float32)
            action = sess.run(None, {"obs": obs[None]})[0][0]
            R = u.cube.pose.to_transformation_matrix()[0, :3, :3].numpy()
            yaw = np.arctan2(R[1, 0], R[0, 0])
            rows.append([int(t == 0)] + list(o["agent"]["qpos"][0].numpy()) + list(o["agent"]["qvel"][0].numpy())
                        + list(o["extra"]["tcp_pose"][0].numpy()) + list(o["extra"]["cube_pos"][0].numpy())
                        + [yaw] + list(obs) + list(action))
            o, _, term, trunc, info = env.step(action[None])
            if term.any() or trunc.any():
                break
        print(f"episode {ep}: {t + 1} steps, success={bool(info['success'].any())}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[f"{x:.9g}" for x in r] for r in rows])
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
