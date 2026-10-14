"""Sweep each SO-101 joint through its range; save a video and a contact sheet of the extremes."""
import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import so101_rl.agents  # noqa: F401
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

STEPS = 40  # per half-sweep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="videos/joint_sweep.mp4")
    args = p.parse_args()

    # Camera from the front-left, looking at the arm's workspace.
    cam_pose = sapien_utils.look_at(eye=[0.45, 0.55, 0.45], target=[0.1, 0.0, 0.15])
    env = gym.make(
        "Empty-v1", robot_uids="so101", num_envs=1, control_mode="pd_joint_pos",
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default", width=640, height=480, pose=cam_pose),
    )
    env.reset(seed=0)
    agent = env.unwrapped.agent
    names = [j.name for j in agent.robot.active_joints]
    limits = agent.robot.get_qlimits()[0].cpu().numpy()

    frames, sheet = [], []
    for i, name in enumerate(names):
        lo, hi = limits[i]
        path = np.concatenate([np.linspace(0, lo, STEPS), np.linspace(lo, hi, 2 * STEPS), np.linspace(hi, 0, STEPS)])
        for k, v in enumerate(path):
            target = np.zeros(len(names))
            target[i] = v
            env.step(torch.tensor(target, dtype=torch.float32)[None])
            img = env.render()[0].cpu().numpy()
            frames.append(img)
            if k in (STEPS - 1, 3 * STEPS - 1):  # at lower / upper limit
                sheet.append(img)
        q = agent.robot.get_qpos()[0, i].item()
        print(f"{name}: range [{lo:+.3f}, {hi:+.3f}] rad, final {q:+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=30)
    # Rows: joints; columns: lower limit, upper limit.
    rows = [np.concatenate(sheet[2 * r:2 * r + 2], axis=1) for r in range(len(names))]
    imageio.imwrite(Path(args.out).with_suffix(".png"), np.concatenate(rows, axis=0)[::2, ::2])
    print("saved", args.out)


if __name__ == "__main__":
    main()
