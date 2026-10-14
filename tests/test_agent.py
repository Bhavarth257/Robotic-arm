import gymnasium as gym
import pytest
import torch

import mani_skill.envs  # noqa: F401
import so101_rl.agents  # noqa: F401

# Gripper ~21 cm in front of the base, 11 cm up, pointing mostly down.
Q_TEST = [0.0, -0.5, 0.25, 1.5, 0.0, 0.0]
STEP = 0.01  # metres per full-scale EE action


def set_qpos(env, qpos):
    u = env.unwrapped
    q = torch.tensor([qpos] * u.num_envs, device=u.device)
    u.agent.robot.set_qpos(q)
    u.agent.robot.set_qvel(q * 0)
    # GPU sim: push the new state and read it back before the controller samples qpos.
    u.scene._gpu_apply_all()
    u.scene.px.gpu_update_articulation_kinematics()
    u.scene._gpu_fetch_all()
    u.agent.controller.reset()


@pytest.fixture(scope="module")
def ee_env():
    env = gym.make("Empty-v1", robot_uids="so101", num_envs=1, sim_backend="gpu",
                   control_mode="pd_ee_target_delta_pos")
    env.reset(seed=0)
    yield env
    env.close()


def test_loads_512_envs_gpu():
    env = gym.make("Empty-v1", robot_uids="so101", num_envs=512, sim_backend="gpu",
                   control_mode="pd_joint_delta_pos")
    env.reset(seed=0)
    env.step(torch.zeros(env.action_space.shape, device="cuda"))
    assert env.unwrapped.agent.robot.get_qpos().shape == (512, 6)
    env.close()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_ee_delta_moves_along_base_axis(ee_env, axis):
    ee = ee_env.unwrapped.agent.robot.links_map["gripper_frame_link"]
    set_qpos(ee_env, Q_TEST)
    zero = torch.zeros(1, 4, device="cuda")
    for _ in range(5):
        ee_env.step(zero)
    start = ee.pose.p[0].clone()

    n = 5
    action = zero.clone()
    action[0, axis] = 1.0
    for _ in range(n):
        ee_env.step(action)
    for _ in range(20):  # settle
        ee_env.step(zero)

    moved = (ee.pose.p[0] - start).cpu()
    commanded = n * STEP
    assert abs(moved[axis] - commanded) <= 0.1 * commanded, moved
    off_axis = [moved[i] for i in range(3) if i != axis]
    assert max(abs(v) for v in off_axis) <= 0.1 * commanded, moved
