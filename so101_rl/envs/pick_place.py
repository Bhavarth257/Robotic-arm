"""SO101PickPlace-v1: pick a cube and place it in a fixed zone on the table."""
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


from so101_rl.agents import SO101

# Provisional home pose (D5): arm raised, gripper ~40 cm up, open ~6 cm.
# Replace once the overhead camera position is known.
HOME_QPOS = [0.0, -0.5, -0.8, 1.2, 0.0, 0.6]
CUBE_HALF = 0.015
SPAWN_CENTER = (0.22, 0.0)  # 20 x 20 cm region, fully reachable with a near top-down grasp
SPAWN_HALF = 0.10
ZONE_CENTER = (0.15, 0.20)
ZONE_HALF = 0.025
GRIPPER_OPEN = 0.4  # rad; jaw gap clears a 3.5 cm cube
GRIPPER_SHUT = 0.0  # rad; below this the jaws are closer than any cube, so nothing is held
LIFT_HEIGHT = 0.04  # lift above resting height that earns full lift reward
MAX_REWARD = 6.0


@register_env("SO101PickPlace-v1", max_episode_steps=200)
class SO101PickPlaceEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["so101"]
    agent: SO101

    def __init__(self, *args, robot_uids="so101", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        @property
        def _default_sim_config(self):
        # Gripper contacts (many convex parts per jaw) overflow PhysX's default GPU buffers
        # at 1024 envs and crash with CUDA error 700. Sized like ManiSkill's own tasks.
         n = max(1024, self.num_envs)
         return SimConfig(
            sim_freq=100, control_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * n * 8,
                max_rigid_patch_count=self.num_envs * n * 2,
                found_lost_pairs_capacity=2**26,
            ),
        )


    @property
    def _default_sensor_configs(self):
        return []  # state-based policy

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.55, 0.45, 0.45], target=[0.2, 0.08, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        # The ground plane is the table top; the arm base sits on it at the origin.
        build_ground(self.scene)
        self.cube = actors.build_cube(
            self.scene, half_size=CUBE_HALF, color=[1, 0, 0, 1], name="cube",
            initial_pose=sapien.Pose(p=[*SPAWN_CENTER, CUBE_HALF]),
        )
        self.zone = actors.build_box(
            self.scene, half_sizes=[ZONE_HALF, ZONE_HALF, 0.0005], color=[0, 1, 0, 1], name="zone",
            body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[*ZONE_CENTER, 0.0005]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.tensor(HOME_QPOS) + torch.randn(b, 6) * self.robot_init_qpos_noise
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose())

            xyz = torch.zeros(b, 3)
            xyz[:, :2] = torch.tensor(SPAWN_CENTER) + (torch.rand(b, 2) * 2 - 1) * SPAWN_HALF
            xyz[:, 2] = CUBE_HALF
            yaw_q = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, yaw_q))

            if not hasattr(self, "last_action"):
                self.last_action = torch.zeros(self.num_envs, self.agent.single_action_space.shape[0])
            self.last_action[env_idx] = 0

    def step(self, action):
        # Keep the clipped action for the "previous action" observation.
        a = common.to_tensor(action, device=self.device).reshape(self.num_envs, -1)
        low = common.to_tensor(self.agent.single_action_space.low, device=self.device)
        high = common.to_tensor(self.agent.single_action_space.high, device=self.device)
        self.last_action = torch.clamp(a.float(), low, high)
        return super().step(action)

    def _get_obs_extra(self, info: dict):
        # Only quantities available on the real robot (FK, ArUco pose, fixed zone); no is_grasped.
        tcp, cube = self.agent.tcp_pose, self.cube.pose
        R = cube.to_transformation_matrix()[:, :3, :3]
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        return dict(
            tcp_pose=tcp.raw_pose,
            cube_pos=cube.p,
            # A cube looks the same every 90 deg, so encode 4*yaw.
            cube_yaw=torch.stack([torch.sin(4 * yaw), torch.cos(4 * yaw)], dim=1),
            zone_pos=self.zone.pose.p,
            tcp_to_cube=cube.p - tcp.p,
            cube_to_zone=self.zone.pose.p - cube.p,
            prev_action=self.last_action,
        )

    def evaluate(self):
        cube_p, zone_p = self.cube.pose.p, self.zone.pose.p
        tcp_p = self.agent.tcp_pose.p
        in_zone = ((cube_p[:, :2] - zone_p[:, :2]).abs() <= ZONE_HALF).all(dim=1)
        on_table = (cube_p[:, 2] < CUBE_HALF + 0.005) & (self.cube.linear_velocity.norm(dim=1) < 0.05)
        gripper_q = self.agent.robot.get_qpos()[:, -1]
        # A fully shut gripper pressing on the cube's top edges also triggers both jaw contacts;
        # the first policy exploited that instead of grasping.
        is_grasped = self.agent.is_grasping(self.cube) & (gripper_q > GRIPPER_SHUT)
        gripper_open = gripper_q > GRIPPER_OPEN
        tcp_above = tcp_p[:, 2] > cube_p[:, 2] + CUBE_HALF + 0.01
        return dict(
            success=in_zone & on_table & gripper_open & tcp_above & ~is_grasped,
            is_grasped=is_grasped,
            in_zone=in_zone,
            on_table=on_table,
            lifted=cube_p[:, 2] > CUBE_HALF + 0.01,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cube_p, zone_p = self.cube.pose.p, self.zone.pose.p
        tcp_p = self.agent.tcp_pose.p
        grasped = info["is_grasped"].float()

        gripper = torch.clamp(self.agent.robot.get_qpos()[:, -1] / GRIPPER_OPEN, 0, 1)
        # Approach with the gripper open: closing early loses half the reach reward,
        # closing on the cube still pays because the grasp term (+1) outweighs it.
        reach = (1 - torch.tanh(5 * (cube_p - tcp_p).norm(dim=1))) * (0.5 + 0.5 * torch.maximum(gripper, grasped))
        lift = torch.clamp((cube_p[:, 2] - CUBE_HALF) / LIFT_HEIGHT, 0, 1)
        # Carrying pays only while lifted (no dragging), or once above the zone, so that lowering
        # the cube there costs nothing: the previous policy held the cube high over the zone.
        carry = torch.maximum(lift, info["in_zone"].float())
        goal = zone_p.clone()
        goal[:, 2] = CUBE_HALF  # resting on the table inside the zone
        place = 1 - torch.tanh(5 * (cube_p - goal).norm(dim=1))
        reward = reach + grasped + grasped * carry + grasped * carry * place

        placed = info["in_zone"] & info["on_table"] & ~info["is_grasped"]
        above = (tcp_p[:, 2] > cube_p[:, 2] + CUBE_HALF + 0.01).float()
        reward = torch.where(placed, 4 + 0.5 * gripper + 0.5 * above, reward)
        reward = torch.where(info["success"], torch.full_like(reward, MAX_REWARD), reward)

        a = torch.clamp(common.to_tensor(action, device=self.device).reshape(self.num_envs, -1), -1, 1)
        reward = reward - 0.1 * (a ** 2).mean(dim=1)
        falling = (self.cube.linear_velocity[:, 2] < -0.1) & ~info["is_grasped"] & ~info["in_zone"]
        reward = reward - 0.5 * falling.float()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Range [-0.1, 1]: penalties go slightly below zero.
        return self.compute_dense_reward(obs=obs, action=action, info=info) / MAX_REWARD
