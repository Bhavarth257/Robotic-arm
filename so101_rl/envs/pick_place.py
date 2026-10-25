"""SO101PickPlace-v1: pick a cube and place it in a fixed zone on the table."""
from typing import Any, Sequence

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from so101_rl.agents import SO101

# Provisional home pose (D5): arm raised, gripper ~40 cm up, open ~6 cm.
# Replace once the overhead camera position is known.
HOME_QPOS = [0.0, -0.5, -0.8, 1.2, 0.0, 0.6]
CUBE_HALF = 0.015
CUBE_FRICTION = 0.3  # ManiSkill's default actor material
CUBE_DENSITY = 1000.0
SPAWN_CENTER = (0.22, 0.0)  # 20 x 20 cm region, fully reachable with a near top-down grasp
SPAWN_HALF = 0.10
ZONE_CENTER = (0.15, 0.20)
ZONE_HALF = 0.025
GRIPPER_OPEN = 0.4  # rad; jaw gap clears a 3.5 cm cube
GRIPPER_SHUT = 0.0  # rad; below this the jaws are closer than any cube, so nothing is held
LIFT_HEIGHT = 0.04  # lift above resting height that earns full lift reward
MAX_REWARD = 6.0

# Domain randomisation (M5). Each can be switched on alone, to find the weakest one.
DR_OPTIONS = {
    "cube_size",      # half size 1.25-1.75 cm (cube 2.5-3.5 cm), per env
    "cube_mass",      # density x0.5-1.5, per env
    "cube_friction",  # friction x0.5-1.5, per env
    "action_gain",    # per-joint action scale 0.8-1.2, per episode (stands in for actuator stiffness/damping +-20%)
    "action_delay",   # 0 or 1 control step of delay, per episode
    "cube_noise",     # observed cube position +-5 mm, yaw +-5 deg, every step
    "cube_latency",   # observed cube pose 0-2 steps old, per episode
    "joint_offset",   # observed joints offset +-2 deg per episode, plus +-0.5 deg noise every step
}
DEG = np.pi / 180


@register_env("SO101PickPlace-v1", max_episode_steps=200)
class SO101PickPlaceEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["so101"]
    agent: SO101

    def __init__(self, *args, robot_uids="so101", robot_init_qpos_noise=0.02,
                 domain_rand: Sequence[str] = (), **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.dr = set(DR_OPTIONS) if "all" in domain_rand else set(domain_rand)
        unknown = self.dr - DR_OPTIONS
        assert not unknown, f"unknown domain_rand options: {unknown}"
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # Gripper contacts (many convex parts per jaw) overflow PhysX's default GPU buffers
        # (2**19 contacts, 2**18 patches) at 1024 envs: CUDA error 700. Defaults were enough
        # at 512 envs, so scale from there. ManiSkill's own sizing (envs * 1024 * 8) does not
        # fit in 4 GB.
        n = max(512, self.num_envs)
        return SimConfig(
            sim_freq=100, control_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=n * 2048,
                max_rigid_patch_count=n * 1024,
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

        # One cube per env so size, mass and friction can differ between envs.
        halves, cubes = [], []
        for i in range(self.num_envs):
            rng = self._batched_episode_rng[i]
            half = rng.uniform(0.0125, 0.0175) if "cube_size" in self.dr else CUBE_HALF
            density = CUBE_DENSITY * (rng.uniform(0.5, 1.5) if "cube_mass" in self.dr else 1.0)
            friction = CUBE_FRICTION * (rng.uniform(0.5, 1.5) if "cube_friction" in self.dr else 1.0)
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[half] * 3, density=density,
                                      material=sapien.physx.PhysxMaterial(friction, friction, 0))
            builder.add_box_visual(half_size=[half] * 3,
                                   material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]))
            builder.initial_pose = sapien.Pose(p=[*SPAWN_CENTER, half])
            builder.set_scene_idxs([i])
            cubes.append(builder.build(name=f"cube_{i}"))
            self.remove_from_state_dict_registry(cubes[-1])
            halves.append(half)
        self.cube = Actor.merge(cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)
        self.cube_half = torch.tensor(halves, dtype=torch.float32, device=self.device)

        self.zone = actors.build_box(
            self.scene, half_sizes=[ZONE_HALF, ZONE_HALF, 0.0005], color=[0, 1, 0, 1], name="zone",
            body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[*ZONE_CENTER, 0.0005]),
        )

        n, n_act = self.num_envs, 6
        self.last_action = torch.zeros(n, n_act, device=self.device)   # what the policy commanded
        self.applied_action = torch.zeros(n, n_act, device=self.device)  # what reached the robot
        self.action_gain = torch.ones(n, n_act, device=self.device)
        self.delayed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.cube_lag = torch.zeros(n, dtype=torch.long, device=self.device)
        self.cube_history = torch.zeros(3, n, 4, device=self.device)  # last 3 (x, y, z, yaw), newest first
        self.joint_offset = torch.zeros(n, n_act, device=self.device)
        self.fresh = torch.ones(n, dtype=torch.bool, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.tensor(HOME_QPOS) + torch.randn(b, 6) * self.robot_init_qpos_noise
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose())

            xyz = torch.zeros(b, 3)
            xyz[:, :2] = torch.tensor(SPAWN_CENTER) + (torch.rand(b, 2) * 2 - 1) * SPAWN_HALF
            xyz[:, 2] = self.cube_half[env_idx]
            yaw_q = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, yaw_q))

            self.last_action[env_idx] = 0
            self.applied_action[env_idx] = 0
            if "action_gain" in self.dr:
                self.action_gain[env_idx] = 0.8 + 0.4 * torch.rand(b, 6)
            if "action_delay" in self.dr:
                self.delayed[env_idx] = torch.rand(b) < 0.5
            if "cube_latency" in self.dr:
                self.cube_lag[env_idx] = torch.randint(0, 3, (b,))
            if "joint_offset" in self.dr:
                self.joint_offset[env_idx] = (torch.rand(b, 6) * 2 - 1) * 2 * DEG
            self.fresh[env_idx] = True

    def step(self, action):
        a = common.to_tensor(action, device=self.device).reshape(self.num_envs, -1).float()
        a = torch.clamp(a, -1, 1)
        self.last_action = a  # the policy's own command, as the real controller would know it
        applied = torch.clamp(a * self.action_gain, -1, 1)
        # A delayed env executes the previous command this step.
        applied = torch.where(self.delayed[:, None], self.applied_action, applied)
        self.applied_action = applied
        return super().step(applied)

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        if "joint_offset" in self.dr:
            noise = (torch.rand_like(obs["qpos"]) * 2 - 1) * 0.5 * DEG
            obs["qpos"] = obs["qpos"] + self.joint_offset + noise
        return obs

    def _observed_cube(self):
        """Cube position and yaw as the camera would report them: possibly stale and noisy."""
        R = self.cube.pose.to_transformation_matrix()[:, :3, :3]
        now = torch.cat([self.cube.pose.p, torch.atan2(R[:, 1, 0], R[:, 0, 0])[:, None]], dim=1)
        self.cube_history = torch.roll(self.cube_history, 1, dims=0)
        self.cube_history[0] = now
        self.cube_history[:, self.fresh] = now[self.fresh]  # new episode: no older frames yet
        self.fresh[:] = False
        seen = self.cube_history[self.cube_lag, torch.arange(self.num_envs, device=self.device)]
        if "cube_noise" in self.dr:
            noise = torch.rand_like(seen) * 2 - 1
            seen = seen + noise * torch.tensor([0.005, 0.005, 0.005, 5 * DEG], device=self.device)
        return seen[:, :3], seen[:, 3]

    def _get_obs_extra(self, info: dict):
        # Only quantities available on the real robot (FK, ArUco pose, fixed zone); no is_grasped.
        tcp = self.agent.tcp_pose
        cube_p, yaw = self._observed_cube()
        return dict(
            tcp_pose=tcp.raw_pose,
            cube_pos=cube_p,
            # A cube looks the same every 90 deg, so encode 4*yaw.
            cube_yaw=torch.stack([torch.sin(4 * yaw), torch.cos(4 * yaw)], dim=1),
            zone_pos=self.zone.pose.p,
            tcp_to_cube=cube_p - tcp.p,
            cube_to_zone=self.zone.pose.p - cube_p,
            prev_action=self.last_action,
        )

    def evaluate(self):
        # Rewards and success use the true state; only the observations are corrupted.
        cube_p, zone_p = self.cube.pose.p, self.zone.pose.p
        tcp_p = self.agent.tcp_pose.p
        half = self.cube_half
        in_zone = ((cube_p[:, :2] - zone_p[:, :2]).abs() <= ZONE_HALF).all(dim=1)
        on_table = (cube_p[:, 2] < half + 0.005) & (self.cube.linear_velocity.norm(dim=1) < 0.05)
        gripper_q = self.agent.robot.get_qpos()[:, -1]
        # A fully shut gripper pressing on the cube's top edges also triggers both jaw contacts;
        # the first policy exploited that instead of grasping.
        is_grasped = self.agent.is_grasping(self.cube) & (gripper_q > GRIPPER_SHUT)
        gripper_open = gripper_q > GRIPPER_OPEN
        tcp_above = tcp_p[:, 2] > cube_p[:, 2] + half + 0.01
        return dict(
            success=in_zone & on_table & gripper_open & tcp_above & ~is_grasped,
            is_grasped=is_grasped,
            in_zone=in_zone,
            on_table=on_table,
            lifted=cube_p[:, 2] > half + 0.01,
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        cube_p, zone_p = self.cube.pose.p, self.zone.pose.p
        tcp_p = self.agent.tcp_pose.p
        half = self.cube_half
        grasped = info["is_grasped"].float()

        gripper = torch.clamp(self.agent.robot.get_qpos()[:, -1] / GRIPPER_OPEN, 0, 1)
        # Approach with the gripper open: closing early loses half the reach reward,
        # closing on the cube still pays because the grasp term (+1) outweighs it.
        reach = (1 - torch.tanh(5 * (cube_p - tcp_p).norm(dim=1))) * (0.5 + 0.5 * torch.maximum(gripper, grasped))
        lift = torch.clamp((cube_p[:, 2] - half) / LIFT_HEIGHT, 0, 1)
        # Carrying pays only while lifted (no dragging), or once above the zone, so that lowering
        # the cube there costs nothing: the previous policy held the cube high over the zone.
        carry = torch.maximum(lift, info["in_zone"].float())
        goal = zone_p.clone()
        goal[:, 2] = half  # resting on the table inside the zone
        place = 1 - torch.tanh(5 * (cube_p - goal).norm(dim=1))
        reward = reach + grasped + grasped * carry + grasped * carry * place

        # Count as placed as soon as the cube is let go low over the zone, not only once it has
        # settled: otherwise opening the gripper first loses the grasp reward and the policy
        # learned to hold the cube 5 cm above the zone forever.
        placed = info["in_zone"] & ~info["is_grasped"] & (cube_p[:, 2] < half + 0.05)
        above = (tcp_p[:, 2] > cube_p[:, 2] + half + 0.01).float()
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
