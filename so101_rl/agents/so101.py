"""SO-101 follower arm for ManiSkill. Adapted from ManiSkill's built-in SO-100 agent."""
import copy
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from .controllers import PDEEPosOnlyControllerConfig

ASSET = Path(__file__).resolve().parents[2] / "assets" / "so101" / "so101.urdf"


@register_agent()
class SO101(BaseAgent):
    uid = "so101"
    urdf_path = str(ASSET)
    # Jaw collisions are CoACD decompositions with several convex parts per file.
    load_multiple_collisions = True
    urdf_config = dict(
        _materials=dict(gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)),
        link=dict(
            gripper_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            moving_jaw_so101_v1_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    # "New calibration" URDF: zero is the middle of every joint's range (same as LeRobot).
    keyframes = dict(
        zero=Keyframe(qpos=np.zeros(6), pose=sapien.Pose()),
    )

    arm_joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    gripper_joint_names = ["gripper"]
    ee_link_name = "gripper_frame_link"

    @property
    def _controller_configs(self):
        stiffness, damping, force_limit = 1e3, 1e2, 100

        pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names + self.gripper_joint_names,
            lower=None, upper=None,
            stiffness=stiffness, damping=damping, force_limit=force_limit,
            normalize_action=False,
        )
        # Small per-step deltas: the real servos shake and wear if driven fast.
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names + self.gripper_joint_names,
            [-0.05] * 5 + [-0.2], [0.05] * 5 + [0.2],
            stiffness=stiffness, damping=damping, force_limit=force_limit,
            use_delta=True,
        )
        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # EE position deltas in the base frame (5-DoF arm: no full 6-D pose control).
        arm_ee_delta_pos = PDEEPosOnlyControllerConfig(
            self.arm_joint_names,
            pos_lower=-0.01, pos_upper=0.01,
            stiffness=stiffness, damping=damping, force_limit=force_limit,
            ee_link=self.ee_link_name, urdf_path=self.urdf_path,
        )
        gripper_pos = PDJointPosControllerConfig(
            self.gripper_joint_names, lower=-0.2, upper=0.2,
            stiffness=stiffness, damping=damping, force_limit=force_limit,
            use_delta=True,
        )
        # Accumulates deltas onto the previous target, so PD lag does not eat the motion.
        arm_ee_target_delta_pos = copy.deepcopy(arm_ee_delta_pos)
        arm_ee_target_delta_pos.use_target = True

        return deepcopy_dict(dict(
            pd_joint_pos=pd_joint_pos,
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_ee_delta_pos=dict(arm=arm_ee_delta_pos, gripper=gripper_pos),
            pd_ee_target_delta_pos=dict(arm=arm_ee_target_delta_pos, gripper=gripper_pos),
        ))

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper_link"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        self.finger1_tip = self.robot.links_map["Fixed_Jaw_tip"]
        self.finger2_tip = self.robot.links_map["Moving_Jaw_tip"]

    @property
    def tcp_pos(self):
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        # The object must push each jaw roughly in its opening direction (away from the other jaw).
        l_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        opening = self.finger2_tip.pose.p - self.finger1_tip.pose.p
        l_angle = common.compute_angle_between(-opening, l_forces)
        r_angle = common.compute_angle_between(opening, r_forces)
        l_ok = (torch.linalg.norm(l_forces, axis=1) >= min_force) & (torch.rad2deg(l_angle) <= max_angle)
        r_ok = (torch.linalg.norm(r_forces, axis=1) >= min_force) & (torch.rad2deg(r_angle) <= max_angle)
        return l_ok & r_ok

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]  # arm joints only
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
