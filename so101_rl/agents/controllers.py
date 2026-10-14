"""Position-only EE controller for a 5-DoF arm.

ManiSkill's GPU IK always solves the full 6-D pose and tries to hold the current
orientation. A 5-DoF arm cannot do that while moving sideways (shoulder pan turns
the gripper), so y/z moves came out wrong. Here IK uses only the 3 position rows.
"""
from dataclasses import dataclass

import torch

from mani_skill.agents.controllers import PDEEPosController, PDEEPosControllerConfig

DAMPING = 1e-4


class PDEEPosOnlyController(PDEEPosController):
    def _initialize_joints(self):
        super()._initialize_joints()
        k = self.kinematics
        assert k.use_gpu_ik, "position-only IK is implemented for the GPU sim only"

        def compute_ik(pose, q0, is_delta_pose=False, current_pose=None, **_):
            q = q0[:, k.active_ancestor_joint_idxs]
            if is_delta_pose:
                dp = pose[:, :3] if isinstance(pose, torch.Tensor) else pose.p
            else:
                dp = pose.p - k.pk_chain.forward_kinematics(q).get_matrix()[:, :3, 3]
            J = k.pk_chain.jacobian(q)[:, :3, k.qmask]
            # Damped least squares: J^T (J J^T + λI)^-1 dp
            JJt = J @ J.transpose(1, 2) + DAMPING * torch.eye(3, device=J.device)
            dq = J.transpose(1, 2) @ torch.linalg.solve(JJt, dp.unsqueeze(-1))
            return q[:, k.qmask] + dq.squeeze(-1)

        k.compute_ik = compute_ik


@dataclass
class PDEEPosOnlyControllerConfig(PDEEPosControllerConfig):
    controller_cls = PDEEPosOnlyController
