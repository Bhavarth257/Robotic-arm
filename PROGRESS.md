# Progress

## M0. Environment and smoke test — 2026-10-14

**Built**
- Git repo, `.gitignore`, venv `.venv` (Python 3.10), pinned `requirements.txt`.
- `scripts/smoke_test.py`: PickCube-v1, GPU sim, state obs, random actions; peak VRAM measured with `nvidia-smi` (includes PhysX memory).

**Versions:** torch 2.14.1+cu130 (CUDA available), mani_skill 3.0.1, sapien 3.0.3, driver 580.178.04, RTX 3050 Laptop 4 GB.

**Results** (1000 steps, seed 0):

| Envs | FPS | Peak VRAM |
|---|---|---|
| 256 | 5,562 | 1,303 MiB |
| 512 | 9,587 | 1,369 MiB |
| 1024 | 15,533 | 1,499 MiB |

Acceptance (runs without error, peak VRAM < 3.5 GB): **pass**.

**Open issues**
- 2048/4096 envs not measured: the session ended during that run twice. Possible cause is system RAM (11 of 15 GB already in use); check free RAM before trying again.
- VRAM barely grows with env count, so local state-based training at 1024 envs looks feasible; M0b (Kaggle) may only be needed for long runs.

## M1. SO-101 robot asset in ManiSkill — 2026-10-14

**Built**
- ManiSkill 3.0.1 has no SO-101 agent (only SO-100), so SO-100 was used as the reference.
- `scripts/prepare_so101_asset.py` → `assets/so101/` from TheRobotStudio SO-ARM100 @ `a758567` (`so101_new_calib.urdf`):
  - Jaw collision meshes split with CoACD (16 convex parts each). A single convex hull of the fixed-jaw body fills the gap between the jaws.
  - Other collision meshes saved as single convex hulls (OBJ).
  - Added `Fixed_Jaw_tip` / `Moving_Jaw_tip` links; TCP = their midpoint (as in SO-100). Computed fixed tip is within 5 mm of the URDF's `gripper_frame_link`.
- `so101_rl/agents/so101.py`: agent `so101` with `pd_joint_pos`, `pd_joint_delta_pos` (±0.05 rad arm, ±0.2 rad gripper per step), `pd_joint_target_delta_pos`, `pd_ee_delta_pos`, `pd_ee_target_delta_pos`. Gains copied from SO-100 (stiffness 1e3, damping 1e2, force limit 100).
- `so101_rl/agents/controllers.py`: position-only EE IK (see deviations).
- `scripts/view_robot.py`: joint sweep video + contact sheet (`videos/joint_sweep.{mp4,png}`, not committed).
- `configs/so101_calibration.yaml`: sim ↔ LeRobot conventions (D6).
- `pyproject.toml`: editable install of `so101_rl`; pytest config disables ROS's pytest plugins (ROS is sourced in every shell and its plugins need ROS-only deps).

**Acceptance**
- Loads in GPU sim with 512 envs: **pass** (`test_loads_512_envs_gpu`).
- Sweep video: all joints move through their full range with no jumps; gripper closed at −0.175 rad, open at +1.745 rad: **pass** (sim side; real directions checked at M8a).
- EE delta test from q = [0, −0.5, 0.25, 1.5, 0, 0] (gripper 21 cm out, 11 cm up), 5 × 1 cm commands then settle: x → +5.00 cm, y → +5.00 cm, z → +5.00 cm, off-axis < 0.01 cm: **pass** (`pytest tests/test_agent.py`, 4 passed).

**Unit conversion (LeRobot → sim)**, also in `configs/so101_calibration.yaml`:
- Arm joints: use LeRobot's `use_degrees=True`; sim_rad = deg2rad(sign × (real_deg − offset)). Both conventions put zero at mid-range, so offsets should be ~0. Signs unverified.
- Gripper: LeRobot 0 (closed)–100 (open) → sim_rad = −0.175 + (g/100) × 1.920. Assumed linear over the calibrated range; verify at M8a.

**Deviations from plan**
- `pd_ee_delta_pose` (x, y, z, yaw) replaced by `pd_ee_delta_pos` (x, y, z) + gripper. ManiSkill's GPU IK solves the full 6-D pose and tries to hold orientation, which a 5-DoF arm cannot do while moving sideways: y moved 1.3 cm of 5 cm, z moved 7.4 cm with 3.9 cm of x drift. The new controller solves position only (damped least squares on the 3 position rows). Yaw is not controlled directly; use the `wrist_roll` joint. The policy uses joint deltas anyway (D1).
- Added `pd_ee_target_delta_pos`: with the SO-100 gains the joints reach only ~40% of a step's target within one 50 ms control step (time constant ~0.1 s), so the non-target variant loses most of each command.

**Open issues**
- PD gains and force limit (100 N·m) are SO-100 placeholders. The STS3215 stall torque is ~1–3 N·m depending on voltage. Fit in M8b; consider a realistic force limit in M5.
- ManiSkill disables gravity on robot links (stand-in for gravity compensation). The real arm sags under load. Revisit in M8b system ID.
- Joint directions, wrist_roll sign and the gripper mapping must be checked on hardware (M8a).
- Home pose (D5) not chosen yet; needs the overhead camera's position. Will pick a provisional one in M2.
