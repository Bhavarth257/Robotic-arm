# SO-ARM 101 RL Pick-and-Place: Implementation Plan (v2)

Plan for Claude Code. Work through milestones in order. A milestone is done only when every acceptance check passes and the result is logged in `PROGRESS.md`.

## 1. Context

- **Robot:** 2× SO-ARM 101 (5 arm joints + gripper, STS3215 servos, Feetech bus). Both arms are followers driven by a computer; no leader-arm teleop.
- **Goal:** train skills (pick-and-place, then handover) with RL in simulation, run them on the real arms through ROS 2, then chain skills into two-arm tasks (e.g. pick an apple, hand it to the other arm, cut it) with C++ task nodes.
- **Order:** single arm sim → single arm real → two-arm handover (sim, then real) → task orchestration.
- **Simulator:** ManiSkill3 (SAPIEN, GPU-parallel). Do not use Isaac Sim/Lab.
- **Dev machine:** laptop, RTX 3050 Laptop **4 GB VRAM**, 16 GB RAM, Ubuntu **22.04**, Python 3.10, ROS 2 **Humble** at `/opt/ros/humble`.
- **Cloud training (free):** Kaggle notebooks (primary, T4), Lightning AI (secondary), Colab (fallback). Hugging Face ZeroGPU is for short inference calls and is not usable for training.
- **Runtime:** laptop first; Raspberry Pi later (Pi 5 recommended, Ubuntu 22.04 + Humble, ONNX Runtime on CPU).
- **Perception:** ArUco markers on objects. Camera: Evontro IMX335 (5 MP, 30 fps, USB), one per arm on top of the wrist (rotates with wrist roll); plus a fixed overhead camera. No RealSense needed. Run ArUco at 1280×720 or 1920×1080 (MJPEG): full 5 MP adds latency without improving pose accuracy at this range.
- **Hardware status:** arms, wrist cameras and overhead camera available. Nothing runs on hardware until M8 and I say so.

## 2. Working rules

1. One milestone at a time. Do not start the next until acceptance checks pass.
2. Keep code minimal: smallest change that solves the task, no speculative abstractions, no unused config options.
3. Comments: short and plain. Explain why, not what.
4. Ask before installing system packages (`apt`) or changing anything under `/opt/ros`.
5. **Never send commands to real hardware** unless I explicitly say so in the session. Hardware code defaults to dry-run.
6. VRAM: local GPU has 4 GB. Start with 256 envs locally, peak VRAM under 3.5 GB. On a cloud T4 (16 GB) start at 1024–2048.
7. Pin versions in `requirements.txt` (same file locally and on cloud). Fix seeds for every eval.
8. After each milestone, append to `PROGRESS.md`: date, what was built, acceptance results (numbers), open issues.
9. If something in this plan turns out wrong (API changed, asset missing), stop, explain, and propose the fix before working around it.
10. Training must resume from the latest checkpoint (cloud sessions get killed without warning).
11. Git: commit after each milestone. Never commit checkpoints, videos or datasets.

## 3. Software stack

- **Training (Python venv `.venv`, not ROS's Python):** `mani_skill`, `torch` (CUDA build matching driver), `onnx`, `onnxruntime`, `tensorboard`, `pytest`.
- **Robot side (ROS 2 Humble, C++, `ament_cmake`):** ros2_control, MoveIt 2, OpenCV ArUco, ONNX Runtime C++ (prebuilt release unpacked in the repo's `third_party/`, not system-wide), BehaviorTree.CPP (M10). Python only for tests and offline tools. Nothing on the robot side imports ManiSkill.

## 4. Design decisions

- **D1 Policy action = joint deltas.** `pd_joint_delta_pos` for the 5 arm joints + gripper target. Joint targets mean the same thing in sim and on ros2_control, so there is no IK mismatch between sim and real. (ManiSkill's PickCube PPO baseline uses this controller too.) `pd_ee_delta_pose` is still registered for tests and scripted motions.
- **D2 Real execution path.** Policy node → joint position targets at 20 Hz → ros2_control forward position controller → STS3215 hardware interface. MoveIt 2 handles planned motions (home, place, cut) through a JointTrajectoryController. Switch controllers with `controller_manager` when going from a MoveIt motion to a policy skill. MoveIt Servo is not used (poor fit for a 5-DoF arm).
- **D3 Object pose from ArUco, overhead camera first.** Marker of known size on the object → 6D pose from one camera. The fixed overhead camera is the primary source of `/object_pose`: `T_base_obj = T_base_cam · T_cam_marker`, with `T_base_cam` from an ArUco board at a known spot on the table. This avoids arm FK and backlash errors. The wrist camera is kept for later (close-range refinement, image-based policies).
- **D4 Occlusion while grasped.** A top-down grasp hides a marker on the cube's top face from the overhead camera. Put markers on the sides as well as the top, and when none is visible while grasped the bridge uses `object pose = EE pose · (offset recorded at grasp)`. The `/object_pose` watchdog is suspended while grasped.
- **D5 Fixed home pose.** Each episode starts from one home pose that keeps the arm out of the overhead camera's view of the spawn region. Sim reset uses the same pose (+ small noise).
- **D6 One calibration file.** `configs/so101_calibration.yaml` holds joint zero offsets, directions and unit conversions (LeRobot degrees and gripper 0–100 → sim radians). The hardware interface and sim tooling both read it.
- **D7 Handover policy (M9).** One centralised policy for both arms (to confirm at M9).

## 5. Repo layout

```
robotic_arm/                       # git root
├── PLAN.md, PROGRESS.md, requirements.txt, .gitignore
├── assets/so101/                  # URDF + meshes (TheRobotStudio SO-ARM100 repo, Simulation/SO101)
├── so101_rl/
│   ├── agents/so101.py
│   ├── envs/pick_place.py         # SO101PickPlace-v1
│   ├── envs/handover.py           # SO101Handover-v1 (M9)
│   └── randomization.py
├── scripts/                       # smoke_test, view_robot, train_ppo, evaluate, export_onnx, gen_demos (only if needed)
├── notebooks/kaggle_train.ipynb   # thin wrapper: clone repo, install, run train_ppo with resume
├── configs/
├── tests/
├── third_party/                   # ONNX Runtime C++ release (gitignored)
└── ros2_ws/src/
    ├── so101_description/         # same URDF as sim, xacro for two arms
    ├── so101_hardware/            # C++ ros2_control hardware interface (STS3215)
    ├── so101_moveit_config/
    ├── so101_perception/          # C++ ArUco → /object_pose
    ├── so101_policy_bridge/       # C++ ONNX policy node
    ├── so101_tasks/               # C++ BehaviorTree task nodes (M10)
    └── so101_bringup/             # launch files (mock / dry-run / real)
```

## 6. Milestones

### M0. Environment and smoke test (local)

- `git init`, `.gitignore`, venv, install packages, confirm `torch.cuda.is_available()`.
- `scripts/smoke_test.py`: `PickCube-v1`, `sim_backend="gpu"`, state obs, 256 envs, 1000 steps; print FPS and peak VRAM. Retry with 512 if there is headroom.

**Acceptance:** runs without error; FPS and peak VRAM in `PROGRESS.md`; peak VRAM < 3.5 GB.

### M0b. Cloud training check

- `notebooks/kaggle_train.ipynb`: clone repo, install `requirements.txt`, run the smoke test at 2048 envs on a T4, then a 2-minute PPO run on `PickCube-v1`, kill it, resume.
- Use T4, not P100 (recent PyTorch builds may not support P100). State-only training needs no display; videos are rendered locally from checkpoints.

**Acceptance:** smoke test passes on Kaggle; a killed run resumes from the checkpoint at the same step count.

### M1. SO-101 robot asset in ManiSkill

- First check whether the installed ManiSkill version already ships an SO-101 agent (it has SO-100). Reuse it if so.
- Otherwise: URDF + meshes from TheRobotStudio repo; `agents/so101.py` registers `so101` with `pd_joint_delta_pos`, `pd_joint_pos`, `pd_ee_delta_pose`.
- Check joint limits and directions against the URDF and LeRobot's SO-101 config. Write `configs/so101_calibration.yaml` (D6).
- Define the home pose (D5).
- `scripts/view_robot.py`: sweep each joint through its range, save a video.

**Acceptance:**
- Arm loads in GPU sim at 256 envs.
- Sweep video looks correct (no flipped axes, gripper opens/closes).
- Test: a +x EE delta command moves the gripper +x in the base frame (±10% of commanded distance).

### M2. Single-arm pick-and-place task: `SO101PickPlace-v1`

- Scene: table, one arm at origin, one 3 cm cube, one 5 × 5 cm place zone.
- Reset: arm at home pose + noise; cube uniform in 20 × 20 cm in front of the arm, random yaw; place zone fixed for v1.
- Observation (state): joint pos + vel, gripper, TCP pose (from FK), cube position + yaw (as sin/cos, since ArUco gives it), place-zone position, previous action.
- Action: D1, clipped to ±0.05 rad per joint per step, plus gripper. 20 Hz.
- Reward: dense, staged (reach → grasp → lift → move to zone → release) + success bonus; penalties for large actions and dropping the cube outside the zone. Per-step reward in [−0.2, 1].
- Success: cube inside zone, resting on table, gripper open and above cube.
- Episode length: 200 steps (10 s).

**Acceptance:**
- `pytest tests/test_pick_place.py`: reset deterministic with fixed seed; success predicate true for a hand-placed cube in the zone and false outside; reward finite for random actions over 256 envs.
- Random-action run at 256 envs fits under 3.5 GB.

### M3. Scripted baseline (only if M4 stalls)

- Skip unless M4 is under 50% success after 20M steps.
- Scripted waypoints with the EE-delta controller (mplib needs an SRDF and struggles with 5-DoF orientation goals).

**Acceptance:** ≥ 80% scripted success over 200 episodes; 300+ successful demos in ManiSkill trajectory format.

### M4. PPO training (state-based)

- `scripts/train_ppo.py` adapted from ManiSkill's PPO baseline; config `configs/ppo_pickplace.yaml`.
- Run locally first. If projected time to 95% is over ~6 h, move to Kaggle.
- TensorBoard, checkpoint every 10 min, keep the best by eval success, resume support.
- `scripts/evaluate.py`: 500 episodes, fixed seed, success rate, grasp rate, mean episode length; 5 videos.
- If < 50% after 20M steps: do M3, warm-start (BC init or RLPD), log which.

**Acceptance:** ≥ 95% success over 500 eval episodes, no randomisation. Curve and numbers in `PROGRESS.md`.

### M5. Domain randomisation and robustness

- Per episode: cube mass ±50%, friction 0.5–1.5×, cube size 2.5–3.5 cm, actuator stiffness/damping ±20%, action delay 0–1 control steps.
- Added: per-joint zero offset ±2° (calibration error), joint position noise ±0.5° (backlash), cube pose noise ±5 mm / ±5° yaw, cube pose latency 0–2 steps (camera).
- Fine-tune from the M4 checkpoint.

**Acceptance:** ≥ 90% success over 500 episodes with all randomisation on. Report success with each randomisation turned up on its own to find the weakest.

### M6. Export and CPU inference

- `scripts/export_onnx.py`: actor only (deterministic mean) → ONNX; observation normalisation stats → JSON (read by C++).
- Parity: ONNX vs PyTorch on 1000 random observations, max abs diff < 1e-4. Same check from a small C++ ONNX Runtime test program.
- Latency on laptop CPU: median < 2 ms per step.
- Closed loop: M5 eval with the ONNX policy.

**Acceptance:** parity (Python and C++), latency, and closed-loop success within 2% of PyTorch.

### M7. ROS 2 software stack (no hardware)

- `so101_description`: same URDF as sim.
- `so101_moveit_config`: MoveIt Setup Assistant, groups `arm` (5 joints) and `gripper`; `pick_ik` (handles 5-DoF goals); `mock_components/GenericSystem` for testing.
- `so101_policy_bridge` (C++):
  - subscribes `/joint_states`, `/object_pose` (PoseStamped, base frame)
  - builds the same observation vector as the sim env and runs ONNX at 20 Hz
  - publishes joint targets to the forward position controller
  - `dry_run:=true` by default (logs, publishes nothing)
  - watchdog: stop if `/joint_states` is older than 100 ms or `/object_pose` is older than 200 ms (except while grasped, D4)
  - enforces joint limits, workspace box and per-step delta clamp itself
- Replay test: fake publisher replays a sim episode; node actions match sim policy actions (< 1e-4).

**Acceptance:** replay test passes; `so101_bringup` launches with mock hardware, MoveIt and RViz without hardware; the policy drives the mock arm in RViz from a fake `/object_pose`.

### M8. Hardware (blocked until I confirm)

- **M8a Bring-up and calibration:** C++ hardware interface for the STS3215 bus. Check for an existing Humble-compatible package first. Read-only first (joint states only). Zero-offset check at 4 known poses: sim FK vs real within 1 cm / 2°. Safety items from section 7 in place before any motion.
- **M8b System ID:** step command per joint, log response, fit stiffness/damping/delay, update sim. Acceptance: sim vs real step response within 10%.
- **M8c Perception:** overhead camera intrinsics (`camera_calibration`), camera-to-base calibration with an ArUco board at a measured spot on the table, C++ ArUco node → `/object_pose` in base frame. Wrist camera intrinsics + hand-eye calibration (camera ↔ gripper) done now too, for later use. Acceptance: < 1 cm error over 20 placements across the spawn region.
- **M8d MoveIt on real:** home and place motions; controller switching between MoveIt and policy.
- **M8e Real single-arm test:** 50 trials. Acceptance: ≥ 70% success.

### M9. Two-arm handover: `SO101Handover-v1`

- Two arms at nominal base poses (update to measured poses once hardware is set up). Left arm picks and hands over; right arm takes and places.
- Object: apple-sized proxy that fits the gripper (check max opening at M1), with an ArUco marker.
- Centralised policy (D7), warm-started from single-arm weights where shapes allow; handover reward + arm-to-arm proximity penalty.
- Same path: sim ≥ 95% → randomised ≥ 90% → ONNX → real.

**Acceptance (sim):** ≥ 90% with randomisation over 500 episodes.

### M10. Task orchestration (C++)

- `so101_tasks`: BehaviorTree.CPP tree that chains skills: RL pick → RL handover → MoveIt scripted motions.
- Cutting is a scripted MoveIt motion with the knife held in a fixture, not RL. Feasibility risk: STS3215 torque is probably too low to cut a real apple. Test on soft material (banana, clay) first.
- New tasks are new trees reusing the same skill nodes.

## 7. Hardware safety (required before any real motion)

- Per-joint torque limit and max speed set in the servos.
- Software e-stop: one key/topic → torque off on all servos.
- Servo temperature and voltage monitoring; stop above threshold.
- Joint limits and workspace box enforced in the hardware interface and in the policy node.
- First real runs at reduced speed with a hand on the power switch.

## 8. Out of scope (for now)

Leader-arm teleop, real-world data collection, image-based policies (revisit after M9; the wrist camera makes this a natural next step), clutter, deformable objects in sim, RL for cutting, MoveIt Servo, Isaac Sim/Lab.

## 9. Open questions

1. Overhead camera: model, and height above the table (it must see both arms' workspaces for M9).
2. IMX335 lens FOV (from the listing, or measured in M8c).
3. STS3215 variant: 7.4 V or 12 V (affects torque in sim and cutting feasibility).
4. Raspberry Pi model.
5. Real cube: size and where the ArUco marker goes.
