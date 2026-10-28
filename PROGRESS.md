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

## M2. Single-arm pick-and-place task `SO101PickPlace-v1` — 2026-10-14

**Built**
- `so101_rl/envs/pick_place.py`: `SO101PickPlace-v1`, 200 steps, 20 Hz (sim 100 Hz).
  - Scene: ground plane = table top, arm base at the origin, 3 cm cube (27 g), 5 × 5 cm place zone (visual only).
  - Reset: arm at provisional home `[0, −0.5, −0.8, 1.2, 0, 0.6]` (gripper ~40 cm up, open ~6 cm) + N(0, 0.02 rad) noise. Cube uniform in x 12–32 cm, y ±10 cm, random yaw. Zone fixed at (15, 20) cm.
  - Region chosen by measurement: every point in x 10–34 cm, y ±12 cm at cube height is reachable (0 cm EE error); the gripper points near straight down (cos ≥ 0.9) out to x = 30 cm.
  - Observation (39-D): joint pos + vel (incl. gripper), TCP pose, cube position, cube yaw as sin/cos(4·yaw) (a cube repeats every 90°), zone position, TCP→cube, cube→zone, previous action. No `is_grasped` (not available on the real robot).
  - Action: `pd_joint_delta_pos` (now the agent's default mode), ±0.05 rad arm / ±0.2 rad gripper per step.
  - Reward: reach + grasp + lift (full at 4 cm) + move-to-zone (only while lifted, so dragging is not rewarded); once placed: 4 + open gripper + TCP above cube; success = 6. Penalties: 0.1·mean(a²), 0.5 while the cube falls outside the zone. Normalised by 6 → range [−0.1, 1].
  - Success: cube centre inside zone, resting on table (z < half + 5 mm, speed < 5 cm/s), gripper > 0.4 rad (open), TCP > 1 cm above cube top, not grasping.
- `scripts/smoke_test.py`: `--env` option.

**Gripper collision fix (asset change)**
- Scripted grasp check (move above cube, descend, close, lift) lifted **0/16** cubes with the M1 asset. Jaw-cube contacts were nearly horizontal, but the squeezed cube twisted and slipped out between the hull pieces; rendered frames showed the moving jaw riding over the cube.
- Tried and rejected: realistic gripper torque alone, deeper grasp (5/16), uncapped CoACD (2/16), wrist roll ±90° (0/16).
- Fix: `prepare_so101_asset.py` replaces the last 4.5 cm of each finger with a flat box pad flush with the finger's flat inner face (fixed 1.2 × 1.8 × 4.5 cm, moving 1.2 × 4.4 × 1.8 cm); the rest of each jaw stays CoACD (13 / 8 parts).
- Gripper force limit set to 2.5 N·m (real STS3215 ≈ 2–3 N·m); arm joints still 100 N·m.
- Same naive script after the fix (64 envs, no wrist rotation): **42/64** lifted with cube aligned to the jaws, **29/64** at random yaw (was 0). The policy can also rotate the wrist, which the script does not.

**Acceptance**
- `pytest tests/test_pick_place.py`: deterministic reset with fixed seed; success true for a cube placed in the zone (16/16 envs) and false at 3 points outside it; reward finite and within [−0.1, 1] for 100 random-action steps over 512 envs: **pass** (all 10 tests pass, including M1's).
- Random actions (policy action space), 1000 steps: 512 envs 12,648 FPS / 1,495 MiB; 1024 envs 19,090 FPS / 1,689 MiB. Fits under 3.5 GB: **pass**.

**Open issues**
- Home pose is provisional (overhead camera position unknown).
- Arm force limit (100 N·m) and gains still placeholders. The SO-101 MuJoCo model gives STS3215 values to start M8b from: kp 17.8 N·m/rad, force ±3.35 N·m, damping 0.6, armature 0.028, friction loss 0.052.
- The success predicate also accepts a cube pushed into the zone without lifting; check training videos in M4.

## M4. PPO training (state-based, no randomisation) — 2026-10-24

**Built**
- `scripts/train_ppo.py`: ManiSkill v3.0.1 `ppo.py` adapted (PPO update unchanged): YAML config, checkpoint every 10 min + best-by-eval, `--resume` (continue a run), `--init` (start from another run's weights).
- `configs/ppo_pickplace.yaml` (from scratch, 20M steps), `configs/ppo_pickplace_ft.yaml` (fine-tune, 10M steps), `configs/ppo_smoke.yaml`. 512 envs, 50 steps/rollout, γ 0.9, λ 0.9, 8 epochs, 32 minibatches, ent_coef 0.005.
- `scripts/evaluate.py`: N episodes at once with a fixed seed, ending at first success; success, grasp rate, episode length; 3 success + 2 failure videos.
- `scripts/play_policy.py`: watch a policy in the SAPIEN viewer (CPU sim; needs the GPU free of training, 4 GB is not enough for both).
- Env: PhysX GPU contact buffers enlarged (`GPUMemoryConfig`); 1024 envs still crashed with CUDA error 700 once the policy started grasping, 512 envs run stably.

**Training history (each run diagnosed with a 64-episode probe of the policy)**
| Run | Steps | Behaviour | Fix applied next |
|---|---|---|---|
| `ppo_pickplace_1792343056` | 20M | eval return ~35: shut the gripper early and pressed it on the cube; `is_grasping` fired on the cube's top edges (42/64) without holding it | grasp only counts if gripper > 0 rad; reach reward halved while the gripper is closed; ent_coef 0.005 |
| `ppo_pickplace_1792834675` | 20M | grasp 64/64, lift 59/64, carried over the zone (58/64) but held it ~35 cm high | carry/place reward = 3-D distance to the resting spot in the zone; lowering inside the zone keeps lift credit |
| `ppo_pickplace_ft_1792839283` (init from above) | 10M | lowered to ~5 cm over the zone, never released (56/64 still holding) | release low over the zone counts as placed immediately (no reward dip while the cube drops) |
| `ppo_pickplace_ft_1792841596` | 10M | first successes from ~7M steps; 80.5% on 512 episodes | more training |
| `ppo_pickplace_ft_1792844327` | 10M | 95.7% on 512 episodes (seed 2026), 94.0% on the seed-0 acceptance test | 64 eval envs; no training videos |
| `ppo_pickplace_ft_1792858771` | 10M | 96.2% (best.pt) — **M4 policy** | — |

Total ≈ 80M env steps, ~4.7k steps/s at 512 envs.

**Acceptance** (`python scripts/evaluate.py --ckpt <ckpt>`, 500 episodes, seed 0, no randomisation):
| Checkpoint | Success | Grasp | Mean ep. length |
|---|---|---|---|
| `ppo_pickplace_ft_1792844327/final.pt` | 94.0% | 99.8% | 107.8 |
| `ppo_pickplace_ft_1792858771/final.pt` | 95.8% | 100% | 100.4 |
| `ppo_pickplace_ft_1792858771/best.pt` | **96.2%** (481/500) | 100% | 103.7 (99.9 for successes) |

Target ≥ 95%: **pass**. M4 policy: `runs/ppo_pickplace_ft_1792858771/best.pt` (runs/ is not in git; keep a copy).
- With 16 eval envs `best.pt` was unreliable (86.7% vs 95.7% for `final.pt`); 64 eval envs fixed that.
- Training videos are off in the fine-tune config: 64 eval envs tiled into one video needed ~10 GB RAM and froze the laptop. `evaluate.py` saves small videos instead.

**Open issues**
- Both failure videos are carry failures at the workspace edge after a good grasp: one ends holding the cube just past the zone corner, one folds the arm over its base.
- `success_at_end` ≈ 0: after placing, the policy keeps moving and often knocks the cube. Fine for deployment (stop at success); worth a "stay still after placing" reward before M5.
- Raise `num_eval_envs` (16 → 64) so `best.pt` is meaningful.

## M5. Domain randomisation — 2026-10-28

**Built**
- `SO101PickPlace-v1(domain_rand=[...])`, any of 8 options or `all`:
  cube size 2.5–3.5 cm, mass ×0.5–1.5, friction ×0.5–1.5 (per env, at build time); per-joint action gain 0.8–1.2 (stands in for actuator stiffness/damping ±20%: per-env PD gains cannot be changed on the GPU after start); 1-step (50 ms) action delay in half the episodes; cube position ±5 mm / yaw ±5° noise; cube pose 0–2 steps stale; joint offsets ±2° + ±0.5° noise. Only observations are corrupted; reward and success use the true state.
- Observation 39 → 45: added the second-last action, so the policy can tell from the arm's motion whether its commands arrive late. `--init` widens a checkpoint's input layer with zero weights (identical behaviour at start, verified to 1e-6).
- `evaluate.py --dr`, `play_policy.py --dr`; older checkpoints still run (they use the first n_obs observations).
- `train_ppo.py` logs RAM and GPU memory every 10 iterations.

**Bugs found on the way**
- `_default_sim_config` had been indented inside `__init__`, so the larger PhysX buffers were never used (why 1024 envs crashed in M4). Fixed; buffers re-sized to fit 4 GB (1024 envs: 1.75 GB).
- Action delay replayed the episode's first command forever: delayed envs (half of them) never moved. Found from the per-option test (43.8%, i.e. ~0% for delayed episodes). The first M5 run trained on this and got worse (46% → 30%).
- The laptop froze at ~iteration 520 of that run: no swap configured; RAM rose 50% → 96% over a run, mostly outside the training process (it grows only ~1.2 MB/iteration). Added 8 GB swap; close TensorBoard during long runs.

**M4 policy under each randomisation alone** (500 episodes, seed 0, after the delay fix):
cube_friction 97.6%, cube_noise 96.8%, cube_latency 95.2%, joint_offset 95.0%, action_gain 94.2%, action_delay 89.6%; all on 78.8%.

**Training** (fine-tune, all randomisation on, 20M steps each)
| Run | Init | Result (500 ep., `--dr all`) |
|---|---|---|
| `ppo_pickplace_dr_1792916849` | M4 best | best.pt 84.2%, final.pt 77.2% |
| `ppo_pickplace_dr_1792929366` | above + prev_action2 | **best.pt 86.2%** (431/500), grasp 99.8%, mean length 128 steps |

Failure breakdown of the first run's best.pt (82 failures): 54 never got the cube to the zone (carry), 21 released over the zone but the cube did not rest properly, 7 never released; 72% of failures were delayed episodes (49% of all episodes).

**Acceptance:** ≥ 90% with all randomisation on — **not met (86.2%)**. Proceeding anyway (agreed 2026-10-28): the ranges are guesses until M8b measures real latency, servo response and camera noise; retrain against measured ranges then. M5 policy: `runs/ppo_pickplace_dr_1792929366/best.pt`.

## M6. Export and CPU inference — 2026-10-28

**Built**
- `scripts/export_onnx.py --ckpt <ckpt> --out models/so101_pickplace`: actor only (deterministic mean action) → `models/so101_pickplace.onnx`, plus `.json` with the observation layout (10 groups, 45 values, in order), action mapping (output clipped to [−1, 1] × [0.05 ×5, 0.2] rad per step, 20 Hz), home pose and zone centre. Uses PyTorch's legacy TorchScript exporter (works for this MLP; prints a deprecation warning).
- `scripts/evaluate.py --onnx <file>`: closed-loop evaluation with ONNX Runtime on the CPU.
- No observation normalisation stats: the PPO baseline trains on raw observations (`obs_normalisation: null` in the JSON).

**Acceptance** (M5 policy `runs/ppo_pickplace_dr_1792929366/best.pt`)
- Parity, 1000 random observations: max |ONNX − PyTorch| = 4.8e-6 (< 1e-4): **pass**.
- Latency, laptop CPU, batch 1: median 0.014 ms, p99 0.026 ms (< 2 ms): **pass**.
- Closed loop, 500 episodes, seed 0, all randomisation: ONNX 84.6% vs PyTorch 86.2% (within 2 points): **pass**.

## M7a/M7b. ROS 2 description and MoveIt config — 2026-10-28

**Built**
- `ros2_ws/src/so101_description`: URDF generated at build time from `assets/so101/so101.urdf` (single source of truth; only mesh paths become `package://` URIs), meshes, `display.launch.py` (robot_state_publisher + joint sliders + RViz).
- `ros2_ws/src/so101_moveit_config`: MoveIt Setup Assistant (groups `arm` = chain base_link→gripper_frame_link, `gripper`; poses home = sim home pose, zero, rest, open 0.6, closed −0.17; mock ros2_control hardware; JointTrajectoryControllers), with three fixes to the generated files:
  - `joint_limits.yaml`: integer limits made move_group exit at startup (Humble expects doubles); now 2.0 rad/s, 4.0 rad/s² with acceleration limits on.
  - `moveit_controllers.yaml`: added `action_ns: follow_joint_trajectory` + `default: true` (plan worked, execute did not).
  - `kinematics.yaml`: pick_ik position-only (`rotation_scale: 0`): a 5-DoF arm cannot reach arbitrary orientations.
- ROS installed packages: MoveIt + pick_ik from apt (Humble).

**Acceptance:** RViz shows the arm; MoveIt plans and executes home ↔ zero, gripper open/closed and dragged goals on mock hardware: **pass** (user check).

## M7c. Policy bridge (C++) — 2026-10-28

**Built** `ros2_ws/src/so101_policy_bridge`
- `policy_core` (no ROS): builds the 45-value observation exactly like the env (same order; TCP quaternion flipped to w ≥ 0 as in the sim; cube yaw as sin/cos(4·yaw); zone at the env's (0.15, 0.20, 0.0005); last two clipped actions), runs the ONNX actor with ONNX Runtime 1.23.2 C++ (prebuilt release in `third_party/`, not in git), and computes targets = q + action × [0.05 ×5, 0.2] rad.
- `bridge_node` (20 Hz): `/joint_states` (by joint name) + `/object_pose` (PoseStamped in base_link) + TF (TCP = midpoint of `Fixed_Jaw_tip`/`Moving_Jaw_tip`, orientation of `gripper_link`) → `Float64MultiArray` on `/forward_position_controller/commands`. `dry_run:=true` by default (logs only). Holds (publishes nothing, resets action history) if joint states > 100 ms or object pose > 200 ms old, TF missing, or the TCP leaves a box (x −0.05…0.45, y ±0.35, z 0…0.5 m). Targets clamped to URDF joint limits. Parameters in `config/bridge.yaml`.
- `scripts/record_replay.py`: 3 sim episodes (305 steps, all successful) driven by the ONNX policy → `test/data/replay.csv`.

**Acceptance**
- Replay test (`colcon test --packages-select so101_policy_bridge`): over 305 steps, C++ observation and action match the sim within 4.6e-5 (inputs agree to ~4e-8; the rest is float32 vs float64 rounding fed back through prev_action): **pass**.
- TF check: TCP from robot_state_publisher vs the sim's TCP for 11 recorded poses: 0.000 mm, quaternion 2e-7: **pass**.
- `ros2 launch so101_policy_bridge bridge.launch.py` runs without hardware: loads the model, dry run, watchdog holds with no input: **pass**.

**Open issues**
- No `forward_position_controller` is configured yet (MoveIt's mock setup only has trajectory controllers); add it with the real hardware interface (M8a) and switch controllers between MoveIt and the policy.
- D4 (keep running while the gripper hides the cube from the camera) not implemented: needs the real camera to see how often it happens.
- Testing tip: a running MoveIt demo publishes `/joint_states` too; run checks with a separate `ROS_DOMAIN_ID`.
