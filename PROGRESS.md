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
