"""PPO for SO101PickPlace-v1 (state obs).

Adapted from ManiSkill v3.0.1 examples/baselines/ppo/ppo.py. The PPO update is unchanged.
Changes: YAML config, so101 envs, checkpoint every N minutes + best-by-eval-success,
resume from `latest.pt` (cloud sessions get killed), no wandb.

Usage:
  python scripts/train_ppo.py --config configs/ppo_pickplace.yaml
  python scripts/train_ppo.py --config configs/ppo_pickplace.yaml --resume runs/<name>/latest.pt
  python scripts/train_ppo.py --config configs/ppo_pickplace_ft.yaml --init runs/<name>/final.pt
"""
import argparse
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import gymnasium as gym
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

import so101_rl.envs  # noqa: F401
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Config:
    exp_name: str = "ppo_pickplace"
    seed: int = 1
    env_id: str = "SO101PickPlace-v1"
    control_mode: str = "pd_joint_delta_pos"
    total_timesteps: int = 20_000_000
    learning_rate: float = 3e-4
    num_envs: int = 1024
    num_eval_envs: int = 16
    num_steps: int = 50
    num_eval_steps: int = 200
    gamma: float = 0.9
    gae_lambda: float = 0.9
    num_minibatches: int = 32
    update_epochs: int = 8
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.1
    reward_scale: float = 1.0
    eval_freq: int = 10  # iterations
    checkpoint_minutes: float = 10.0
    capture_video: bool = True
    domain_rand: tuple = ()  # names from so101_rl.envs.pick_place.DR_OPTIONS, or ["all"]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, n_act), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, n_act) * -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        mean = self.actor_mean(x)
        if deterministic:
            return mean
        return Normal(mean, torch.exp(self.actor_logstd.expand_as(mean))).sample()

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        probs = Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def gpu_used_mib():
    # Whole-GPU usage incl. PhysX and the desktop: a 4 GB laptop GPU shared with Xorg can freeze the screen.
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else -1


def load_config(path):
    cfg = Config()
    if path:
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(Config)}
        unknown = set(data) - known
        assert not unknown, f"unknown config keys: {unknown}"
        for k, v in data.items():
            setattr(cfg, k, v)
    return cfg


def make_envs(cfg, run_dir):
    kw = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", control_mode=cfg.control_mode,
              domain_rand=tuple(cfg.domain_rand))
    envs = gym.make(cfg.env_id, num_envs=cfg.num_envs, **kw)
    eval_envs = gym.make(cfg.env_id, num_envs=cfg.num_eval_envs, **kw)
    if cfg.capture_video:
        eval_envs = RecordEpisode(eval_envs, output_dir=str(run_dir / "videos"), save_trajectory=False,
                                  max_steps_per_video=cfg.num_eval_steps, video_fps=20)
    envs = ManiSkillVectorEnv(envs, cfg.num_envs, ignore_terminations=False, record_metrics=True)
    # Eval episodes run the full horizon; success_once counts any success within it.
    eval_envs = ManiSkillVectorEnv(eval_envs, cfg.num_eval_envs, ignore_terminations=True, record_metrics=True)
    return envs, eval_envs


def evaluate(agent, eval_envs, cfg):
    obs, _ = eval_envs.reset()
    metrics = defaultdict(list)
    for _ in range(cfg.num_eval_steps):
        with torch.no_grad():
            obs, _, _, _, infos = eval_envs.step(agent.get_action(obs, deterministic=True))
        if "final_info" in infos:
            for k, v in infos["final_info"]["episode"].items():
                metrics[k].append(v[infos["_final_info"]])
    return {k: torch.cat(v).float().mean().item() for k, v in metrics.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ppo_pickplace.yaml")
    p.add_argument("--resume", default=None, help="path to a latest.pt to continue training from")
    p.add_argument("--init", default=None, help="checkpoint to take policy weights from; fresh optimizer and step count")
    args = p.parse_args()
    cfg = load_config(args.config)

    batch_size = cfg.num_envs * cfg.num_steps
    minibatch_size = batch_size // cfg.num_minibatches
    num_iterations = cfg.total_timesteps // batch_size

    resume = torch.load(args.resume, weights_only=False) if args.resume else None
    run_dir = Path(args.resume).parent if resume else Path("runs") / f"{cfg.exp_name}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(vars(cfg)))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    envs, eval_envs = make_envs(cfg, run_dir)
    n_obs = int(np.prod(envs.single_observation_space.shape))
    n_act = int(np.prod(envs.single_action_space.shape))
    agent = Agent(n_obs, n_act).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    writer = SummaryWriter(str(run_dir))

    start_iter, global_step, best_success = 1, 0, -1.0
    if resume:
        agent.load_state_dict(resume["agent"])
        optimizer.load_state_dict(resume["optimizer"])
        start_iter, global_step, best_success = resume["iteration"] + 1, resume["global_step"], resume["best_success"]
        print(f"resumed from {args.resume} at iteration {resume['iteration']}, step {global_step}")
    elif args.init:
        state = torch.load(args.init, weights_only=False)["agent"]
        for net in ("actor_mean", "critic"):
            w = state[f"{net}.0.weight"]
            if w.shape[1] < n_obs:
                # New observations were appended at the end: zero weights for them, so the
                # network starts out behaving exactly like the checkpoint.
                state[f"{net}.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], n_obs - w.shape[1])], dim=1)
        agent.load_state_dict(state)
        print(f"initialised policy from {args.init}")

    def save(name, iteration):
        torch.save(dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration,
                        global_step=global_step, best_success=best_success, n_obs=n_obs, n_act=n_act),
                   run_dir / name)

    obs = torch.zeros((cfg.num_steps, cfg.num_envs, n_obs), device=device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs, n_act), device=device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)

    act_low = torch.from_numpy(envs.single_action_space.low).to(device)
    act_high = torch.from_numpy(envs.single_action_space.high).to(device)
    next_obs, _ = envs.reset(seed=cfg.seed + start_iter)  # new seed on resume, avoids replaying episodes
    eval_envs.reset(seed=cfg.seed)
    next_done = torch.zeros(cfg.num_envs, device=device)
    print(f"run dir {run_dir}: {num_iterations} iterations, batch {batch_size}, minibatch {minibatch_size}")

    start_time, last_ckpt = time.time(), time.time()
    steps_at_start = global_step
    for iteration in range(start_iter, num_iterations + 1):
        agent.eval()
        if iteration % cfg.eval_freq == 1 or iteration == num_iterations:
            m = evaluate(agent, eval_envs, cfg)
            for k, v in m.items():
                writer.add_scalar(f"eval/{k}", v, global_step)
            success = m.get("success_once", 0.0)
            print(f"iter {iteration} step {global_step}: eval success_once {success:.3f} "
                  f"success_at_end {m.get('success_at_end', 0):.3f} return {m.get('return', 0):.2f}")
            if success > best_success:
                best_success = success
                save("best.pt", iteration)

        final_values = torch.zeros((cfg.num_steps, cfg.num_envs), device=device)
        rollout_t = time.time()
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(torch.clamp(action, act_low, act_high))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * cfg.reward_scale
            if "final_info" in infos:
                done_mask = infos["_final_info"]
                for k, v in infos["final_info"]["episode"].items():
                    writer.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(cfg.num_envs, device=device)[done_mask]] = \
                        agent.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_t = time.time() - rollout_t

        # GAE with bootstrapping at done (same as the baseline).
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    next_not_done, nextvalues = 1.0 - next_done, next_value
                else:
                    next_not_done, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + cfg.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape(-1, n_obs)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1, n_act)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs = []
        update_t = time.time()
        for epoch in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    break
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break
        update_t = time.time() - update_t

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        sps = int((global_step - steps_at_start) / (time.time() - start_time))
        for k, v in dict(value_loss=v_loss.item(), policy_loss=pg_loss.item(), entropy=entropy_loss.item(),
                         approx_kl=approx_kl.item(), clipfrac=np.mean(clipfracs),
                         explained_variance=np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y).items():
            writer.add_scalar(f"losses/{k}", v, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("time/rollout_time", rollout_t, global_step)
        writer.add_scalar("time/update_time", update_t, global_step)
        if iteration % 10 == 0:
            ram = psutil.virtual_memory().percent
            gpu = gpu_used_mib()
            writer.add_scalar("system/ram_percent", ram, global_step)
            writer.add_scalar("system/gpu_used_mib", gpu, global_step)
            print(f"iter {iteration}/{num_iterations} step {global_step} SPS {sps} RAM {ram:.0f}% GPU {gpu} MiB")

        if time.time() - last_ckpt > cfg.checkpoint_minutes * 60:
            save("latest.pt", iteration)
            last_ckpt = time.time()

    save("latest.pt", num_iterations)
    save("final.pt", num_iterations)
    writer.close()
    envs.close()
    eval_envs.close()


if __name__ == "__main__":
    main()
