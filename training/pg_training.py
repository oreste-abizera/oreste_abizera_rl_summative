"""
pg_training.py — Train REINFORCE and PPO on the Code Mentorship Environment.

Uses Stable-Baselines3 for PPO.
REINFORCE is implemented from scratch (SB3 does not ship REINFORCE).

Each algorithm runs 10 hyperparameter combinations.

Usage:
    python training/pg_training.py
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from environment.custom_env import CodeMentorshipEnv

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR_PPO = "models/pg/ppo"
MODEL_DIR_REINFORCE = "models/pg/reinforce"
PLOT_DIR = "plots"
for d in [MODEL_DIR_PPO, MODEL_DIR_REINFORCE, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_STEPS = 200_000
EVAL_EPISODES = 20


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared callback
# ═══════════════════════════════════════════════════════════════════════════════

class EntropyRewardLogger(BaseCallback):
    """Log episode rewards + policy entropy (where available)."""

    def __init__(self, log_freq=2000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []
        self.entropy_vals = []
        self._ep_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])

        if self.n_calls % self.log_freq == 0 and self._ep_rewards:
            self.rewards.append(np.mean(self._ep_rewards[-20:]))
            self._ep_rewards = []

        # Entropy from SB3 logger
        if hasattr(self.model, "logger") and self.model.logger is not None:
            try:
                ent = self.model.logger.name_to_value.get("train/entropy_loss", None)
                if ent is not None:
                    self.entropy_vals.append(float(ent))
            except Exception:
                pass
        return True


def make_env(seed=0):
    env = CodeMentorshipEnv(render_mode=None)
    env = Monitor(env)
    return env


def convergence_step(rewards, threshold_pct=0.70, log_freq=2000):
    if not rewards:
        return None
    threshold = threshold_pct * max(rewards)
    for i, r in enumerate(rewards):
        if r >= threshold:
            return i * log_freq
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  REINFORCE (from scratch)
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


def reinforce_train_run(run_id: int, params: dict) -> dict:
    lr = params["lr"]
    gamma = params["gamma"]
    hidden = params["hidden"]
    n_episodes = params["n_episodes"]
    entropy_coef = params["entropy_coef"]
    baseline = params.get("baseline", True)

    print(f"\n[REINFORCE Run {run_id+1}/10] lr={lr} gamma={gamma} "
          f"hidden={hidden} episodes={n_episodes} entropy={entropy_coef} "
          f"baseline={baseline}")

    env = CodeMentorshipEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim, hidden)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    reward_log = []
    entropy_log = []
    all_ep_rewards = []

    t0 = time.time()
    baseline_val = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        log_probs, rewards_ep, entropies = [], [], []

        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            probs = policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards_ep.append(reward)
            done = terminated or truncated

        # Compute returns
        G = 0.0
        returns = []
        for r in reversed(rewards_ep):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)

        if baseline:
            baseline_val = 0.9 * baseline_val + 0.1 * returns_t.mean().item()
            returns_t = returns_t - baseline_val

        # Normalise
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy loss
        policy_loss = 0.0
        for lp, G_t, ent in zip(log_probs, returns_t, entropies):
            policy_loss += -lp * G_t - entropy_coef * ent

        optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        ep_reward = sum(rewards_ep)
        all_ep_rewards.append(ep_reward)
        entropy_log.append(float(torch.stack(entropies).mean().item()))

        if (ep + 1) % 50 == 0:
            mean_r = np.mean(all_ep_rewards[-50:])
            reward_log.append(mean_r)

    train_time = time.time() - t0

    # Evaluate
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = policy(obs_t)
            action = probs.argmax(dim=-1).item()
            obs, r, te, tr, _ = env.step(action)
            total += r
            done = te or tr
        eval_rewards.append(total)

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    # Save model
    save_path = os.path.join(MODEL_DIR_REINFORCE, f"reinforce_run{run_id+1}.pt")
    torch.save(policy.state_dict(), save_path)

    env.close()
    print(f"  ↳ Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}  "
          f"| Time: {train_time:.1f}s")

    return {
        "run": run_id + 1,
        "learning_rate": lr,
        "gamma": gamma,
        "hidden_layers": str(hidden),
        "n_episodes": n_episodes,
        "entropy_coef": entropy_coef,
        "baseline": baseline,
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "convergence_step": convergence_step(reward_log, log_freq=50),
        "train_time_s": round(train_time, 1),
        "reward_curve": reward_log,
        "entropy_curve": entropy_log[::50] if entropy_log else [],
    }


REINFORCE_GRID = [
    {"lr": 1e-3,  "gamma": 0.99, "hidden": [64, 64],   "n_episodes": 1000, "entropy_coef": 0.01, "baseline": True},
    {"lr": 5e-4,  "gamma": 0.99, "hidden": [128, 128], "n_episodes": 1000, "entropy_coef": 0.01, "baseline": True},
    {"lr": 1e-3,  "gamma": 0.95, "hidden": [64, 64],   "n_episodes": 1200, "entropy_coef": 0.05, "baseline": True},
    {"lr": 2e-3,  "gamma": 0.99, "hidden": [64, 32],   "n_episodes": 800, "entropy_coef": 0.01, "baseline": False},
    {"lr": 1e-4,  "gamma": 0.99, "hidden": [128, 64],  "n_episodes": 2000,"entropy_coef": 0.001,"baseline": True},
    {"lr": 1e-3,  "gamma": 0.90, "hidden": [64, 64],   "n_episodes": 1000, "entropy_coef": 0.02, "baseline": True},
    {"lr": 5e-4,  "gamma": 0.95, "hidden": [256, 128], "n_episodes": 1000, "entropy_coef": 0.01, "baseline": True},
    {"lr": 1e-3,  "gamma": 0.99, "hidden": [64, 64],   "n_episodes": 1000, "entropy_coef": 0.0,  "baseline": True},
    {"lr": 3e-4,  "gamma": 0.99, "hidden": [128, 128], "n_episodes": 1500, "entropy_coef": 0.05, "baseline": False},
    {"lr": 1e-3,  "gamma": 0.99, "hidden": [64, 64],   "n_episodes": 1000, "entropy_coef": 0.01, "baseline": True},  # replicate best
]


# ═══════════════════════════════════════════════════════════════════════════════
#  PPO
# ═══════════════════════════════════════════════════════════════════════════════

PPO_GRID = [
    # lr,   gamma, n_steps, batch, n_epochs, ent_coef, clip, gae_lambda, arch
    (3e-4, 0.99, 2048, 64,  10, 0.01, 0.2, 0.95, [64, 64]),
    (1e-4, 0.99, 2048, 64,  10, 0.01, 0.2, 0.95, [128, 128]),
    (3e-4, 0.95, 1024, 32,  5,  0.01, 0.1, 0.90, [64, 64]),
    (3e-4, 0.99, 2048, 128, 10, 0.05, 0.2, 0.95, [128, 128]),
    (1e-3, 0.99, 2048, 64,  10, 0.01, 0.3, 0.95, [64, 64]),
    (3e-4, 0.99, 4096, 64,  5,  0.00, 0.2, 0.95, [256, 256]),
    (5e-4, 0.99, 2048, 64,  10, 0.02, 0.2, 0.98, [128, 64]),
    (3e-4, 0.90, 2048, 64,  10, 0.01, 0.2, 0.95, [64, 64]),
    (3e-4, 0.99, 1024, 64,  15, 0.01, 0.15,0.95, [64, 64]),
    (2e-4, 0.99, 2048, 128, 10, 0.03, 0.2, 0.95, [256, 128]),
]


def train_ppo_run(run_id: int, params: tuple) -> dict:
    lr, gamma, n_steps, batch, n_epochs, ent_coef, clip, gae_lambda, arch = params

    print(f"\n[PPO Run {run_id+1}/10] lr={lr} gamma={gamma} n_steps={n_steps} "
          f"batch={batch} ent_coef={ent_coef} clip={clip}")

    env = make_env(seed=run_id)
    eval_env = make_env(seed=run_id + 200)
    callback = EntropyRewardLogger(log_freq=2000)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=lr, gamma=gamma, n_steps=n_steps,
        batch_size=batch, n_epochs=n_epochs, ent_coef=ent_coef,
        clip_range=clip, gae_lambda=gae_lambda,
        policy_kwargs={"net_arch": [{"pi": arch, "vf": arch}]},
        verbose=0, seed=run_id, device="cpu",
    )

    t0 = time.time()
    model.learn(TRAIN_STEPS, callback=callback, progress_bar=False)
    train_time = time.time() - t0

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    save_path = os.path.join(MODEL_DIR_PPO, f"ppo_run{run_id+1}")
    model.save(save_path)

    env.close(); eval_env.close()
    print(f"  ↳ Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}  "
          f"| Time: {train_time:.1f}s")

    return {
        "run": run_id + 1,
        "learning_rate": lr, "gamma": gamma, "n_steps": n_steps,
        "batch_size": batch, "n_epochs": n_epochs,
        "ent_coef": ent_coef, "clip_range": clip, "gae_lambda": gae_lambda,
        "net_arch": str(arch),
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "convergence_step": convergence_step(callback.rewards),
        "train_time_s": round(train_time, 1),
        "reward_curve": callback.rewards,
        "entropy_curve": callback.entropy_vals,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def plot_algorithm_curves(results: list, algo_name: str, color: str, log_freq: int = 2000):
    plt.style.use("dark_background")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="#0d1117")
    fig.suptitle(f"{algo_name} — Training Reward Curves (All 10 Runs)",
                 fontsize=14, color="#64dcff", fontfamily="monospace")

    for i, r in enumerate(results):
        ax = axes[i // 5][i % 5]
        ax.set_facecolor("#161b27")
        curve = r["reward_curve"]
        if curve:
            xs = [j * log_freq for j in range(len(curve))]
            ax.plot(xs, curve, color=color, linewidth=1.5)
            ax.fill_between(xs, curve, alpha=0.15, color=color)
        ax.set_title(
            f"Run {r['run']}\nlr={r['learning_rate']}, γ={r['gamma']}",
            color="#aaaaaa", fontsize=8, fontfamily="monospace"
        )
        ax.set_xlabel("Steps", fontsize=7, color="#666")
        ax.set_ylabel("Mean Reward", fontsize=7, color="#666")
        ax.tick_params(labelsize=7, colors="#666")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{algo_name.lower()}_reward_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[plot] Saved {path}")


def plot_entropy_curves(results: list, algo_name: str, color: str):
    curves = [r for r in results if r.get("entropy_curve")]
    if not curves:
        return
    best = max(results, key=lambda x: x["mean_reward"])
    curve = best.get("entropy_curve", [])
    if not curve:
        return
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b27")
    ax.plot(curve, color=color, linewidth=1.8)
    ax.fill_between(range(len(curve)), curve, alpha=0.15, color=color)
    ax.set_title(
        f"{algo_name} Best Run (Run {best['run']}) — Policy Entropy",
        color=color, fontsize=12, fontfamily="monospace"
    )
    ax.set_xlabel("Log Step (x2000)", color="#aaa")
    ax.set_ylabel("Entropy", color="#aaa")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"{algo_name.lower()}_entropy.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[plot] Saved {path}")


def save_pg_csv(results: list, name: str):
    rows = [{k: v for k, v in r.items() if k not in ("reward_curve", "entropy_curve")}
            for r in results]
    df = pd.DataFrame(rows)
    path = os.path.join(PLOT_DIR, f"{name}_results.csv")
    df.to_csv(path, index=False)
    print(f"[csv] Saved {path}")
    return df


def plot_combined_comparison(reinforce_results, ppo_results):
    """Plot all algorithms' best runs side-by-side."""
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
    fig.suptitle("Policy Gradient Methods — Best Run Comparison",
                 fontsize=14, color="#64dcff", fontfamily="monospace")

    algo_data = [
        ("REINFORCE", reinforce_results, "#c880ff", 50),
        ("PPO",       ppo_results,       "#50c8a0", 2000),
    ]

    for ax, (name, results, color, freq) in zip(axes, algo_data):
        ax.set_facecolor("#161b27")
        best = max(results, key=lambda x: x["mean_reward"])
        curve = best["reward_curve"]
        if curve:
            xs = [j * freq for j in range(len(curve))]
            ax.plot(xs, curve, color=color, linewidth=2)
            ax.fill_between(xs, curve, alpha=0.2, color=color)
            if best["convergence_step"]:
                ax.axvline(best["convergence_step"], color="#ff6080",
                           linestyle="--", linewidth=1.2,
                           label=f"Converge @ {best['convergence_step']}")
        ax.set_title(
            f"{name}\nBest: Run {best['run']} | "
            f"Reward: {best['mean_reward']:.2f}",
            color=color, fontsize=10, fontfamily="monospace"
        )
        ax.set_xlabel("Steps / Episodes", color="#aaa", fontsize=9)
        ax.set_ylabel("Mean Reward", color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa")
        ax.legend(fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "pg_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[plot] Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_all():
    print("=" * 60)
    print("Policy Gradient Hyperparameter Sweep")
    print("Algorithms: REINFORCE, PPO")
    print("=" * 60)

    # ── REINFORCE ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("REINFORCE")
    print("─" * 40)
    reinforce_results = []
    for i, params in enumerate(REINFORCE_GRID):
        reinforce_results.append(reinforce_train_run(i, params))
    save_pg_csv(reinforce_results, "reinforce")
    plot_algorithm_curves(reinforce_results, "REINFORCE", "#c880ff", log_freq=50)
    plot_entropy_curves(reinforce_results, "REINFORCE", "#c880ff")

    best_r = max(reinforce_results, key=lambda x: x["mean_reward"])
    with open(os.path.join(MODEL_DIR_REINFORCE, "best_run.json"), "w") as f:
        json.dump({k: v for k, v in best_r.items()
                   if k not in ("reward_curve", "entropy_curve")}, f, indent=2)

    # ── PPO ───────────────────────────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("PPO")
    print("─" * 40)
    ppo_results = []
    for i, params in enumerate(PPO_GRID):
        ppo_results.append(train_ppo_run(i, params))
    save_pg_csv(ppo_results, "ppo")
    plot_algorithm_curves(ppo_results, "PPO", "#50c8a0", log_freq=2000)
    plot_entropy_curves(ppo_results, "PPO", "#50c8a0")

    best_ppo = max(ppo_results, key=lambda x: x["mean_reward"])
    with open(os.path.join(MODEL_DIR_PPO, "best_run.json"), "w") as f:
        json.dump({k: v for k, v in best_ppo.items()
                   if k not in ("reward_curve", "entropy_curve")}, f, indent=2)

    # ── Combined comparison ────────────────────────────────────────────────────
    plot_combined_comparison(reinforce_results, ppo_results)

    print("\n" + "=" * 60)
    print("POLICY GRADIENT SWEEP COMPLETE")
    print(f"  Best REINFORCE: Run {best_r['run']}  "
          f"| Reward: {best_r['mean_reward']:.3f}")
    print(f"  Best PPO:       Run {best_ppo['run']}  "
          f"| Reward: {best_ppo['mean_reward']:.3f}")
    print("=" * 60)

    return reinforce_results, ppo_results


if __name__ == "__main__":
    run_all()
