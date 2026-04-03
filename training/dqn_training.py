"""
dqn_training.py — Train DQN on the Code Mentorship Environment.

Uses Stable-Baselines3.  Runs 10 hyperparameter combinations and saves
results to CSV + plots.

Usage:
    python training/dqn_training.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import CodeMentorshipEnv

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = "models/dqn"
PLOT_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Hyperparameter grid (10 runs) ──────────────────────────────────────────────
HYPERPARAM_GRID = [
    # lr,     gamma, buffer,  batch, eps_start, eps_end, target_update, net_arch, tau
    (1e-3, 0.95,  50_000, 64,  1.0, 0.05, 500,  [64, 64],   1.0),
    (5e-4, 0.99,  50_000, 64,  1.0, 0.05, 500,  [64, 64],   1.0),
    (1e-3, 0.99, 100_000, 128, 1.0, 0.05, 1000, [128, 128], 1.0),
    (1e-4, 0.95,  50_000, 32,  1.0, 0.10, 500,  [64, 64],   1.0),
    (1e-3, 0.99, 100_000, 64,  1.0, 0.02, 1000, [256, 256], 1.0),
    (5e-4, 0.95,  80_000, 128, 1.0, 0.05, 800,  [128, 64],  0.05),
    (1e-3, 0.99,  50_000, 64,  1.0, 0.05, 500,  [128, 128], 0.1),
    (2e-4, 0.99, 100_000, 64,  1.0, 0.02, 1000, [64, 64],   1.0),
    (1e-3, 0.90,  50_000, 32,  1.0, 0.10, 500,  [64, 32],   1.0),
    (5e-3, 0.95,  30_000, 128, 1.0, 0.05, 300,  [256, 128], 1.0),
]

TRAIN_STEPS = 200_000
EVAL_EPISODES = 20


class RewardLoggerCallback(BaseCallback):
    """Logs mean episode reward every N steps."""

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rewards = []
        self._ep_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
        if self.n_calls % self.log_freq == 0 and self._ep_rewards:
            self.rewards.append(np.mean(self._ep_rewards[-20:]))
            self._ep_rewards = []
        return True


def make_env(seed=0):
    env = CodeMentorshipEnv(render_mode=None)
    env = Monitor(env)
    return env


def train_dqn_run(run_id: int, params: tuple) -> dict:
    lr, gamma, buffer, batch, eps_start, eps_end, target_update, net_arch, tau = params

    print(f"\n[DQN Run {run_id+1}/10] lr={lr} gamma={gamma} buffer={buffer} "
          f"batch={batch} eps_end={eps_end} arch={net_arch}")

    env = make_env(seed=run_id)
    eval_env = make_env(seed=run_id + 100)

    callback = RewardLoggerCallback(log_freq=2000)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        gamma=gamma,
        buffer_size=buffer,
        batch_size=batch,
        exploration_initial_eps=eps_start,
        exploration_final_eps=eps_end,
        target_update_interval=target_update,
        tau=tau,
        policy_kwargs={"net_arch": net_arch},
        verbose=0,
        seed=run_id,
        device="cpu",
    )

    t0 = time.time()
    model.learn(total_timesteps=TRAIN_STEPS, callback=callback, progress_bar=False)
    train_time = time.time() - t0

    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
    )

    # Convergence: step where reward first exceeds 70% of best reward
    rewards = callback.rewards
    convergence_step = None
    if rewards:
        threshold = 0.70 * max(rewards)
        for i, r in enumerate(rewards):
            if r >= threshold:
                convergence_step = i * 2000
                break

    # Save best model
    save_path = os.path.join(MODEL_DIR, f"dqn_run{run_id+1}")
    model.save(save_path)

    result = {
        "run": run_id + 1,
        "learning_rate": lr,
        "gamma": gamma,
        "buffer_size": buffer,
        "batch_size": batch,
        "eps_end": eps_end,
        "target_update_interval": target_update,
        "net_arch": str(net_arch),
        "tau": tau,
        "mean_reward": round(mean_reward, 3),
        "std_reward": round(std_reward, 3),
        "convergence_step": convergence_step,
        "train_time_s": round(train_time, 1),
        "reward_curve": rewards,
    }

    env.close()
    eval_env.close()
    print(f"  ↳ Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}  |  "
          f"Converged @ step: {convergence_step}  |  Time: {train_time:.1f}s")
    return result


def plot_dqn_results(results: list):
    plt.style.use("dark_background")

    # ── 1. Reward curves ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), facecolor="#0d1117")
    fig.suptitle("DQN — Training Reward Curves (All 10 Runs)",
                 fontsize=14, color="#64dcff", fontfamily="monospace")

    for i, r in enumerate(results):
        ax = axes[i // 5][i % 5]
        ax.set_facecolor("#161b27")
        curve = r["reward_curve"]
        if curve:
            xs = [j * 2000 for j in range(len(curve))]
            ax.plot(xs, curve, color="#5096ff", linewidth=1.5)
            ax.fill_between(xs, curve, alpha=0.15, color="#5096ff")
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
    path = os.path.join(PLOT_DIR, "dqn_reward_curves.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[plot] Saved {path}")

    # ── 2. Final mean reward comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b27")
    runs = [r["run"] for r in results]
    means = [r["mean_reward"] for r in results]
    stds = [r["std_reward"] for r in results]
    colors = ["#50c8a0" if m == max(means) else "#5096ff" for m in means]
    ax.bar([f"R{r}" for r in runs], means, yerr=stds, color=colors,
           edgecolor="#333", capsize=5, linewidth=0.8)
    ax.set_title("DQN — Final Mean Reward per Run",
                 color="#64dcff", fontsize=13, fontfamily="monospace")
    ax.set_ylabel("Mean Reward", color="#aaa")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    plt.tight_layout()
    path2 = os.path.join(PLOT_DIR, "dqn_final_rewards.png")
    plt.savefig(path2, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[plot] Saved {path2}")

    # ── 3. Best run DQN objective (loss proxy) ────────────────────────────────
    best_run = max(results, key=lambda x: x["mean_reward"])
    curve = best_run["reward_curve"]
    if curve:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
        ax.set_facecolor("#161b27")
        xs = [j * 2000 for j in range(len(curve))]
        ax.plot(xs, curve, color="#ffaa40", linewidth=2)
        ax.fill_between(xs, curve, alpha=0.2, color="#ffaa40")
        if best_run["convergence_step"]:
            ax.axvline(best_run["convergence_step"], color="#ff6080",
                       linestyle="--", linewidth=1.5, label=f"Convergence @ {best_run['convergence_step']}")
        ax.legend(fontsize=9)
        ax.set_title(
            f"DQN Best Run (Run {best_run['run']}) — Objective Curve\n"
            f"lr={best_run['learning_rate']}, γ={best_run['gamma']}, "
            f"Mean Reward={best_run['mean_reward']:.3f}",
            color="#ffaa40", fontsize=11, fontfamily="monospace"
        )
        ax.set_xlabel("Timesteps", color="#aaa")
        ax.set_ylabel("Mean Episode Reward", color="#aaa")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        plt.tight_layout()
        path3 = os.path.join(PLOT_DIR, "dqn_best_objective.png")
        plt.savefig(path3, dpi=130, bbox_inches="tight", facecolor="#0d1117")
        plt.close()
        print(f"[plot] Saved {path3}")


def save_results_csv(results: list):
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "reward_curve"}
        rows.append(row)
    df = pd.DataFrame(rows)
    path = os.path.join(PLOT_DIR, "dqn_results.csv")
    df.to_csv(path, index=False)
    print(f"[csv] Saved {path}")
    return df


def run_all():
    print("=" * 60)
    print("DQN Hyperparameter Sweep — Code Mentorship Environment")
    print("=" * 60)
    all_results = []
    for i, params in enumerate(HYPERPARAM_GRID):
        result = train_dqn_run(i, params)
        all_results.append(result)

    df = save_results_csv(all_results)
    plot_dqn_results(all_results)

    best = max(all_results, key=lambda x: x["mean_reward"])
    print(f"\n✅ Best DQN Run: Run {best['run']}  "
          f"| Mean Reward: {best['mean_reward']:.3f} ± {best['std_reward']:.3f}")

    # Save best run info
    with open(os.path.join(MODEL_DIR, "best_run.json"), "w") as f:
        info = {k: v for k, v in best.items() if k != "reward_curve"}
        json.dump(info, f, indent=2)

    print("\n[DQN] All runs complete.")
    print(df[["run", "learning_rate", "gamma", "buffer_size",
              "batch_size", "eps_end", "mean_reward", "std_reward",
              "convergence_step"]].to_string(index=False))

    return all_results, best


if __name__ == "__main__":
    run_all()
