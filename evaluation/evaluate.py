"""
evaluation/evaluate.py — Post-training evaluation and visualization.

Generates:
  - Cumulative reward plots (all 4 algorithms, best run each)
  - Convergence comparison plot
  - Generalization test (unseen initial states)
  - Combined summary figure

Usage:
    python evaluation/evaluate.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import CodeMentorshipEnv
from training.pg_training import PolicyNet

PLOT_DIR = "plots"
MODEL_DIR = "models"
os.makedirs(PLOT_DIR, exist_ok=True)

N_EVAL = 30
GENERALIZATION_SEEDS = list(range(200, 230))  # unseen seeds


# ──────────────────────────────────────────────────────────────────────────────
# Load best models
# ──────────────────────────────────────────────────────────────────────────────

def load_best_dqn():
    info_path = os.path.join(MODEL_DIR, "dqn", "best_run.json")
    if not os.path.exists(info_path):
        return None, None
    with open(info_path) as f:
        info = json.load(f)
    run = info["run"]
    model_path = os.path.join(MODEL_DIR, "dqn", f"dqn_run{run}")
    try:
        env = Monitor(CodeMentorshipEnv())
        model = DQN.load(model_path, env=env)
        return model, info
    except Exception as e:
        print(f"[warn] DQN load failed: {e}")
        return None, None


def load_best_ppo():
    info_path = os.path.join(MODEL_DIR, "pg", "ppo", "best_run.json")
    if not os.path.exists(info_path):
        return None, None
    with open(info_path) as f:
        info = json.load(f)
    run = info["run"]
    model_path = os.path.join(MODEL_DIR, "pg", "ppo", f"ppo_run{run}")
    try:
        env = Monitor(CodeMentorshipEnv())
        model = PPO.load(model_path, env=env)
        return model, info
    except Exception as e:
        print(f"[warn] PPO load failed: {e}")
        return None, None


def load_best_reinforce():
    info_path = os.path.join(MODEL_DIR, "pg", "reinforce", "best_run.json")
    if not os.path.exists(info_path):
        return None, None
    with open(info_path) as f:
        info = json.load(f)
    run = info["run"]
    model_path = os.path.join(MODEL_DIR, "pg", "reinforce", f"reinforce_run{run}.pt")
    hidden_str = info.get("hidden_layers", "[64, 64]")
    hidden = eval(hidden_str)
    try:
        obs_dim = CodeMentorshipEnv().observation_space.shape[0]
        act_dim = CodeMentorshipEnv().action_space.n
        net = PolicyNet(obs_dim, act_dim, hidden)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
        return net, info
    except Exception as e:
        print(f"[warn] REINFORCE load failed: {e}")
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Run episode helpers
# ──────────────────────────────────────────────────────────────────────────────

def run_sb3_episode(model, seed):
    env = CodeMentorshipEnv()
    obs, _ = env.reset(seed=seed)
    done = False
    rewards = []
    cum = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, te, tr, _ = env.step(int(action))
        cum += r
        rewards.append(cum)
        done = te or tr
    env.close()
    return rewards


def run_reinforce_episode(net, seed):
    env = CodeMentorshipEnv()
    obs, _ = env.reset(seed=seed)
    done = False
    rewards = []
    cum = 0.0
    while not done:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = net(obs_t)
        action = probs.argmax(dim=-1).item()
        obs, r, te, tr, _ = env.step(action)
        cum += r
        rewards.append(cum)
        done = te or tr
    env.close()
    return rewards


def run_random_episode(seed):
    env = CodeMentorshipEnv()
    obs, _ = env.reset(seed=seed)
    done = False
    rewards = []
    cum = 0.0
    while not done:
        action = env.action_space.sample()
        obs, r, te, tr, _ = env.step(action)
        cum += r
        rewards.append(cum)
        done = te or tr
    env.close()
    return rewards


# ──────────────────────────────────────────────────────────────────────────────
# 1. Cumulative reward comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_cumulative_rewards(models_dict):
    """
    models_dict: {name: (run_fn, color)}
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 5),
                             facecolor="#0d1117")
    if len(models_dict) == 1:
        axes = [axes]

    for ax, (name, (run_fn, color)) in zip(axes, models_dict.items()):
        ax.set_facecolor("#161b27")
        episode_curves = []
        for seed in range(N_EVAL):
            curve = run_fn(seed)
            # Pad / truncate to common length for mean
            episode_curves.append(curve)

        max_len = max(len(c) for c in episode_curves)
        padded = [c + [c[-1]] * (max_len - len(c)) for c in episode_curves]
        arr = np.array(padded)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        steps = np.arange(max_len)
        ax.plot(steps, mean, color=color, linewidth=2, label="Mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_title(f"{name}\nMean Final: {mean[-1]:.2f}",
                     color=color, fontsize=11, fontfamily="monospace")
        ax.set_xlabel("Step within Episode", color="#aaa", fontsize=9)
        ax.set_ylabel("Cumulative Reward", color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    fig.suptitle("Cumulative Rewards — Best Model per Algorithm",
                 color="#64dcff", fontsize=13, fontfamily="monospace")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "cumulative_rewards.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[eval] Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Convergence comparison (episodes to stable performance)
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence_comparison():
    """Load saved CSVs and compare convergence steps."""
    import pandas as pd

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b27")

    algo_colors = {
        "DQN": "#5096ff",
        "REINFORCE": "#c880ff",
        "PPO": "#50c8a0",
    }
    algo_files = {
        "DQN": os.path.join(PLOT_DIR, "dqn_results.csv"),
        "REINFORCE": os.path.join(PLOT_DIR, "reinforce_results.csv"),
        "PPO": os.path.join(PLOT_DIR, "ppo_results.csv"),
    }

    for algo, fpath in algo_files.items():
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        if "convergence_step" not in df.columns:
            continue
        valid = df["convergence_step"].dropna()
        if valid.empty:
            continue
        ax.scatter(
            [algo] * len(valid), valid,
            color=algo_colors[algo], alpha=0.7, s=80, zorder=3
        )
        ax.scatter(
            [algo], [valid.mean()],
            color=algo_colors[algo], marker="D", s=150, zorder=4,
            label=f"{algo} mean={valid.mean():.0f}"
        )

    ax.set_title("Convergence Step per Algorithm (All Runs)",
                 color="#64dcff", fontsize=12, fontfamily="monospace")
    ax.set_ylabel("Step to Convergence (70% of best reward)", color="#aaa")
    ax.legend(fontsize=9)
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "convergence_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[eval] Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Generalization test
# ──────────────────────────────────────────────────────────────────────────────

def generalization_test(models_dict):
    """Test on 30 unseen initial seeds."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b27")

    summary = {}
    for name, (run_fn, color) in models_dict.items():
        ep_rewards = []
        for seed in GENERALIZATION_SEEDS:
            curve = run_fn(seed)
            ep_rewards.append(curve[-1])
        summary[name] = {"mean": np.mean(ep_rewards), "std": np.std(ep_rewards),
                         "all": ep_rewards, "color": color}

    names = list(summary.keys())
    means = [summary[n]["mean"] for n in names]
    stds = [summary[n]["std"] for n in names]
    colors = [summary[n]["color"] for n in names]

    ax.bar(names, means, yerr=stds, color=colors, edgecolor="#333",
           capsize=8, linewidth=0.8, alpha=0.85)

    for n, m, s in zip(names, means, stds):
        ax.text(n, m + s + 0.5, f"{m:.1f}±{s:.1f}",
                ha="center", fontsize=9, color="#ffffff", fontfamily="monospace")

    ax.set_title("Generalization Test — Unseen Initial States (30 episodes)",
                 color="#64dcff", fontsize=12, fontfamily="monospace")
    ax.set_ylabel("Final Cumulative Reward", color="#aaa")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "generalization_test.png")
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[eval] Saved {path}")

    # Print summary
    print("\n[generalization] Results on unseen states:")
    for n, v in summary.items():
        print(f"  {n:10s}: {v['mean']:.3f} ± {v['std']:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("Post-Training Evaluation")
    print("=" * 60)

    dqn_model, _ = load_best_dqn()
    ppo_model, _ = load_best_ppo()
    reinforce_net, _ = load_best_reinforce()

    models_dict = {}

    if dqn_model is not None:
        models_dict["DQN"] = (lambda s: run_sb3_episode(dqn_model, s), "#5096ff")
    if reinforce_net is not None:
        models_dict["REINFORCE"] = (lambda s: run_reinforce_episode(reinforce_net, s), "#c880ff")
    if ppo_model is not None:
        models_dict["PPO"] = (lambda s: run_sb3_episode(ppo_model, s), "#50c8a0")

    models_dict["Random"] = (run_random_episode, "#888888")

    if len(models_dict) > 1:
        print("\n[1] Cumulative reward plots...")
        plot_cumulative_rewards(models_dict)

        print("\n[2] Convergence comparison...")
        plot_convergence_comparison()

        print("\n[3] Generalization test...")
        generalization_test(models_dict)
    else:
        print("[warn] No trained models found. Train first.")


if __name__ == "__main__":
    run_evaluation()
