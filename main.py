"""
main.py — Entry point for the Code Mentorship RL project.

Loads and runs the best-performing trained model (PPO by default,
with fallback chain: PPO → DQN → A2C → REINFORCE → Random).

Usage:
    python main.py                        # auto-select best model
    python main.py --algo ppo             # select specific algorithm
    python main.py --algo dqn --no-render # headless run
    python main.py --episodes 5 --seed 7  # multiple episodes

Flags:
    --algo     [ppo|dqn|a2c|reinforce|random]
    --episodes Number of episodes to run (default: 3)
    --seed     Random seed (default: 0)
    --no-render  Disable pygame window
    --export-json  Serialise episode summary as JSON (for API/frontend use)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from environment.custom_env import CodeMentorshipEnv
from training.pg_training import PolicyNet


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_ppo():
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    info_path = "models/pg/ppo/best_run.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(info_path)
    with open(info_path) as f:
        info = json.load(f)
    env = Monitor(CodeMentorshipEnv())
    model = PPO.load(f"models/pg/ppo/ppo_run{info['run']}", env=env)
    return model, "PPO", info


def _load_dqn():
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    info_path = "models/dqn/best_run.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(info_path)
    with open(info_path) as f:
        info = json.load(f)
    env = Monitor(CodeMentorshipEnv())
    model = DQN.load(f"models/dqn/dqn_run{info['run']}", env=env)
    return model, "DQN", info


def _load_a2c():
    from stable_baselines3 import A2C
    from stable_baselines3.common.monitor import Monitor

    info_path = "models/pg/a2c/best_run.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(info_path)
    with open(info_path) as f:
        info = json.load(f)
    env = Monitor(CodeMentorshipEnv())
    model = A2C.load(f"models/pg/a2c/a2c_run{info['run']}", env=env)
    return model, "A2C", info


def _load_reinforce():
    info_path = "models/pg/reinforce/best_run.json"
    if not os.path.exists(info_path):
        raise FileNotFoundError(info_path)
    with open(info_path) as f:
        info = json.load(f)
    hidden = eval(info.get("hidden_layers", "[64, 64]"))
    obs_dim = CodeMentorshipEnv().observation_space.shape[0]
    act_dim = CodeMentorshipEnv().action_space.n
    net = PolicyNet(obs_dim, act_dim, hidden)
    net.load_state_dict(
        torch.load(f"models/pg/reinforce/reinforce_run{info['run']}.pt",
                   map_location="cpu")
    )
    net.eval()
    return net, "REINFORCE", info


LOADER_MAP = {
    "ppo":       _load_ppo,
    "dqn":       _load_dqn,
    "a2c":       _load_a2c,
    "reinforce": _load_reinforce,
}

FALLBACK_ORDER = ["ppo", "dqn", "a2c", "reinforce"]


def load_model(algo: str):
    if algo == "random":
        return None, "Random", {}

    if algo != "auto":
        return LOADER_MAP[algo]()

    # Auto: try fallback order
    for name in FALLBACK_ORDER:
        try:
            return LOADER_MAP[name]()
        except Exception:
            continue
    print("[main] No trained models found — falling back to random agent.")
    return None, "Random", {}


# ──────────────────────────────────────────────────────────────────────────────
# Run episode
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(model, algo_name: str, seed: int, render: bool) -> dict:
    render_mode = "human" if render else None
    env = CodeMentorshipEnv(render_mode=render_mode)
    obs, _ = env.reset(seed=seed)

    step = 0
    total_reward = 0.0
    done = False
    action_log = []
    reward_log = []

    while not done:
        # Choose action
        if model is None or algo_name == "Random":
            action = env.action_space.sample()
        elif algo_name == "REINFORCE":
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = model(obs_t)
            action = probs.argmax(dim=-1).item()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_log.append(CodeMentorshipEnv.ACTION_NAMES[action])
        reward_log.append(reward)
        done = terminated or truncated
        step += 1

        if render:
            env.render()
            time.sleep(0.08)

    env.close()

    # Verbose terminal output
    print("\n" + "─" * 55)
    print(f"  Algorithm      : {algo_name}")
    print(f"  Seed           : {seed}")
    print(f"  Steps          : {step}")
    print(f"  Total Reward   : {total_reward:.3f}")
    print(f"  Tasks Completed: {info['tasks_completed']} / 10")
    print(f"  Portfolio Score: {info['portfolio_score']:.3f}")
    print(f"  Concept Mastery: {info['concept_mastery']:.3f}")
    print(f"  Frustration    : {info['frustration']:.3f}")
    print(f"  Engagement     : {info['engagement']:.3f}")
    print(f"  Top Actions    : {_top_actions(action_log)}")
    print("─" * 55)

    return {
        "algorithm": algo_name,
        "seed": seed,
        "steps": step,
        "total_reward": round(total_reward, 3),
        "tasks_completed": info["tasks_completed"],
        "portfolio_score": round(info["portfolio_score"], 3),
        "concept_mastery": round(info["concept_mastery"], 3),
        "frustration": round(info["frustration"], 3),
        "engagement": round(info["engagement"], 3),
        "reward_per_step": round(total_reward / max(step, 1), 4),
        "action_distribution": {
            name: action_log.count(name) for name in CodeMentorshipEnv.ACTION_NAMES
        },
    }


def _top_actions(log, n=3):
    counts = {a: log.count(a) for a in set(log)}
    top = sorted(counts.items(), key=lambda x: -x[1])[:n]
    return ", ".join(f"{k}({v})" for k, v in top)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the best RL agent on the Code Mentorship environment."
    )
    parser.add_argument("--algo", default="auto",
                        choices=["auto", "ppo", "dqn", "a2c", "reinforce", "random"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true",
                        help="Disable pygame window (headless mode)")
    parser.add_argument("--export-json", action="store_true",
                        help="Export episode results as JSON")
    args = parser.parse_args()

    print("=" * 55)
    print("  Code Mentorship RL — Agent Demo")
    print("  Mission: AI Mentorship for Rwandan Schools")
    print("  Author : Oreste Abizera | ALU")
    print("=" * 55)

    # Load model
    print(f"\nLoading model: {args.algo} ...")
    model, algo_name, info = load_model(args.algo)
    print(f"  Loaded: {algo_name}")
    if info:
        print(f"  Trained mean reward: {info.get('mean_reward', 'N/A')}")

    # Run episodes
    episode_results = []
    render = not args.no_render
    for ep in range(args.episodes):
        print(f"\n[Episode {ep+1}/{args.episodes}]")
        result = run_episode(model, algo_name, seed=args.seed + ep, render=render)
        episode_results.append(result)

    # Summary
    rewards = [r["total_reward"] for r in episode_results]
    tasks = [r["tasks_completed"] for r in episode_results]
    print(f"\n{'='*55}")
    print(f"  SUMMARY over {args.episodes} episode(s):")
    print(f"  Mean reward : {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean tasks  : {np.mean(tasks):.1f} / 10")
    print(f"{'='*55}\n")

    # JSON export (for API/frontend serialization)
    if args.export_json:
        payload = {
            "algorithm": algo_name,
            "episodes": episode_results,
            "summary": {
                "mean_reward": round(float(np.mean(rewards)), 3),
                "std_reward": round(float(np.std(rewards)), 3),
                "mean_tasks_completed": round(float(np.mean(tasks)), 2),
            },
        }
        out_path = "episode_results.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[export] Results saved → {out_path}")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
