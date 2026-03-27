"""
rendering.py — Static visualization of the Code Mentorship RL Environment.

Demonstrates the environment with a RANDOM agent (no model, no training).
Run:  python environment/rendering.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import pygame

from environment.custom_env import CodeMentorshipEnv


# ──────────────────────────────────────────────────────────────────────────────
# 1. Static diagram: environment components
# ──────────────────────────────────────────────────────────────────────────────

def draw_environment_diagram(save_path: str = "plots/environment_diagram.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    def box(x, y, w, h, color, label, sublabels=None, text_color="white"):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            linewidth=1.5,
            edgecolor=color,
            facecolor=(*matplotlib.colors.to_rgb(color), 0.15),
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h - 0.3, label,
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=color, fontfamily="monospace",
        )
        if sublabels:
            for i, sl in enumerate(sublabels):
                ax.text(
                    x + w / 2, y + h - 0.65 - i * 0.32, sl,
                    ha="center", va="top", fontsize=8.5,
                    color=text_color, fontfamily="monospace",
                )

    def arrow(x1, y1, x2, y2, label="", color="#aaaaaa"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        if label:
            ax.text(mx, my + 0.12, label, ha="center", va="bottom",
                    fontsize=8, color="#cccccc", fontfamily="monospace")

    # Title
    ax.text(7, 8.6, "Code Mentorship RL Environment — Component Diagram",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#64dcff", fontfamily="monospace")

    # Agent box
    box(0.4, 5.5, 3.0, 2.5, "#50c8a0",
        "RL AGENT (Mentor)",
        ["Policy: π(a|s)", "Value Fn: V(s)", "Learns mentoring", "strategy over time"])

    # Environment box
    box(5.3, 5.5, 3.5, 2.5, "#5096ff",
        "ENVIRONMENT",
        ["Student simulator", "Stochastic dynamics", "Task progression", "Session management"])

    # Student State
    box(10.0, 6.2, 3.5, 1.8, "#ffaa40",
        "STUDENT STATE",
        ["skill_level, task_diff", "frustration, engagement", "progress, mastery"])

    # Action Space
    box(0.4, 2.0, 3.0, 3.0, "#c880ff",
        "ACTION SPACE (7)",
        ["0: Provide Hint", "1: Direct Solution", "2: Simpler Subtask",
         "3: Encouragement", "4: Socratic Q", "5: Share Resource", "6: Observe"])

    # Observation space
    box(5.3, 2.0, 3.5, 3.0, "#ff6080",
        "OBSERVATION (11-dim)",
        ["skill_level, task_diff", "frustration, engagement", "progress_ratio",
         "hint_budget, time_left", "consec_failures", "portfolio_score, mastery",
         "help_request_flag"])

    # Reward
    box(10.0, 2.0, 3.5, 3.0, "#40e0d0",
        "REWARD STRUCTURE",
        ["Task complete: +5.0", "Session success: +10-15", "Good hint: +1.5",
         "Socratic (skilled): +2.0", "Direct solution: -2.0", "Student quits: -10.0",
         "Step cost: -0.05"])

    # Arrows
    arrow(3.4, 6.8, 5.3, 6.8, "action a_t", "#50c8a0")
    arrow(8.8, 6.8, 10.0, 6.8, "state s_t", "#5096ff")
    arrow(8.8, 5.8, 5.3, 3.8, "obs o_t", "#ff6080")
    arrow(5.3, 5.7, 3.4, 4.5, "reward r_t", "#ffaa40")

    # Terminal conditions note
    ax.text(5.5, 1.4,
            "Terminal: all tasks done ✓ | frustration≥1 & engagement≤0.05 ✗ | max_steps reached ⏱",
            ha="center", va="top", fontsize=9, color="#aaaaaa", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[diagram] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Static file: random agent run (no model)
# ──────────────────────────────────────────────────────────────────────────────

def run_random_agent_static(
    n_steps: int = 150,
    save_path: str = "plots/random_agent_run.png",
    seed: int = 42,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = CodeMentorshipEnv(render_mode=None, max_steps=n_steps)
    obs, info = env.reset(seed=seed)

    history = {
        "skill": [], "frustration": [], "engagement": [],
        "portfolio": [], "mastery": [], "rewards": [],
        "actions": [], "tasks": [],
    }

    total_reward = 0.0
    for step in range(n_steps):
        action = env.action_space.sample()  # RANDOM — no model
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        history["skill"].append(obs[CodeMentorshipEnv.OBS_SKILL])
        history["frustration"].append(obs[CodeMentorshipEnv.OBS_FRUSTRATION])
        history["engagement"].append(obs[CodeMentorshipEnv.OBS_ENGAGEMENT])
        history["portfolio"].append(obs[CodeMentorshipEnv.OBS_PORTFOLIO])
        history["mastery"].append(obs[CodeMentorshipEnv.OBS_MASTERY])
        history["rewards"].append(reward)
        history["actions"].append(action)
        history["tasks"].append(info["tasks_completed"])

        if terminated or truncated:
            break

    steps = list(range(len(history["skill"])))

    # Plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {
        "skill": "#50c8a0", "engagement": "#5096ff",
        "frustration": "#ff6080", "portfolio": "#40e0d0",
        "mastery": "#c880ff", "reward": "#ffaa40",
    }

    def sub(pos, title, y_data, color, ylabel="Value"):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#161b27")
        ax.plot(steps, y_data, color=color, linewidth=1.6, alpha=0.9)
        ax.fill_between(steps, y_data, alpha=0.15, color=color)
        ax.set_title(title, color=color, fontsize=11, fontfamily="monospace", pad=6)
        ax.set_ylabel(ylabel, color="#888", fontsize=9)
        ax.set_xlabel("Step", color="#888", fontsize=9)
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        return ax

    sub(gs[0, 0], "Student Skill Level", history["skill"], colors["skill"])
    sub(gs[0, 1], "Student Engagement", history["engagement"], colors["engagement"])
    sub(gs[0, 2], "Frustration Level", history["frustration"], colors["frustration"])
    sub(gs[1, 0], "Portfolio Score", history["portfolio"], colors["portfolio"])
    sub(gs[1, 1], "Concept Mastery", history["mastery"], colors["mastery"])

    # Cumulative reward
    cum_reward = np.cumsum(history["rewards"])
    ax_cr = fig.add_subplot(gs[1, 2])
    ax_cr.set_facecolor("#161b27")
    ax_cr.plot(steps, cum_reward, color=colors["reward"], linewidth=1.8)
    ax_cr.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax_cr.set_title("Cumulative Reward", color=colors["reward"], fontsize=11,
                    fontfamily="monospace", pad=6)
    ax_cr.set_ylabel("Reward", color="#888", fontsize=9)
    ax_cr.set_xlabel("Step", color="#888", fontsize=9)
    ax_cr.tick_params(colors="#666")
    for sp in ax_cr.spines.values():
        sp.set_edgecolor("#333")

    # Action distribution
    ax_act = fig.add_subplot(gs[2, :2])
    ax_act.set_facecolor("#161b27")
    act_counts = [history["actions"].count(i) for i in range(7)]
    act_colors = ["#50c8a0", "#ff6080", "#ffaa40", "#5096ff",
                  "#c880ff", "#40e0d0", "#888888"]
    bars = ax_act.bar(CodeMentorshipEnv.ACTION_NAMES, act_counts,
                      color=act_colors, edgecolor="#333", linewidth=0.8)
    ax_act.set_title("Action Distribution (Random Agent)", color="white",
                     fontsize=11, fontfamily="monospace", pad=6)
    ax_act.set_ylabel("Count", color="#888", fontsize=9)
    ax_act.tick_params(colors="#888", labelrotation=20, labelsize=9)
    for sp in ax_act.spines.values():
        sp.set_edgecolor("#333")

    # Tasks completed over time
    ax_task = fig.add_subplot(gs[2, 2])
    ax_task.set_facecolor("#161b27")
    ax_task.step(steps, history["tasks"], color="#50c8a0", linewidth=1.8, where="post")
    ax_task.set_title("Tasks Completed", color="#50c8a0", fontsize=11,
                      fontfamily="monospace", pad=6)
    ax_task.set_ylabel("Tasks", color="#888", fontsize=9)
    ax_task.set_xlabel("Step", color="#888", fontsize=9)
    ax_task.tick_params(colors="#666")
    for sp in ax_task.spines.values():
        sp.set_edgecolor("#333")

    fig.suptitle(
        f"Random Agent — Code Mentorship Environment  |  "
        f"Steps: {len(steps)}  |  Total Reward: {total_reward:.2f}  |  "
        f"Tasks: {history['tasks'][-1]}/10",
        fontsize=13, color="#64dcff", fontfamily="monospace", y=0.98,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"[random_agent] Saved → {save_path}  |  Total reward: {total_reward:.2f}")
    env.close()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Interactive pygame demo (random agent live)
# ──────────────────────────────────────────────────────────────────────────────

def run_pygame_demo(n_steps: int = 300, fps: int = 8, seed: int = 0):
    """Live pygame window — random agent, no training."""
    import pygame

    env = CodeMentorshipEnv(render_mode="human", max_steps=n_steps)
    obs, _ = env.reset(seed=seed)

    pygame.init()
    running = True
    step = 0

    print("[pygame demo] Running random agent. Close window or press Q to exit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1

        if terminated or truncated or step >= n_steps:
            print(f"[pygame demo] Episode ended at step {step}.")
            print(f"  Portfolio score: {info['portfolio_score']:.3f}")
            print(f"  Tasks completed: {info['tasks_completed']}")
            time.sleep(2)
            break

    env.close()
    pygame.quit()


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Generating environment diagram ===")
    draw_environment_diagram()

    print("=== Running random agent (static plots) ===")
    run_random_agent_static()

    print("=== Launching pygame demo (close window to exit) ===")
    try:
        run_pygame_demo()
    except Exception as e:
        print(f"[pygame demo] Skipped: {e}")
