# oreste_abizera_rl_summative

**Intelligent Code Mentorship RL Agent**
*Mission-Based Reinforcement Learning Summative — Oreste Abizera | ALU*

> An RL agent learns to be an AI code mentor: deciding when to give hints,
> ask Socratic questions, simplify tasks, or offer encouragement — all to
> maximise a Rwandan high school student's portfolio score within a session.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Quick Reference](#3-environment-quick-reference)
4. [Prerequisites](#4-prerequisites)
5. [Installation](#5-installation)
6. [How to Run (Step by Step)](#6-how-to-run-step-by-step)
   - [A — Visualise the environment (random agent, no training)](#a--visualise-the-environment-random-agent-no-training)
   - [B — Train DQN (10 hyperparameter runs)](#b--train-dqn-10-hyperparameter-runs)
   - [C — Train Policy Gradient methods (REINFORCE + PPO)](#c--train-policy-gradient-methods-reinforce--ppo)
   - [D — Evaluate and generate all report plots](#d--evaluate-and-generate-all-report-plots)
   - [E — Run the best agent (main entry point)](#e--run-the-best-agent-main-entry-point)
   - [F — Train everything in one command](#f--train-everything-in-one-command)
7. [Output Files](#7-output-files)
8. [JSON API / Frontend Integration](#8-json-api--frontend-integration)
9. [Troubleshooting](#9-troubleshooting)
10. [Algorithms Implemented](#10-algorithms-implemented)
11. [Author](#11-author)

---

## 1. Project Overview

This project implements a custom Gymnasium environment that simulates an AI
code-mentoring session. A reinforcement learning **mentor agent** interacts
with a stochastic **student model** whose internal state (skill, frustration,
engagement, mastery) evolves in response to the mentor's actions.

**Goal of the agent:** maximise the student's portfolio score and concept
mastery by the end of a session, while keeping frustration low.

Four RL algorithms are compared:

| Algorithm | Category | Library |
|-----------|----------|---------|
| DQN | Value-Based | Stable-Baselines3 |
| REINFORCE | Policy Gradient | PyTorch (custom) |
| PPO | Policy Gradient | Stable-Baselines3 |

Each algorithm is evaluated over **10 hyperparameter combinations**.

---

## 2. Repository Structure

```
oreste_abizera_rl_summative/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py        # Gymnasium environment
│   └── rendering.py         # Pygame GUI + static plots
├── training/
│   ├── __init__.py
│   ├── dqn_training.py      # DQN: 10 runs, saves models + CSVs + plots
│   └── pg_training.py       # REINFORCE / PPO: 10 runs each
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py          # Post-training evaluation + generalization
├── models/
│   ├── dqn/                 # Saved DQN .zip models (after training)
│   └── pg/
│       ├── ppo/             # Saved PPO .zip models
│       └── reinforce/       # Saved REINFORCE .pt models
├── plots/                   # All generated plots + CSV tables
├── main.py                  # Run best-performing agent (entry point)
├── requirements.txt
└── README.md
```

---

## 3. Environment Quick Reference

| Item | Detail |
|------|--------|
| Observation space | `Box(11,)` — all features in [0, 1] |
| Action space | `Discrete(7)` |
| Actions | Provide Hint · Direct Solution · Simpler Subtask · Encouragement · Socratic Question · Share Resource · Observe |
| Max steps | 200 (configurable) |
| Terminal conditions | All 10 tasks done · Student quits (frustration=1 & engagement=0) · Max steps |
| Key reward signals | +5 task complete · +10–15 session bonus · +2 Socratic hit · −2 direct solution · −10 student quits |

---

## 4. Prerequisites

| Requirement | Minimum version |
|-------------|-----------------|
| Python | 3.10 |
| pip | 23+ |
| Operating System | Linux / macOS / Windows (WSL recommended on Windows) |
| RAM | 4 GB (8 GB recommended for all 30 training runs) |
| GPU | Optional — CPU is sufficient; all device flags set to `"cpu"` |

> **Windows users:** Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
> and run inside Ubuntu for the smoothest experience, especially for pygame.

---

## 5. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/oreste-abizera/oreste_abizera_rl_summative.git
cd oreste_abizera_rl_summative
```

### Step 2 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3 — Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

> This installs: `gymnasium`, `stable-baselines3`, `torch`, `numpy`,
> `pandas`, `matplotlib`, `pygame`, and `tensorboard`.

### Step 5 — Verify installation

```bash
python -c "import gymnasium, stable_baselines3, torch, pygame; print('All OK')"
```

Expected output: `All OK`

---

## 6. How to Run (Step by Step)

> **Important:** All commands must be run from the **project root directory**
> (`oreste_abizera_rl_summative/`), not from a subdirectory.

---

### A — Visualise the environment (random agent, no training)

This step creates the **static visualisation file** (required by the rubric)
and optionally opens a live pygame window.

```bash
python environment/rendering.py
```

**What this does:**
- Generates `plots/environment_diagram.png` — component diagram of the environment
- Generates `plots/random_agent_run.png` — 6-panel plot of a random agent's episode
- Opens a live pygame window showing the random agent in action (close with Q or ✕)

**Headless / no display (e.g. SSH server):**
```bash
# The pygame demo will fail gracefully — static files are still saved
python environment/rendering.py 2>/dev/null
```

---

### B — Train DQN (10 hyperparameter runs)

```bash
python training/dqn_training.py
```

**What this does:**
- Runs 10 different DQN configurations (~80,000 steps each)
- Saves trained models to `models/dqn/dqn_run1.zip … dqn_run10.zip`
- Saves `models/dqn/best_run.json` — metadata for the best run
- Saves `plots/dqn_results.csv` — hyperparameter table
- Saves `plots/dqn_reward_curves.png` — all 10 training curves
- Saves `plots/dqn_final_rewards.png` — bar chart comparison
- Saves `plots/dqn_best_objective.png` — objective curve for best run

**Estimated time:** 5–25 minutes depending on hardware.

**Watch live progress:**
```bash
python training/dqn_training.py 2>&1 | tee logs/dqn_log.txt
```

---

### C — Train Policy Gradient methods (REINFORCE + PPO)

```bash
python training/pg_training.py
```

**What this does:**
- Trains REINFORCE (10 episodes-based runs, PyTorch custom implementation)
- Trains PPO (10 runs, Stable-Baselines3)
- Saves all models under `models/pg/{ppo,reinforce}/`
- Saves CSVs: `plots/reinforce_results.csv`, `plots/ppo_results.csv`
- Saves reward curve plots per algorithm
- Saves entropy curve plots per algorithm
- Saves `plots/pg_comparison.png` — side-by-side best-run comparison

**Estimated time:** 15–60 minutes depending on hardware.

**Tip — Run DQN and PG in parallel (two terminals):**
```bash
# Terminal 1
python training/dqn_training.py

# Terminal 2
python training/pg_training.py
```

---

### D — Evaluate and generate all report plots

> Run this **after** training (steps B and C).

```bash
python evaluation/evaluate.py
```

**What this does:**
- Loads best model from each algorithm
- Generates `plots/cumulative_rewards.png` — cumulative reward per algorithm (subplots)
- Generates `plots/convergence_comparison.png` — episodes-to-convergence scatter
- Generates `plots/generalization_test.png` — performance on 30 unseen seeds

---

### E — Run the best agent (main entry point)

```bash
# Auto-select best available trained model, 3 episodes, with pygame window
python main.py

# Specific algorithm
python main.py --algo ppo
python main.py --algo dqn
python main.py --algo reinforce
python main.py --algo random

# More episodes
python main.py --algo ppo --episodes 5

# Different seed
python main.py --algo ppo --seed 42

# Headless (no pygame window) — useful for servers
python main.py --no-render

# Export results as JSON (for frontend/API integration)
python main.py --algo ppo --no-render --export-json
```

The `--export-json` flag writes `episode_results.json` which can be consumed
by any frontend or REST API — see [Section 8](#8-json-api--frontend-integration).

**Terminal verbose output example:**
```
═══════════════════════════════════════════════════════
  Algorithm      : PPO
  Seed           : 0
  Steps          : 143
  Total Reward   : 47.821
  Tasks Completed: 7 / 10
  Portfolio Score: 0.612
  Concept Mastery: 0.734
  Frustration    : 0.231
  Engagement     : 0.805
  Top Actions    : Socratic Question(38), Encouragement(27), Provide Hint(19)
═══════════════════════════════════════════════════════
```

---

### F — Train everything in one command

```bash
python training/dqn_training.py && \
python training/pg_training.py && \
python evaluation/evaluate.py
```

Or on Windows:
```cmd
python training\dqn_training.py && python training\pg_training.py && python evaluation\evaluate.py
```

---

## 7. Output Files

After a full training + evaluation run, the `plots/` directory contains:

| File | Description |
|------|-------------|
| `environment_diagram.png` | Environment component diagram (for report) |
| `random_agent_run.png` | Static random-agent visualisation |
| `dqn_reward_curves.png` | DQN — all 10 training curves |
| `dqn_final_rewards.png` | DQN — bar chart of final rewards |
| `dqn_best_objective.png` | DQN — best run objective (convergence) curve |
| `dqn_results.csv` | DQN hyperparameter table (10 rows) |
| `reinforce_reward_curves.png` | REINFORCE — all 10 curves |
| `reinforce_entropy.png` | REINFORCE — entropy curve |
| `reinforce_results.csv` | REINFORCE hyperparameter table |
| `ppo_reward_curves.png` | PPO — all 10 curves |
| `ppo_entropy.png` | PPO — entropy curve |
| `ppo_results.csv` | PPO hyperparameter table |
| `pg_comparison.png` | REINFORCE vs PPO — side by side |
| `cumulative_rewards.png` | All methods cumulative rewards (subplots) |
| `convergence_comparison.png` | Episodes-to-convergence scatter plot |
| `generalization_test.png` | Generalisation on 30 unseen seeds |

---

## 8. JSON API / Frontend Integration

The `--export-json` flag serialises a full episode summary:

```bash
python main.py --algo ppo --no-render --episodes 1 --export-json
```

Output (`episode_results.json`):
```json
{
  "algorithm": "PPO",
  "episodes": [
    {
      "algorithm": "PPO",
      "seed": 0,
      "steps": 143,
      "total_reward": 47.821,
      "tasks_completed": 7,
      "portfolio_score": 0.612,
      "concept_mastery": 0.734,
      "frustration": 0.231,
      "engagement": 0.805,
      "action_distribution": { ... }
    }
  ],
  "summary": {
    "mean_reward": 47.821,
    "std_reward": 0.0,
    "mean_tasks_completed": 7.0
  }
}
```

This JSON can be served directly as a REST API response to a frontend
(e.g. the Growthwave Academy platform) to display real-time mentoring
agent decisions in a web interface.

---

## 9. Troubleshooting

### `pygame.error: No video mode has been set` or display errors

You are running on a headless server. Use:
```bash
python main.py --no-render
python environment/rendering.py  # will skip pygame demo but save static files
```

Or set a virtual display:
```bash
sudo apt-get install xvfb
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
python main.py
```

### `ModuleNotFoundError: No module named 'stable_baselines3'`

Re-run the installation inside the virtual environment:
```bash
source venv/bin/activate   # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### `FileNotFoundError: models/dqn/best_run.json`

You are running `main.py` or `evaluate.py` before training. Train first:
```bash
python training/dqn_training.py
python training/pg_training.py
```

### Training is too slow

Each run uses `TRAIN_STEPS = 80_000`. To do a quick smoke test, edit
`training/dqn_training.py` and `training/pg_training.py` and temporarily
change:
```python
TRAIN_STEPS = 10_000   # quick test
```

### `AssertionError: Episode has ended. Call reset()`

This is expected behaviour — the environment signals termination. The
training loops handle this correctly; if you see it in a custom script,
call `env.reset()` before the next episode.

---

## 10. Algorithms Implemented

### DQN (Deep Q-Network)
- MLP policy network with configurable architecture
- Experience replay buffer
- Target network with configurable update interval
- ε-greedy exploration with linear decay
- Soft/hard target updates (τ parameter)

### REINFORCE
- Custom PyTorch implementation (no SB3 dependency)
- Categorical policy network
- Monte Carlo returns with optional running baseline
- Entropy regularisation coefficient
- Gradient clipping

### PPO (Proximal Policy Optimization)
- Clipped surrogate objective
- Generalised Advantage Estimation (GAE)
- Configurable clip range, entropy coefficient, value function coefficient
- Multiple update epochs per rollout

---

## 11. Author

| | |
|---|---|
| **Name** | Oreste Abizera |
| **Email** | o.abizera@alustudent.com |
| **Program** | BS(Hons) Software Engineering (ML), ALU |
| **Mission** | AI-driven mentorship for Rwandan high schools |
| **Platform** | [Growthwave Academy](https://academy.growthwave.rw) |

---

*"Technology is only as good as its usability."*
