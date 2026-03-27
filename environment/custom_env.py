"""
Custom Gymnasium Environment: Intelligent Code Mentorship Environment
Mission: Automated Code Mentorship and Portfolio Development in Rwandan Schools

The agent acts as an AI mentor that guides a student through a software project.
At each step, the mentor agent chooses a mentoring action (feedback, hint, challenge,
resource, etc.) based on the student's current state (skill level, frustration,
engagement, progress, etc.). The goal is to maximize the student's final portfolio
score within a session budget.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class CodeMentorshipEnv(gym.Env):
    """
    Observation Space (11 dimensions, all normalized to [0, 1]):
      0: student_skill_level        - how competent the student is (0=novice, 1=expert)
      1: task_difficulty            - difficulty of current task (0=easy, 1=hard)
      2: frustration_level          - student frustration (0=calm, 1=max frustration)
      3: engagement_level           - student engagement (0=disengaged, 1=fully engaged)
      4: progress_ratio             - tasks completed / total tasks
      5: hint_budget_remaining      - fraction of hint budget left
      6: session_time_remaining     - fraction of session time left
      7: consecutive_failures       - normalized count of consecutive fails (0-1)
      8: portfolio_score            - current portfolio score (0-1)
      9: concept_mastery            - fraction of concepts understood
      10: help_request_flag         - 1 if student just requested help, else 0

    Action Space (Discrete, 7 actions):
      0: provide_hint               - small scaffold nudge
      1: give_direct_solution       - show full solution (heavy hint penalty)
      2: assign_simpler_subtask     - break task down
      3: give_encouragement         - boost engagement/morale
      4: pose_socratic_question     - challenge thinking
      5: share_resource_link        - share relevant reading/video
      6: do_nothing / observe       - wait and observe

    Terminal Conditions:
      - All tasks completed (success)
      - Frustration maxes out AND engagement drops to 0 (student quits)
      - Session time runs out
      - Max steps reached (default 200)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    # Action meanings for rendering
    ACTION_NAMES = [
        "Provide Hint",
        "Give Direct Solution",
        "Assign Simpler Subtask",
        "Give Encouragement",
        "Pose Socratic Question",
        "Share Resource Link",
        "Observe / Do Nothing",
    ]

    # Observation indices
    OBS_SKILL = 0
    OBS_DIFFICULTY = 1
    OBS_FRUSTRATION = 2
    OBS_ENGAGEMENT = 3
    OBS_PROGRESS = 4
    OBS_HINT_BUDGET = 5
    OBS_TIME = 6
    OBS_CONSEC_FAIL = 7
    OBS_PORTFOLIO = 8
    OBS_MASTERY = 9
    OBS_HELP_REQ = 10

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        total_tasks: int = 10,
        hint_budget: int = 15,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.total_tasks = total_tasks
        self.hint_budget = hint_budget

        # Observation space: 11 continuous features in [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(11, dtype=np.float32),
            high=np.ones(11, dtype=np.float32),
            dtype=np.float32,
        )

        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(7)

        # Rendering
        self.window = None
        self.clock = None
        self.screen_width = 900
        self.screen_height = 600

        # Internal state (not directly observed)
        self._np_random = None
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        rng = self.np_random

        # Randomly initialise student profile
        self.skill_level = float(rng.uniform(0.05, 0.35))          # novice-ish student
        self.task_difficulty = float(rng.uniform(0.3, 0.7))
        self.frustration = float(rng.uniform(0.0, 0.2))
        self.engagement = float(rng.uniform(0.6, 1.0))
        self.tasks_completed = 0
        self.hints_used = 0
        self.steps_taken = 0
        self.consecutive_failures = 0
        self.portfolio_score = 0.0
        self.concept_mastery = float(rng.uniform(0.0, 0.2))
        self.help_requested = False
        self.done = False
        self.episode_reward = 0.0
        self.action_history = []

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action {action}"
        assert not self.done, "Episode has ended. Call reset()."

        self.steps_taken += 1
        reward = 0.0
        terminated = False
        truncated = False

        # ---- Student internal dynamics each step -------------------------
        # Natural drift: engagement slowly decays, frustration slowly rises
        self.engagement = np.clip(self.engagement - 0.01, 0.0, 1.0)
        self.frustration = np.clip(self.frustration + 0.005, 0.0, 1.0)

        # Student randomly requests help with higher probability when frustrated
        p_help = 0.05 + 0.25 * self.frustration
        self.help_requested = bool(self.np_random.random() < p_help)

        # ---- Apply action ------------------------------------------------
        if action == 0:  # provide_hint
            reward += self._apply_hint()

        elif action == 1:  # give_direct_solution
            reward += self._apply_direct_solution()

        elif action == 2:  # assign_simpler_subtask
            reward += self._apply_simpler_subtask()

        elif action == 3:  # give_encouragement
            reward += self._apply_encouragement()

        elif action == 4:  # pose_socratic_question
            reward += self._apply_socratic_question()

        elif action == 5:  # share_resource_link
            reward += self._apply_share_resource()

        elif action == 6:  # observe / do_nothing
            reward += self._apply_observe()

        # ---- Student makes progress attempt (stochastic) -----------------
        attempt_success_prob = (
            0.2
            + 0.5 * self.skill_level
            - 0.3 * self.task_difficulty
            + 0.1 * self.engagement
            - 0.1 * self.frustration
        )
        attempt_success_prob = float(np.clip(attempt_success_prob, 0.05, 0.95))

        if self.np_random.random() < attempt_success_prob:
            # Student succeeds on this task step
            self.consecutive_failures = 0
            progress_gain = 0.08 + 0.05 * self.skill_level
            self.skill_level = min(self.skill_level + 0.03, 1.0)
            self.concept_mastery = min(self.concept_mastery + 0.05, 1.0)
            self.frustration = max(self.frustration - 0.05, 0.0)
            self.engagement = min(self.engagement + 0.03, 1.0)

            # Check task completion
            task_completion_threshold = 1.0 / self.total_tasks
            self._task_progress = getattr(self, "_task_progress", 0.0) + progress_gain
            if self._task_progress >= task_completion_threshold:
                self._task_progress -= task_completion_threshold
                self.tasks_completed += 1
                portfolio_gain = 0.06 + 0.04 * self.skill_level
                self.portfolio_score = min(self.portfolio_score + portfolio_gain, 1.0)
                reward += 5.0  # reward for completing a task

        else:
            # Student fails this attempt
            self.consecutive_failures += 1
            self.frustration = min(self.frustration + 0.04, 1.0)
            self.engagement = max(self.engagement - 0.02, 0.0)
            reward -= 0.5  # small penalty for failure (environment difficulty)

        # ---- Time penalty ------------------------------------------------
        reward -= 0.05  # small step cost

        # ---- Terminal conditions -----------------------------------------
        if self.tasks_completed >= self.total_tasks:
            bonus = 10.0 * self.portfolio_score + 5.0 * self.concept_mastery
            reward += bonus
            terminated = True

        elif self.frustration >= 1.0 and self.engagement <= 0.05:
            # Student quits
            reward -= 10.0
            terminated = True

        elif self.steps_taken >= self.max_steps:
            truncated = True

        self.done = terminated or truncated
        self.episode_reward += reward
        self.action_history.append(action)

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _apply_hint(self) -> float:
        if self.hints_used >= self.hint_budget:
            return -1.0  # over budget
        self.hints_used += 1
        self.frustration = max(self.frustration - 0.08, 0.0)
        self.engagement = min(self.engagement + 0.03, 1.0)
        self.skill_level = min(self.skill_level + 0.01, 1.0)
        # Good when student is frustrated; less good when not needed
        reward = 1.5 - 2.0 * (1.0 - self.frustration)
        reward -= 0.2  # small hint cost
        return float(reward)

    def _apply_direct_solution(self) -> float:
        # Reduces frustration but kills learning
        self.frustration = max(self.frustration - 0.15, 0.0)
        self.engagement = max(self.engagement - 0.1, 0.0)  # becomes passive
        self.skill_level = max(self.skill_level - 0.02, 0.0)  # no real learning
        self.hints_used += 2  # counts as 2 hints
        return -2.0  # heavily penalised: bad pedagogy

    def _apply_simpler_subtask(self) -> float:
        self.task_difficulty = max(self.task_difficulty - 0.15, 0.1)
        self.frustration = max(self.frustration - 0.12, 0.0)
        self.engagement = min(self.engagement + 0.05, 1.0)
        return 1.0

    def _apply_encouragement(self) -> float:
        gain = 0.1 * (1.0 - self.engagement)  # diminishing returns
        self.engagement = min(self.engagement + gain + 0.05, 1.0)
        self.frustration = max(self.frustration - 0.05, 0.0)
        return float(0.5 + gain)

    def _apply_socratic_question(self) -> float:
        # Great for high-skill students, frustrating for low-skill
        if self.skill_level > 0.5:
            self.skill_level = min(self.skill_level + 0.04, 1.0)
            self.concept_mastery = min(self.concept_mastery + 0.06, 1.0)
            self.engagement = min(self.engagement + 0.04, 1.0)
            return 2.0
        else:
            self.frustration = min(self.frustration + 0.06, 1.0)
            return -0.5

    def _apply_share_resource(self) -> float:
        self.concept_mastery = min(self.concept_mastery + 0.04, 1.0)
        self.skill_level = min(self.skill_level + 0.02, 1.0)
        self.engagement = min(self.engagement + 0.02, 1.0)
        return 0.8

    def _apply_observe(self) -> float:
        # Penalty only if student is frustrated and needs help
        if self.frustration > 0.6 and self.help_requested:
            return -1.0
        return 0.1  # neutral otherwise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        hint_budget_remaining = max(0.0, 1.0 - self.hints_used / self.hint_budget)
        time_remaining = 1.0 - self.steps_taken / self.max_steps
        consec_fail_norm = min(self.consecutive_failures / 10.0, 1.0)
        progress_ratio = self.tasks_completed / self.total_tasks

        obs = np.array(
            [
                self.skill_level,
                self.task_difficulty,
                self.frustration,
                self.engagement,
                progress_ratio,
                hint_budget_remaining,
                time_remaining,
                consec_fail_norm,
                self.portfolio_score,
                self.concept_mastery,
                float(self.help_requested),
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "tasks_completed": self.tasks_completed,
            "portfolio_score": self.portfolio_score,
            "concept_mastery": self.concept_mastery,
            "frustration": self.frustration,
            "engagement": self.engagement,
            "hints_used": self.hints_used,
            "episode_reward": self.episode_reward,
            "steps": self.steps_taken,
        }

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_pygame(self):
        import pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Code Mentorship RL Environment")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = self._build_surface()
        self.window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _render_rgb_array(self) -> np.ndarray:
        import pygame

        surface = self._build_surface()
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _build_surface(self):
        import pygame

        pygame.font.init()
        surface = pygame.Surface((self.screen_width, self.screen_height))

        # Background
        surface.fill((15, 20, 35))

        # Fonts
        title_font = pygame.font.SysFont("monospace", 22, bold=True)
        label_font = pygame.font.SysFont("monospace", 16)
        small_font = pygame.font.SysFont("monospace", 13)

        # Title
        title = title_font.render(
            "Code Mentorship RL Environment — Oreste Abizera", True, (100, 220, 255)
        )
        surface.blit(title, (20, 15))

        # --- Left panel: Student state bars ---
        metrics = [
            ("Skill Level", self.skill_level, (80, 200, 120)),
            ("Engagement", self.engagement, (80, 160, 240)),
            ("Frustration", self.frustration, (240, 80, 80)),
            ("Task Difficulty", self.task_difficulty, (220, 180, 80)),
            ("Concept Mastery", self.concept_mastery, (160, 100, 240)),
            ("Portfolio Score", self.portfolio_score, (80, 240, 200)),
        ]

        y_start = 65
        bar_x = 20
        bar_w = 220
        bar_h = 22
        for i, (label, value, color) in enumerate(metrics):
            y = y_start + i * 38
            lbl = label_font.render(label, True, (200, 200, 220))
            surface.blit(lbl, (bar_x, y))
            pygame.draw.rect(surface, (40, 45, 65), (bar_x, y + 17, bar_w, bar_h))
            pygame.draw.rect(
                surface, color, (bar_x, y + 17, int(bar_w * value), bar_h)
            )
            pct = small_font.render(f"{value:.2f}", True, (220, 220, 220))
            surface.blit(pct, (bar_x + bar_w + 8, y + 18))

        # --- Middle panel: Stats ---
        stats = [
            ("Steps", f"{self.steps_taken}/{self.max_steps}"),
            ("Tasks Done", f"{self.tasks_completed}/{self.total_tasks}"),
            ("Hints Used", f"{self.hints_used}/{self.hint_budget}"),
            ("Consec. Fails", f"{self.consecutive_failures}"),
            ("Episode Reward", f"{self.episode_reward:.2f}"),
            ("Help Requested", "YES" if self.help_requested else "no"),
        ]
        stat_x = 310
        for i, (k, v) in enumerate(stats):
            y = 65 + i * 38
            klbl = label_font.render(k + ":", True, (160, 180, 220))
            vlbl = label_font.render(v, True, (240, 240, 100))
            surface.blit(klbl, (stat_x, y))
            surface.blit(vlbl, (stat_x + 160, y))

        # --- Right panel: last action + action history ---
        action_x = 540
        action_title = label_font.render("Last Action:", True, (200, 200, 220))
        surface.blit(action_title, (action_x, 65))
        if self.action_history:
            last = self.ACTION_NAMES[self.action_history[-1]]
        else:
            last = "—"
        last_lbl = title_font.render(last, True, (100, 240, 180))
        surface.blit(last_lbl, (action_x, 85))

        # Action frequency bar chart
        hist_title = label_font.render("Action Frequency:", True, (200, 200, 220))
        surface.blit(hist_title, (action_x, 135))
        counts = [self.action_history.count(a) for a in range(7)]
        max_count = max(counts) if counts and max(counts) > 0 else 1
        for i, (cnt, name) in enumerate(zip(counts, self.ACTION_NAMES)):
            y = 155 + i * 30
            bar_len = int(280 * cnt / max_count)
            color = (100, 180, 255) if i != (self.action_history[-1] if self.action_history else -1) else (100, 240, 180)
            pygame.draw.rect(surface, color, (action_x, y, bar_len, 20))
            nlbl = small_font.render(f"{name[:22]}: {cnt}", True, (200, 200, 220))
            surface.blit(nlbl, (action_x, y + 2))

        # Progress bar at bottom
        prog = self.tasks_completed / self.total_tasks
        pygame.draw.rect(surface, (40, 45, 65), (20, 560, 860, 20))
        pygame.draw.rect(surface, (80, 200, 120), (20, 560, int(860 * prog), 20))
        prog_lbl = small_font.render(
            f"Project Progress: {prog*100:.0f}%", True, (220, 220, 220)
        )
        surface.blit(prog_lbl, (20, 542))

        return surface
