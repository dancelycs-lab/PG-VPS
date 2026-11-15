"""Progress-Guarded Topology-Adaptive Intention-Verified MPC (PG-TAIV).

The incumbent replacement margin is strict during measured progress and relaxed
only after closed-loop stagnation.  Thus nominal-model predictions can propose a
switch, fine simulation verifies it, and measured task progress determines how
much predicted improvement is required.  The 0.5 s window and 2.5% relative
progress test match the pre-existing event-trigger diagnostic in this project.
"""
from collections import deque

import numpy as np

from research_platform.taiv_mpc import TAIVMPC
from research_platform.tabletop_push import planar_state
from research_platform.terminal_capture import CaptureController


class PGTAIVMPC(TAIVMPC):
    SIMPLE_FACES = 4
    STRICT_RATIO = .65
    RELAXED_RATIO = .95
    WINDOW = 11
    MIN_RELATIVE_PROGRESS = .025

    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, kv,
                         simple_faces=self.SIMPLE_FACES,
                         switch_ratio=self.STRICT_RATIO)
        self.progress_history = deque(maxlen=self.WINDOW)
        self.progress_relaxed = False

    @staticmethod
    def task_energy(state, goal):
        q, _ = planar_state(state)
        error = np.asarray(goal) - np.r_[.15 + q[0], q[1], q[2]]
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        return float(200. * (error[:2] @ error[:2]) + 2. * error[2] ** 2)

    def update_margin(self, state, goal):
        if self.ctrl.lock is not None and self.ctrl.phase == "push":
            self.progress_history.append(self.task_energy(state, goal))
        else:
            self.progress_history.clear()
        stalled = (len(self.progress_history) == self.WINDOW and
                   self.progress_history[0] - self.progress_history[-1] <
                   self.MIN_RELATIVE_PROGRESS * max(self.progress_history[-1], 1e-5))
        self.progress_relaxed = bool(stalled)
        self.switch_ratio = self.RELAXED_RATIO if stalled else self.STRICT_RATIO

    def action(self, state, goal):
        self.update_margin(state, goal)
        return super().action(state, goal)

    def replan(self, state, goal):
        super().replan(state, goal)
        self.plan_records[-1]["switch_ratio"] = self.switch_ratio
        self.plan_records[-1]["progress_relaxed"] = self.progress_relaxed


class PGTAIVCapture(CaptureController):
    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, mode="fixed_simple")
        self.core = PGTAIVMPC(vertices, sampling_seed, kv)
