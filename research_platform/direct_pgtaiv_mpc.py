"""Progress-guarded direct-policy MPC for factorial ablation.

This controller retains the direct one-policy-per-face basis used by the
principal direct comparator and changes only the incumbent-replacement
margin.  It therefore supplies the missing direct-basis/progress-guard cell
in the policy-basis by replacement-rule factorial comparison.
"""
from collections import deque

import numpy as np

from research_platform.intention_fidelity import BudgetIntentionMPC
from research_platform.tabletop_push import planar_state
from research_platform.terminal_capture import CaptureController


class DirectPGTAIVMPC(BudgetIntentionMPC):
    STRICT_RATIO = .65
    RELAXED_RATIO = .95
    WINDOW = 11
    MIN_RELATIVE_PROGRESS = .025

    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, mode="direct_verify", kv=kv)
        self.switch_ratio = self.STRICT_RATIO
        self.active = False
        self.update_geometry = True
        self.strategy = "passive"
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
                   self.MIN_RELATIVE_PROGRESS *
                   max(self.progress_history[-1], 1e-5))
        self.progress_relaxed = bool(stalled)
        self.switch_ratio = (self.RELAXED_RATIO if stalled
                             else self.STRICT_RATIO)

    def action(self, state, goal):
        self.update_margin(state, goal)
        return super().action(state, goal)

    def replan(self, state, goal):
        super().replan(state, goal)
        self.plan_records[-1]["basis"] = "direct"
        self.plan_records[-1]["support_faces"] = len(self.geometry.vertices)
        self.plan_records[-1]["switch_ratio"] = self.switch_ratio
        self.plan_records[-1]["progress_relaxed"] = self.progress_relaxed


class DirectPGTAIVCapture(CaptureController):
    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, mode="fixed_simple")
        self.core = DirectPGTAIVMPC(vertices, sampling_seed, kv)
