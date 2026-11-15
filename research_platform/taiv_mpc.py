"""Topology-Adaptive Intention-Verified MPC (TAIV-MPC).

The controller adapts the *basis* of candidate feedback policies to the
calibrated support topology.  Low-order polygons use task-space rotate/translate
intentions; higher-order polygons use two tangential contact offsets per face.
Both bases have the same 3N candidate count (including direct policies), so the
choice changes representation rather than the rollout budget.

Only measured state, the task goal, and the estimated/calibrated polygon are
used.  The split is fixed before the independent validation cohort.
"""
from research_platform.intention_fidelity import BudgetIntentionMPC
from research_platform.terminal_capture import CaptureController


class TAIVMPC(BudgetIntentionMPC):
    """Support-calibrated MPC with a topology-conditioned policy basis."""

    def __init__(self, vertices, sampling_seed=0, kv=40., simple_faces=4,
                 switch_ratio=.95):
        self.simple_faces = int(simple_faces)
        basis = "library" if len(vertices) <= self.simple_faces else "grid"
        super().__init__(vertices, sampling_seed, mode=f"{basis}_verify", kv=kv)
        self.basis = basis
        self.switch_ratio = float(switch_ratio)
        self.active = False
        self.update_geometry = True
        self.strategy = "passive"

    def replan(self, state, goal):
        super().replan(state, goal)
        self.plan_records[-1]["basis"] = self.basis
        self.plan_records[-1]["support_faces"] = len(self.geometry.vertices)


class TAIVCapture(CaptureController):
    def __init__(self, vertices, sampling_seed=0, kv=40., simple_faces=4,
                 switch_ratio=.95):
        super().__init__(vertices, sampling_seed, mode="fixed_simple")
        self.core = TAIVMPC(vertices, sampling_seed, kv, simple_faces, switch_ratio)
