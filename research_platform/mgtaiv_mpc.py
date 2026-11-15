"""Margin-Guarded Topology-Adaptive Intention-Verified MPC.

MG-TAIV combines passive support calibration, a topology-conditioned feedback
policy basis, coarse proposal/fine verification, and a conservative replacement
rule.  After initialization, a challenger replaces the incumbent only if its
fine-rollout cost is below 0.65 times the incumbent cost.  The ratio was selected
once on the dedicated 19820--19821 development cohort.
"""
from research_platform.taiv_mpc import TAIVMPC
from research_platform.terminal_capture import CaptureController


class MGTAIVMPC(TAIVMPC):
    SIMPLE_FACES = 4
    SWITCH_RATIO = .65

    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, kv,
                         simple_faces=self.SIMPLE_FACES,
                         switch_ratio=self.SWITCH_RATIO)


class MGTAIVCapture(CaptureController):
    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, mode="fixed_simple")
        self.core = MGTAIVMPC(vertices, sampling_seed, kv)
