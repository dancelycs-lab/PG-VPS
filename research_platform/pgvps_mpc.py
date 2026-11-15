"""Progress-guarded verified policy-switching MPC (PG-VPS-MPC).

The finite policy set contains one closed-loop pushing policy per estimated
object face.  Coarse rollouts propose an incumbent replacement, fine rollouts
verify only the incumbent/challenger pair, and measured task progress selects
the replacement ratio.
"""
from research_platform.direct_pgtaiv_mpc import (
    DirectPGTAIVMPC,
)
from research_platform.terminal_capture import CaptureController


class PGVPSMPC(DirectPGTAIVMPC):
    """Submission-facing name for the evidence-supported direct-basis method."""


class PGVPSCapture(CaptureController):
    def __init__(self, vertices, sampling_seed=0, kv=40.):
        super().__init__(vertices, sampling_seed, mode="fixed_simple")
        self.core = PGVPSMPC(vertices, sampling_seed, kv)
