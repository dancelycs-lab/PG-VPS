"""Object-subgoal candidates with established switch-triggered fine verification."""
from research_platform.contact_intention_probe import IntentionMPC
from research_platform.terminal_capture import CaptureController
class BudgetIntentionMPC(IntentionMPC):
 def __init__(self,vertices,sampling_seed=0,mode='library_verify',kv=40.):
  library,precision=mode.rsplit('_',1);super().__init__(vertices,sampling_seed,library,kv);self.fidelity_mode={'fine':'fine','half':'half','verify':'verify_half'}[precision]
class BudgetIntentionCapture(CaptureController):
 def __init__(self,vertices,sampling_seed=0,mode='library_verify',kv=40.):
  super().__init__(vertices,sampling_seed,mode='fixed_simple');self.core=BudgetIntentionMPC(vertices,sampling_seed,mode,kv)
