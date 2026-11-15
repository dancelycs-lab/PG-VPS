"""Strong simple baseline: force-gated contact recovery, no geometry learning."""
import numpy as np
from research_platform.polygon_push import PolygonController,PolygonMPC
from research_platform.contact_policy import rot
class ForceRecoveryController(PolygonController):
 def __init__(self,geometry):
  super().__init__(geometry);self.previous_velocity=None;self.previous_command=None
 def action(self,q,v,goal):
  normal=rot(q[2])@self.geometry.normal[self.lock];force=0.
  if self.previous_velocity is not None:
   reaction=.1*(v[3:5]-self.previous_velocity)/.05-np.clip(20*(self.previous_command-v[3:5]),-5,5);force=float(normal@reaction)
  u=super().action(q,v,goal);e=np.r_[.15+q[0],q[1],q[2]]-goal;e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]))
  if self.phase=='push' and force<.15 and (np.linalg.norm(e[:2])>=.025 or abs(e[2])>=.15):
   u-=normal*max(0.,float(normal@u)+.03)
  u*=min(1.,.15/max(np.linalg.norm(u),1e-12));self.previous_velocity=v[3:5].copy();self.previous_command=u.copy();return u
class ForceRecoveryMPC(PolygonMPC):
 def __init__(self,vertices,sampling_seed=0):
  super().__init__(vertices,'proposal',sampling_seed);self.ctrl=ForceRecoveryController(self.geometry)
 def candidate_controllers(self,state,goal):
  candidates=super().candidate_controllers(state,goal);result=[]
  for candidate in candidates:
   if isinstance(candidate,ForceRecoveryController):result.append(candidate)
   else:
    c=ForceRecoveryController(self.geometry);c.__dict__.update(candidate.__dict__);result.append(c)
  return result
