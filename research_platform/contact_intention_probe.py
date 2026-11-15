"""Object-subgoal policy-library probe inspired by hierarchical contact intentions.
Local adaptation, not a reproduction of RL/ComFree-MPC. No training or truth access.
"""
import copy
import numpy as np
from research_platform.prediction_fidelity_v3 import FidelityMPC
from research_platform.envelope_calibration import EnvelopeController
from research_platform.terminal_capture import CaptureController
from research_platform.tabletop_push import planar_state

class IntentionController(EnvelopeController):
 def __init__(self,geometry,width,intention,anchor,final_goal):
  super().__init__(geometry,width);self.intention=intention;self.anchor=np.asarray(anchor).copy();self.final_goal=np.asarray(final_goal).copy();self.subgoal_active=True
 def effective_goal(self,q,goal):
  if not np.allclose(goal,self.final_goal,atol=1e-12,rtol=0):self.subgoal_active=False
  if self.subgoal_active:
   if self.intention=='rotate':
    yaw=np.arctan2(np.sin(q[2]-goal[2]),np.cos(q[2]-goal[2]))
    if abs(yaw)<.12:self.subgoal_active=False
   elif np.linalg.norm(np.r_[.15+q[0],q[1]]-np.asarray(goal[:2]))<.020:self.subgoal_active=False
  return self.anchor if self.subgoal_active else goal
 def action(self,q,v,goal):return super().action(q,v,self.effective_goal(q,goal))

class IntentionMPC(FidelityMPC):
 def __init__(self,vertices,sampling_seed=0,mode='library',kv=40.):
  self.intention_mode=mode;super().__init__(vertices,sampling_seed,'fine',kv)
 def candidate_controllers(self,state,goal):
  if self.intention_mode=='direct':return super().candidate_controllers(state,goal)
  q,_=planar_state(state);g=self.geometry;width=self.widths()
  if self.intention_mode in ['library','grid']:candidates=super().candidate_controllers(state,goal)
  else:candidates=[copy.deepcopy(self.ctrl)] if self.ctrl.lock is not None else []
  for face in range(len(g.vertices)):
   if self.intention_mode=='grid':
    for frac in [-.3,.3]:
     c=EnvelopeController(g,width);c.lock=face;c.r=g.mid[face]+frac*g.length[face]*g.direction[face];candidates.append(c)
   else:
    for intention in (['rotate','translate'] if self.intention_mode=='library' else [self.intention_mode]):
     anchor=np.asarray(goal).copy()
     if intention=='rotate':anchor[:2]=[.15+q[0],q[1]]
     else:anchor[2]=q[2]
     c=IntentionController(g,width,intention,anchor,goal);c.lock=face;c.r=g.proposal(q,anchor,face);candidates.append(c)
  return candidates
 def replan(self,state,goal):
  super().replan(state,goal);self.plan_records[-1]['intention']=getattr(self.ctrl,'intention','direct');self.plan_records[-1]['subgoal_active']=getattr(self.ctrl,'subgoal_active',False)

class IntentionCapture(CaptureController):
 def __init__(self,vertices,sampling_seed=0,mode='library',kv=40.):
  super().__init__(vertices,sampling_seed,mode='fixed_simple');self.core=IntentionMPC(vertices,sampling_seed,mode,kv)
