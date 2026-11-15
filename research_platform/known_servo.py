"""Matched, known robot servo parameters for a plant-authority diagnostic.
Changes plant and all compared controllers together; not a new control method.
"""
import numpy as np
from research_platform.support_calibration import SupportObserver
from research_platform.envelope_validity import ValidEnvelopeMPC
from research_platform.successor_rollout import SuccessorRolloutMPC
from research_platform.terminal_capture import CaptureController
from research_platform.contact_policy import rot


def configure_servo(model,kv):
 model.actuator_gainprm[:,0]=kv;model.actuator_biasprm[:,2]=-kv
 return model

class ServoObserver(SupportObserver):
 def __init__(self,vertices,kv,samples=5):super().__init__(vertices,samples);self.kv=kv
 def observe(self,q,v):
  if self.previous_velocity is None:return None
  g=self.geometry;R=rot(q[2]);local=R.T@(q[3:5]-[.15+q[0],q[1]])
  reaction=R.T@(.1*(v[3:5]-self.previous_velocity)/.05-np.clip(self.kv*(self.previous_command-v[3:5]),-5,5))
  face=int(np.argmax(g.normal@local-g.bound));n=g.normal[face];normal=float(n@reaction);tangent=float(g.direction[face]@reaction);offset=float(g.direction[face]@(local-g.mid[face]));bound=float(n@local-.02)
  if not self.complete[face] and normal>.15 and abs(tangent)<.8*normal and abs(offset)<.4*g.length[face] and abs(bound-g.bound[face])<.02:
   self.accepted[face]+=1;self.samples[face].append(bound)
   if len(self.samples[face])==self.sample_count:
    self.complete[face]=True;return face,float(np.median(self.samples[face]))
  return None

class KnownServoMixin:
 def __init__(self,*args,kv=20.,**kwargs):
  super().__init__(*args,**kwargs);self.kv=kv;self.observer=ServoObserver(self.geometry.vertices,kv);configure_servo(self.model,kv)
 def replan(self,state,goal):
  configure_servo(self.model,self.kv);return super().replan(state,goal)

class ServoEnvelope(KnownServoMixin,ValidEnvelopeMPC):pass
class ServoSuccessor(KnownServoMixin,SuccessorRolloutMPC):pass

class KnownServoCapture(CaptureController):
 def __init__(self,vertices,sampling_seed=0,mode='fixed_simple',kv=20.):
  super().__init__(vertices,sampling_seed,mode)
  self.core=ServoSuccessor(vertices,sampling_seed,mode='tail60',kv=kv) if mode.startswith('tail') else ServoEnvelope(vertices,sampling_seed,clip_contact=False,kv=kv)
