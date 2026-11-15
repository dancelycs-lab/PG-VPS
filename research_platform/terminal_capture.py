"""Exploratory goal capture; nominal prediction is NOT an invariance certificate.
Simple hysteresis is an explicit competing explanation for predictive capture.
"""
import numpy as np
import mujoco
from research_platform.envelope_validity import ValidEnvelopeMPC
from research_platform.successor_rollout import SuccessorRolloutMPC
from research_platform.tabletop_push import planar_state


def goal_error(q,goal):
 e=np.r_[.15+q[0],q[1],q[2]]-goal
 e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]))
 return float(np.linalg.norm(e[:2])),float(abs(e[2]))

class CaptureController:
 def __init__(self,vertices,sampling_seed=0,mode='fixed_none'):
  base,self.capture_mode=mode.split('_')
  self.core=SuccessorRolloutMPC(vertices,sampling_seed,mode='tail60') if base=='tail' else ValidEnvelopeMPC(vertices,sampling_seed,clip_contact=False)
  self.holding=False;self.capture_records=[];self.capture_steps=0;self.hold_ticks=0
 def __getattr__(self,name):return getattr(self.core,name)
 def predicted_hold(self,state,goal):
  m=self.core.model;d=mujoco.MjData(m);d.qpos[:]=state.qpos;d.qvel[:]=state.qvel;mujoco.mj_forward(m,d);d.ctrl[:]=0
  for k in range(10):
   mujoco.mj_step(m,d,nstep=20);self.capture_steps+=1;q,_=planar_state(d);pos,yaw=goal_error(q,goal)
   if pos>=.023 or yaw>=.14:return False
  return True
 def action(self,state,goal):
  c=self.core;q,v=planar_state(state);pos,yaw=goal_error(q,goal)
  eligible=c.lock is not None and c.probe_face is None and self.capture_mode!='none'
  gate=eligible and pos<(.023 if self.holding else .020) and yaw<(.14 if self.holding else .12)
  if gate and self.capture_mode=='predict':gate=self.predicted_hold(state,goal)
  old=self.holding;self.holding=bool(gate)
  if old!=self.holding:self.capture_records.append(dict(step=c.step,holding=self.holding,position=pos,yaw=yaw))
  if self.holding:
   u=np.zeros(2);c.ctrl.integral[:]=0.;c.observer.remember(v,u);c.step+=1;self.hold_ticks+=1
   return u
  return c.action(state,goal)
