"""Exploratory policy improvement with a cheap contact-switching tail.
Only the first policy is executed. Future geometry remains fixed.
"""
import copy
import numpy as np
import mujoco
from research_platform.envelope_validity import ValidEnvelopeMPC
from research_platform.envelope_calibration import EnvelopeController
from research_platform.tabletop_push import planar_state
from research_platform.contact_policy import rot

class SuccessorRolloutMPC(ValidEnvelopeMPC):
 def __init__(self,vertices,sampling_seed=0,mode='tail60'):
  super().__init__(vertices,sampling_seed=sampling_seed,clip_contact=False);self.tail_mode=mode
 def successor(self,current,q,goal):
  g=self.geometry;width=self.widths();candidates=[current];local=rot(q[2]).T@(q[3:5]-[.15+q[0],q[1]])
  for face in range(len(g.bound)):
   if face==current.lock:continue
   c=EnvelopeController(g,width);c.lock=face;c.r=g.proposal(q,goal,face);candidates.append(c)
  scores=[]
  for c in candidates:
   target=c.r+(.024+width[c.lock])*g.normal[c.lock];travel=0. if c.phase=='push' else np.linalg.norm(local-target)
   scores.append(g.vector(q,goal,c.r,g.normal[c.lock])[0]/(1+travel/.15))
  best=int(np.argmax(scores));return candidates[best] if scores[best]>1.05*scores[0] else current
 def replan(self,state,goal):
  self.plans+=1;candidates=self.candidate_controllers(state,goal);costs=[];switches=[]
  horizon=120 if self.tail_mode=='tail120' else 60
  for initial in candidates:
   c=copy.deepcopy(initial);d=self.data;mujoco.mj_resetData(self.model,d);d.qpos[:]=state.qpos;d.qvel[:]=state.qvel;mujoco.mj_forward(self.model,d);cost=0.;changes=0;push=0
   for k in range(horizon):
    q,v=planar_state(d)
    if push>=20 and k%10==0 and c.phase=='push':
     old=c.lock;c=self.successor(c,q,goal);changes+=int(c.lock!=old)
    d.ctrl[:]=c.action(q,v,goal);push+=int(c.phase=='push');mujoco.mj_step(self.model,d,nstep=20)
    q,v=planar_state(d);e=np.r_[.15+q[0],q[1],q[2]]-goal;e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]));value=200*(e[:2]@e[:2])+2*e[2]**2;cost+=value
   costs.append(float(cost+horizon*value));switches.append(changes)
  best=int(np.argmin(costs))
  if self.ctrl.lock is None or costs[best]<.95*costs[0]:self.ctrl=candidates[best]
  steps=horizon*len(candidates);self.rollout_steps+=steps
  self.plan_records.append(dict(step=self.step,selected_face=self.ctrl.lock,selected_r=self.ctrl.r.tolist(),rollout_steps=steps,costs=costs,tail_switches=switches))
