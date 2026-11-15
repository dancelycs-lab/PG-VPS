"""Prediction discretization screen. Plant and control sampling period unchanged.
Coarse 10 ms predictions retain MuJoCo refsafe; they also soften stiff contacts.
No error certificate is claimed.
"""
import copy
import numpy as np,mujoco
from research_platform.known_servo import ServoEnvelope,configure_servo
from research_platform.terminal_capture import CaptureController
from research_platform.tabletop_push import planar_state

def rollout_cost(model,data,initial,qpos,qvel,goal,timestep):
 previous=model.opt.timestep;model.opt.timestep=timestep
 try:
  c=copy.deepcopy(initial);mujoco.mj_resetData(model,data);data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data);cost=0.;nstep=round(.05/timestep)
  for _ in range(60):
   q,v=planar_state(data);data.ctrl[:]=c.action(q,v,goal);mujoco.mj_step(model,data,nstep=nstep);q,v=planar_state(data);e=np.r_[.15+q[0],q[1],q[2]]-goal;e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]));value=200*(e[:2]@e[:2])+2*e[2]**2;cost+=value
  return float(cost+60*value)
 finally:model.opt.timestep=previous

def decision(costs,has_incumbent,switch_ratio=.95):
 best=int(np.argmin(costs));return best if not has_incumbent or costs[best]<switch_ratio*costs[0] else 0

class FidelityMPC(ServoEnvelope):
 def __init__(self,vertices,sampling_seed=0,mode='fine',kv=40.):
  super().__init__(vertices,sampling_seed=sampling_seed,clip_contact=False,kv=kv);self.fidelity_mode=mode;self.snapshots=[];self.physics_steps=0;self.switch_ratio=.95
 def replan(self,state,goal):
  configure_servo(self.model,self.kv);self.plans+=1;candidates=self.candidate_controllers(state,goal);incumbent=self.ctrl.lock is not None
  if self.fidelity_mode=='fine' and self.plans in [1,5,10,15]:
   self.snapshots.append(dict(plan=self.plans,qpos=state.qpos.copy(),qvel=state.qvel.copy(),vertices=self.geometry.vertices.copy(),candidates=copy.deepcopy(candidates),goal=goal,incumbent=incumbent,kv=self.kv))
  mode=self.fidelity_mode;coarse_dt=.005 if mode=='verify_half' else .01
  timestep={'fine':.0025,'half':.005,'quarter':.01}.get(mode,coarse_dt)
  direct_initial=not incumbent and mode in ['verify','verify_half','top2']
  if direct_initial:timestep=.0025
  costs=[rollout_cost(self.model,self.data,c,state.qpos,state.qvel,goal,timestep) for c in candidates];chosen=decision(costs,incumbent,self.switch_ratio);proposal=chosen;fine_costs={};verified=False
  if mode in ['verify','verify_half','top2']:
   indices=[]
   if not incumbent:indices=list(range(len(candidates)))
   elif mode=='top2':indices=sorted(set([0]+list(np.argsort(costs)[:2])))
   elif chosen!=0:indices=[0,chosen]
   fine_costs={int(j):rollout_cost(self.model,self.data,candidates[j],state.qpos,state.qvel,goal,.0025) for j in indices} if not direct_initial else {j:cost for j,cost in enumerate(costs)}
   if fine_costs:
    best=min(fine_costs,key=fine_costs.get);chosen=best if not incumbent or fine_costs[best]<self.switch_ratio*fine_costs[0] else 0;verified=True
   if incumbent and chosen!=0:assert verified and fine_costs[chosen]<self.switch_ratio*fine_costs[0]
  self.ctrl=candidates[chosen];additional_fine=0 if direct_initial else len(fine_costs);steps=60*(len(candidates)+additional_fine);self.rollout_steps+=steps;physics=60*(len(candidates)*round(.05/timestep)+additional_fine*20);self.physics_steps+=physics
  self.plan_records.append(dict(step=self.step,selected_face=self.ctrl.lock,selected_r=self.ctrl.r.tolist(),rollout_steps=steps,physics_steps=physics,costs=costs,selected_index=chosen,proposed_index=proposal,fine_costs=fine_costs,verified=verified,incumbent=incumbent))

class FidelityCapture(CaptureController):
 def __init__(self,vertices,sampling_seed=0,mode='fine',kv=40.):
  super().__init__(vertices,sampling_seed,mode='fixed_simple');self.core=FidelityMPC(vertices,sampling_seed,mode,kv)
