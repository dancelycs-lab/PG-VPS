"""Independent support-plane calibration with bounded per-face acquisition.
Known face normals/topology; unknown independent offsets. No true contact access.
"""
from collections import deque
import numpy as np,mujoco
from research_platform.polygon_push import PolygonMPC,PolygonGeometry,make_polygon_model
from research_platform.tabletop_push import planar_state
from research_platform.contact_policy import rot

def vertices_from_support(normals,bounds):
 normals=np.asarray(normals);bounds=np.asarray(bounds)
 vertices=np.array([np.linalg.solve(np.array([normals[i-1],normals[i]]),np.array([bounds[i-1],bounds[i]])) for i in range(len(bounds))])
 if np.any(normals@vertices.T>bounds[:,None]+1e-8) or np.min(np.linalg.norm(np.roll(vertices,-1,axis=0)-vertices,axis=1))<.005:raise ValueError('Invalid or degenerate polygon support update')
 return vertices

class SupportObserver:
 def __init__(self,vertices,samples=20):
  self.sample_count=samples;self.geometry=PolygonGeometry(vertices);self.samples=[deque(maxlen=samples) for _ in vertices];self.complete=np.zeros(len(vertices),dtype=bool);self.previous_velocity=None;self.previous_command=None;self.accepted=np.zeros(len(vertices),dtype=int)
 def observe(self,q,v):
  if self.previous_velocity is None:return None
  g=self.geometry;R=rot(q[2]);local=R.T@(q[3:5]-[.15+q[0],q[1]])
  reaction=R.T@(.1*(v[3:5]-self.previous_velocity)/.05-np.clip(20*(self.previous_command-v[3:5]),-5,5))
  face=int(np.argmax(g.normal@local-g.bound));n=g.normal[face];normal=float(n@reaction);tangent=float(g.direction[face]@reaction);offset=float(g.direction[face]@(local-g.mid[face]));bound=float(n@local-.02)
  if not self.complete[face] and normal>.15 and abs(tangent)<.8*normal and abs(offset)<.4*g.length[face] and abs(bound-g.bound[face])<.02:
   self.accepted[face]+=1;self.samples[face].append(bound)
   if len(self.samples[face])==self.sample_count:
    self.complete[face]=True;return face,float(np.median(self.samples[face]))
  return None
 def remember(self,v,u):self.previous_velocity=v[3:5].copy();self.previous_command=u.copy()

class SupportCalibratedMPC(PolygonMPC):
 def __init__(self,vertices,active=True,sampling_seed=0,strategy='always',samples=20,update_geometry=True):
  super().__init__(vertices,'proposal',sampling_seed);self.observer=SupportObserver(vertices,samples=samples);self.active=active;self.update_geometry=update_geometry;self.strategy=strategy;self.last_sample_step=np.zeros(len(vertices),dtype=int);self.first_push_step=np.full(len(vertices),-1,dtype=int);self.attempted=np.zeros(len(vertices),dtype=bool);self.probe_face=None;self.probe_ticks=0;self.calibration_records=[];self.probe_records=[]
 def action(self,state,goal):
  q,v=planar_state(state);previous_accepted=self.observer.accepted.copy();update=self.observer.observe(q,v);self.last_sample_step[self.observer.accepted>previous_accepted]=self.step;needs_plan=False
  if update is not None:
   face,bound=update;old=self.geometry;bounds=old.bound.copy();bounds[face]=bound;valid=True
   try:vertices=vertices_from_support(old.normal,bounds)
   except ValueError:valid=False
   if valid and self.update_geometry:
    lock=self.ctrl.lock;fraction=0. if lock is None else float(old.direction[lock]@(self.ctrl.r-old.mid[lock]))/old.length[lock]
    self.geometry=PolygonGeometry(vertices);self.ctrl.geometry=self.geometry;self.observer.geometry=self.geometry
    if lock is not None:self.ctrl.r=self.geometry.mid[lock]+fraction*self.geometry.length[lock]*self.geometry.direction[lock]
    self.model=make_polygon_model(vertices,.3);self.data=mujoco.MjData(self.model);needs_plan=True
   self.calibration_records.append(dict(step=self.step,face=face,bound=bound,previous_bound=float(old.bound[face]),applied=valid and self.update_geometry,accepted=int(self.observer.accepted[face])))
  if self.probe_face is not None and (self.observer.complete[self.probe_face] or self.probe_ticks>=40):
   self.probe_records.append(dict(end_step=self.step,face=self.probe_face,ticks=self.probe_ticks,observed=bool(self.observer.complete[self.probe_face])));self.probe_face=None;needs_plan=True
  if self.probe_face is None:
   if needs_plan:self.replan(state,goal)
   u=super().action(state,goal)
   face=self.ctrl.lock
   if self.ctrl.phase=='push' and self.first_push_step[face]<0:self.first_push_step[face]=self.step
   e=np.r_[.15+q[0],q[1],q[2]]-goal;e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]));outside=np.linalg.norm(e[:2])>=.025 or abs(e[2])>=.15
   deficit=self.step-max(self.first_push_step[face],self.last_sample_step[face])>=10
   allowed=(getattr(self,'planned_probe',None)==face) if self.strategy=='planned' else (self.strategy=='always' or (deficit and outside))
   if self.active and allowed and self.ctrl.phase=='push' and not self.observer.complete[face] and not self.attempted[face]:self.probe_face=face;self.attempted[face]=True;self.probe_ticks=0
  else:self.step+=1;u=np.zeros(2)
  if self.probe_face is not None:
   g=self.geometry;face=self.probe_face;R=rot(q[2]);local=R.T@(q[3:5]-[.15+q[0],q[1]]);t=g.direction[face]
   u=R@(-.03*g.normal[face]+4*float(t@(self.ctrl.r-local))*t);u*=min(1.,.06/max(np.linalg.norm(u),1e-12));self.ctrl.integral[:]=0.;self.probe_ticks+=1
  self.observer.remember(v,u);return u
