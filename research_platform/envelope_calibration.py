"""Uncertainty-envelope approach and support-wise contact calibration.
Known normals and bounded support errors; conditional geometric certificates only.
"""
import numpy as np
from scipy.spatial import ConvexHull
from research_platform.polygon_push import PolygonController,PolygonGeometry
from research_platform.support_calibration import SupportCalibratedMPC
from research_platform.contact_policy import rot
from research_platform.force_recovery import ForceRecoveryController

def halfspace_vertices(normals,bounds):
 points=[]
 for i in range(len(bounds)):
  for j in range(i):
   A=np.array([normals[i],normals[j]])
   if abs(np.linalg.det(A))<1e-10:continue
   p=np.linalg.solve(A,[bounds[i],bounds[j]])
   if np.all(normals@p<=bounds+1e-8):points.append(p)
 points=np.unique(np.round(points,12),axis=0)
 if len(points)<3:raise ValueError('Empty envelope')
 return points[ConvexHull(points).vertices]

def robust_face_interval(geometry,width,face):
 n=geometry.normal[face];t=geometry.direction[face];lower=geometry.bound-width;upper=geometry.bound+width;lo,hi=-np.inf,np.inf
 for j,nj in enumerate(geometry.normal):
  if j==face:continue
  cosine=float(nj@n);slope=float(nj@t);rhs=lower[j]-(upper[face] if cosine>=0 else lower[face])*cosine
  if abs(slope)<1e-12:
   if rhs<0:return None
  elif slope>0:hi=min(hi,rhs/slope)
  else:lo=max(lo,rhs/slope)
 return (lo,hi) if hi-lo>.004 else None

class EnvelopeController(ForceRecoveryController):
 def __init__(self,geometry,width,recover_contact=False):
  super().__init__(geometry);self.recover_contact=recover_contact;self.width=np.asarray(width).copy();self.guard_key=None;self.guard=None;self.certified=False
 def action(self,q,v,goal):
  if self.phase!='approach':return super().action(q,v,goal) if self.recover_contact else PolygonController.action(self,q,v,goal)
  g=self.geometry;key=tuple(g.bound)+tuple(self.width)
  if key!=self.guard_key:self.guard=PolygonGeometry(halfspace_vertices(g.normal,g.bound+self.width));self.guard_key=key
  R=rot(q[2]);center=np.array([.15+q[0],q[1]]);p=q[3:5];local=R.T@(p-center);target=self.r+(.024+self.width[self.lock])*g.normal[self.lock]
  if np.linalg.norm(local-target)<.014:self.phase='push';self.age=0;return super().action(q,v,goal) if self.recover_contact else PolygonController.action(self,q,v,goal)
  self.integral[:]=0.;u=4*(center+R@self.guard.waypoint(local,target)-p);u*=min(1.,.15/max(np.linalg.norm(u),1e-12));self.age+=1;return u

class EnvelopeCalibratedMPC(SupportCalibratedMPC):
 def __init__(self,vertices,sampling_seed=0,width=.012,clip_contact=True,update_geometry=True,force_recovery=False):
  super().__init__(vertices,active=True,sampling_seed=sampling_seed,samples=5,update_geometry=update_geometry);self.force_recovery=force_recovery;self.initial_width=width;self.clip_contact=clip_contact;self.ctrl=EnvelopeController(self.geometry,self.widths(),recover_contact=force_recovery)
 def widths(self):return np.where(self.observer.complete & self.update_geometry,.001,self.initial_width)
 def candidate_controllers(self,state,goal):
  candidates=super().candidate_controllers(state,goal);width=self.widths();result=[]
  for candidate in candidates:
   if isinstance(candidate,EnvelopeController):c=candidate;c.width=width.copy()
   else:c=EnvelopeController(self.geometry,width);c.__dict__.update(candidate.__dict__)
   c.recover_contact=self.force_recovery
   interval=robust_face_interval(self.geometry,width,c.lock);c.certified=interval is not None
   retained=self.ctrl.lock==c.lock and self.ctrl.r is not None and np.linalg.norm(self.ctrl.r-c.r)<1e-12
   if self.clip_contact and interval is not None and c.phase=='approach' and not retained:
    n=self.geometry.normal[c.lock];t=self.geometry.direction[c.lock];s=float(np.clip(t@c.r,interval[0]+.002,interval[1]-.002));c.r=self.geometry.bound[c.lock]*n+s*t
   result.append(c)
  return result
 def action(self,state,goal):
  self.ctrl.width=self.widths();return super().action(state,goal)
