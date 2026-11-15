"""Geometry-guided feedback rollout baseline for planar pushing.
Uses known contact points and goal pose, not a learned or claimed novel method.
"""
import numpy as np,mujoco

def rot(a):
 c,s=np.cos(a),np.sin(a);return np.array([[c,-s],[s,c]])

def parameters(half_size=(.06,.04)):
 result=[]
 for axis in range(2):
  for sign in [-1.,1.]:
   for offset in [-.6,-.2,.2,.6]:
    r=np.zeros(2);r[axis]=sign*half_size[axis];r[1-axis]=offset*half_size[1-axis];normal=np.zeros(2);normal[axis]=sign
    for gain in [.75,2.]:result.append((r,normal,gain))
 return result

def intersects(a,b,extent):
 # Open rectangle intersection, used only for routing outside the object.
 lo,hi=0.,1.;delta=b-a
 for j in range(2):
  if abs(delta[j])<1e-12:
   if abs(a[j])>=extent[j]:return False
  else:
   t=sorted([(-extent[j]-a[j])/delta[j],(extent[j]-a[j])/delta[j]]);lo=max(lo,t[0]);hi=min(hi,t[1])
 return hi-lo>1e-8 and hi>1e-6 and lo<1.-1e-6

def approach_waypoint(p,target,half_size=(.06,.04)):
 extent=np.asarray(half_size)+.021;outer=extent+.008
 # If currently touching, first leave along the nearest outward face.
 if np.all(abs(p)<extent):
  j=np.argmin(extent-abs(p));w=p.copy();w[j]=np.copysign(outer[j],p[j] if abs(p[j])>1e-8 else 1.);return w
 if not intersects(p,target,extent):return target
 corners=[np.array([x,y]) for x in [-outer[0],outer[0]] for y in [-outer[1],outer[1]]];nodes=[p,target]+corners;n=len(nodes);dist=np.full(n,np.inf);dist[0]=0;prev=np.full(n,-1);done=set()
 for _ in range(n):
  available=[i for i in range(n) if i not in done];i=min(available,key=lambda j:dist[j]);done.add(i)
  for j in available:
   if j==i or intersects(nodes[i],nodes[j],extent):continue
   trial=dist[i]+np.linalg.norm(nodes[i]-nodes[j])
   if trial<dist[j]:dist[j]=trial;prev[j]=i
 if prev[1]<0:return p
 j=1
 while prev[j]>0:j=prev[j]
 return nodes[j]

def action(q,param,goal,pushing=False):
 r,normal,gain=param;center=np.array([.15+q[0],q[1]]);R=rot(q[2]);p=q[3:5];local=R.T@(p-center);target=r+.024*normal
 if np.linalg.norm(local-target)<.014:pushing=True
 if pushing:
  destination=np.asarray(goal[:2])+rot(goal[2])@(r+.019*normal)
  u=gain*(destination-p)
 else:u=4.*(center+R@approach_waypoint(local,target)-p)
 u*=min(1.,.3/max(np.linalg.norm(u),1e-12));return u,pushing

def rollout_policy(m,d,q,v,param,cfg):
 mujoco.mj_resetData(m,d);d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);cost=0.;pushing=False;first=None;w=cfg['weights']
 for k in range(cfg['horizon']):
  u,pushing=action(d.qpos,param,cfg['goal'],pushing)
  if first is None:first=u.copy()
  d.ctrl[:]=u;mujoco.mj_step(m,d,nstep=round(cfg['dt']/m.opt.timestep));obj=np.r_[.15+d.qpos[0],d.qpos[1],d.qpos[2]];err=obj-cfg['goal'];err[2]=np.arctan2(np.sin(err[2]),np.cos(err[2]));cost+=w['position']*(err[:2]@err[:2])+w['yaw']*err[2]**2+w['pusher_object']*np.sum((d.qpos[3:5]-obj[:2])**2)+w['control']*(u@u)+cfg.get('velocity_weight',0.)*np.sum(d.qvel[:2]**2)
 return cost,first

def choose_policy(models,datas,q,v,cfg):
 values=[];controls=[]
 params=parameters()
 if cfg.get('coarse_contacts',False):params=[params[i] for i in [0,6,8,14,16,22,24,30]]
 for param in params:
  results=[rollout_policy(m,d,q,v,param,cfg) for m,d in zip(models,datas)];values.append(np.mean([x[0] for x in results]));controls.append(results[0][1])
 idx=int(np.argmin(values));return controls[idx],float(values[idx])
