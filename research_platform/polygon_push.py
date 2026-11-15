"""Convex-polygon feedback proposals and full rigid-body policy prediction."""
import copy
import numpy as np,mujoco
from research_platform.contact_policy import rot
from research_platform.tabletop_push import planar_state

SHAPES={
 'triangle':np.array([[-.07,-.04],[.07,-.04],[0.,.08]]),
 'hexagon':np.array([[.075*np.cos(t),.05*np.sin(t)] for t in np.arange(6)*np.pi/3]),
 'rectangle':np.array([[-.06,-.04],[.06,-.04],[.06,.04],[-.06,.04]])}

def make_polygon_model(vertices,friction=.3):
 points=' '.join(str(float(x)) for z in [-.02,.02] for v in vertices for x in [v[0],v[1],z])
 return mujoco.MjModel.from_xml_string(f'''<mujoco model="polygon_push"><option timestep=".0025" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"><flag multiccd="enable"/></option><default><geom solref=".01 1" solimp=".95 .99 .001" friction=".5 .005 .0001" condim="3"/></default><asset><mesh name="polygon" vertex="{points}"/></asset><worldbody><light pos="0 0 2"/><geom name="floor" type="plane" size="2 2 .1" rgba=".2 .22 .25 1"/><body name="object" pos=".15 0 .021"><freejoint/><geom name="object_geom" type="mesh" mesh="polygon" mass=".25" rgba=".9 .5 .1 1"/></body><body name="pusher" pos="0 0 .025"><joint name="px" type="slide" axis="1 0 0"/><joint name="py" type="slide" axis="0 1 0"/><geom name="pusher_geom" type="sphere" size=".02" mass=".1" rgba=".1 .5 .9 1"/></body></worldbody><contact><pair geom1="floor" geom2="object_geom" condim="3" friction="{friction} {friction} .005 .0001 .0001"/></contact><actuator><velocity joint="px" kv="20" ctrllimited="true" ctrlrange="-.15 .15" forcelimited="true" forcerange="-5 5"/><velocity joint="py" kv="20" ctrllimited="true" ctrlrange="-.15 .15" forcelimited="true" forcerange="-5 5"/></actuator></mujoco>''')

class PolygonGeometry:
 def __init__(self,vertices):
  self.vertices=np.array(vertices);edges=np.roll(self.vertices,-1,axis=0)-self.vertices;self.length=np.linalg.norm(edges,axis=1);self.direction=edges/self.length[:,None];self.normal=np.column_stack([self.direction[:,1],-self.direction[:,0]]);self.mid=(self.vertices+np.roll(self.vertices,-1,axis=0))/2;self.bound=np.sum(self.normal*self.vertices,axis=1);self.c2=float(np.mean(np.sum(self.vertices**2,axis=1)));self.outer=np.array([np.linalg.solve(np.array([self.normal[i-1],self.normal[i]]),np.array([self.bound[i-1],self.bound[i]])+.029) for i in range(len(vertices))])
 def intersects(self,a,b):
  low,high=0.,1.;delta=b-a
  for n,bound in zip(self.normal,self.bound+.021):
   value=float(n@a-bound);slope=float(n@delta)
   if abs(slope)<1e-12:
    if value>=-1e-10:return False
   elif slope>0:high=min(high,-value/slope)
   else:low=max(low,-value/slope)
  return high-low>1e-8 and high>1e-6 and low<1-1e-6
 def waypoint(self,p,target):
  gaps=self.bound+.021-self.normal@p
  if np.all(gaps>0):
   i=int(np.argmin(gaps));return p+self.normal[i]*(gaps[i]+.008)
  if not self.intersects(p,target):return target
  nodes=[p,target]+list(self.outer);dist=np.full(len(nodes),np.inf);dist[0]=0.;previous=np.full(len(nodes),-1);done=set()
  for _ in nodes:
   available=[i for i in range(len(nodes)) if i not in done];i=min(available,key=lambda j:dist[j]);done.add(i)
   for j in available:
    if j==i or self.intersects(nodes[i],nodes[j]):continue
    value=dist[i]+np.linalg.norm(nodes[i]-nodes[j])
    if value<dist[j]:dist[j]=value;previous[j]=i
  if previous[1]<0:return p
  j=1
  while previous[j]>0:j=previous[j]
  return nodes[j]
 def vector(self,q,goal,r,n):
  R=rot(q[2]);rw=R@r;inward=-R@n;t=np.array([-inward[1],inward[0]]);e=np.asarray(goal)-np.r_[.15+q[0],q[1],q[2]];e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]));best=None
  for eta in [-.4,0.,.4]:
   f=inward+eta*t;omega=(rw[0]*f[1]-rw[1]*f[0])/self.c2;inner=200*float(f@e[:2])+2*omega*e[2];denom=200*float(f@f)+2*omega**2;scale=max(0.,inner)/denom;value=max(0.,inner)**2/denom;velocity=scale*(f+omega*np.array([-rw[1],rw[0]]))
   if best is None or value>best[0]:best=(value,velocity)
  return best
 def proposal(self,q,goal,face):
  r0=self.mid[face];direction=self.direction[face];n=self.normal[face];R=rot(q[2]);inward=-R@n;t=np.array([-inward[1],inward[0]]);e=np.asarray(goal)-np.r_[.15+q[0],q[1],q[2]];e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]));limit=.3*self.length[face];best=None
  def cross(a,b):return a[0]*b[1]-a[1]*b[0]
  for eta in [-.4,0.,.4]:
   f=inward+eta*t;A=200*float(f@e[:2]);C=200*float(f@f);w0=cross(R@r0,f)/self.c2;slope=cross(R@direction,f)/self.c2;points=[-limit,0.,limit]
   if abs(A)>1e-12:points.append(float(np.clip((e[2]*C/A-w0)/slope,-limit,limit)))
   for s in points:
    omega=w0+slope*s;value=max(0.,A+2*e[2]*omega)**2/(C+2*omega**2)
    if best is None or value>best[0]:best=(value,r0+s*direction)
  return best[1]

class PolygonController:
 def __init__(self,geometry):self.geometry=geometry;self.lock=None;self.r=None;self.phase='approach';self.age=0;self.integral=np.zeros(2)
 def action(self,q,v,goal):
  g=self.geometry;r=self.r;n=g.normal[self.lock];R=rot(q[2]);center=np.array([.15+q[0],q[1]]);p=q[3:5];local=R.T@(p-center);distance=np.linalg.norm(local-r-.024*n)
  if self.phase=='approach' and distance<.014:self.phase='push';self.age=0
  if self.phase=='approach':self.integral[:]=0.;u=4*(center+R@g.waypoint(local,r+.024*n)-p)
  else:
   desired=g.vector(q,goal,r,n)[1]+2*(center+R@(r+.019*n)-p)-.5*v[:2];self.integral=np.clip(self.integral+.05*(desired-v[3:5]),-.05,.05);u=desired+2*self.integral
   if distance>.06:self.phase='approach'
  self.age+=1;u*=min(1.,.15/max(np.linalg.norm(u),1e-12));return u

class PolygonMPC:
 def __init__(self,vertices,method='proposal',sampling_seed=0):
  self.rng=np.random.default_rng(sampling_seed);self.geometry=PolygonGeometry(vertices);self.model=make_polygon_model(vertices,.3);self.data=mujoco.MjData(self.model);self.ctrl=PolygonController(self.geometry);self.method=method;self.step=0;self.plans=0;self.rollout_steps=0;self.plan_records=[]
 @property
 def lock(self):return self.ctrl.lock
 @property
 def phase(self):return self.ctrl.phase
 def candidate_controllers(self,state,goal):
  q,v=planar_state(state);g=self.geometry;candidates=[]
  if self.ctrl.lock is not None:candidates.append(copy.deepcopy(self.ctrl))
  faces=range(len(g.vertices))
  if self.method=='global_uniform':faces=self.rng.integers(len(g.vertices),size=len(g.vertices)-len(candidates))
  for face in faces:
   face=int(face)
   if self.method in ['proposal','rank','stratified_uniform','stratified_endpoint'] and self.ctrl.lock==face:continue
   points=[g.proposal(q,goal,face)] if self.method=='proposal' else [g.mid[face]+o*.5*g.length[face]*g.direction[face] for o in [-.6,-.2,.2,.6]]
   if self.method in ['stratified_uniform','global_uniform']:points=[g.mid[face]+self.rng.uniform(-.3,.3)*g.length[face]*g.direction[face]]
   if self.method=='stratified_endpoint':points=[g.mid[face]+self.rng.choice([-.3,.3])*g.length[face]*g.direction[face]]
   if self.method=='rank':
    local=rot(q[2]).T@(q[3:5]-[.15+q[0],q[1]])
    def score(r):
     target=r+.024*g.normal[face];via=g.waypoint(local,target);travel=np.linalg.norm(local-via)+np.linalg.norm(target-via);return g.vector(q,goal,r,g.normal[face])[0]/(1+travel/.15)
    points=[max(points,key=score)]
   for r in points:
    if self.ctrl.lock==face and np.linalg.norm(r-self.ctrl.r)<1e-12:continue
    c=PolygonController(g);c.lock=face;c.r=r;candidates.append(c)
  return candidates
 def replan(self,state,goal):
  self.plans+=1;candidates=self.candidate_controllers(state,goal);costs=[]
  for initial in candidates:
   c=copy.deepcopy(initial);d=self.data;mujoco.mj_resetData(self.model,d);d.qpos[:]=state.qpos;d.qvel[:]=state.qvel;mujoco.mj_forward(self.model,d);cost=0.
   for k in range(60):
    cq,cv=planar_state(d);d.ctrl[:]=c.action(cq,cv,goal);mujoco.mj_step(self.model,d,nstep=20);cq,cv=planar_state(d);e=np.r_[.15+cq[0],cq[1],cq[2]]-goal;e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]));value=200*(e[:2]@e[:2])+2*e[2]**2;cost+=value
   costs.append(float(cost+60*value))
  best=int(np.argmin(costs))
  if self.ctrl.lock is None or costs[best]<.95*costs[0]:self.ctrl=candidates[best]
  steps=60*len(candidates);self.rollout_steps+=steps;self.plan_records.append(dict(step=self.step,selected_face=self.ctrl.lock,selected_r=self.ctrl.r.tolist(),rollout_steps=steps,costs=costs))
 def action(self,state,goal):
  if self.ctrl.lock is None or (self.step%10==0 and self.ctrl.phase=='push'):self.replan(state,goal)
  self.step+=1;q,v=planar_state(state);return self.ctrl.action(q,v,goal)
