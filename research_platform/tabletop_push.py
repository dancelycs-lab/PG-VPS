"""Free rigid body pushing with gravity and explicit Coulomb floor contacts."""
import numpy as np
import mujoco

def make_model(half_size=(.06,.04), friction=.3, mass=.25):
 a,b=half_size
 return mujoco.MjModel.from_xml_string(f'''<mujoco model="tabletop_push">
 <option timestep=".0025" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
 <default><geom solref=".01 1" solimp=".95 .99 .001" friction=".5 .005 .0001" condim="3"/></default>
 <worldbody><light pos="0 0 2"/><geom name="floor" type="plane" size="2 2 .1" rgba=".2 .22 .25 1"/>
 <body name="object" pos=".15 0 .021"><freejoint/><geom name="object_geom" type="box" size="{a} {b} .02" mass="{mass}" rgba=".9 .5 .1 1"/></body>
 <body name="pusher" pos="0 0 .025"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><geom name="pusher_geom" type="sphere" size=".02" mass=".1" rgba=".1 .5 .9 1"/></body>
 </worldbody><contact><pair geom1="floor" geom2="object_geom" condim="3" friction="{friction} {friction} .005 .0001 .0001"/></contact>
 <actuator><velocity joint="" name="unused"/></actuator></mujoco>'''.replace('<joint type="slide" axis="1 0 0"/>','<joint name="px" type="slide" axis="1 0 0"/>').replace('<joint type="slide" axis="0 1 0"/>','<joint name="py" type="slide" axis="0 1 0"/>').replace('<velocity joint="" name="unused"/>','<velocity joint="px" kv="20" ctrllimited="true" ctrlrange="-.15 .15" forcelimited="true" forcerange="-5 5"/><velocity joint="py" kv="20" ctrllimited="true" ctrlrange="-.15 .15" forcelimited="true" forcerange="-5 5"/>'))

def planar_state(d):
 w,x,y,z=d.qpos[3:7];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
 return np.r_[d.qpos[0]-.15,d.qpos[1],yaw,d.qpos[7:9]],np.r_[d.qvel[:2],d.qvel[5],d.qvel[6:8]]

def initialize(m,seed):
 d=mujoco.MjData(m);rng=np.random.default_rng(seed);d.qpos[1]=rng.uniform(-.03,.03);angle=rng.uniform(-.3,.3);d.qpos[3:7]=[np.cos(angle/2),0,0,np.sin(angle/2)];mujoco.mj_forward(m,d);mujoco.mj_step(m,d,nstep=200);return d
