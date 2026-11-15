"""Synthetic measurement noise applied before every controller call."""
import numpy as np,mujoco

def noisy_observation(model,state,rng,noise):
 d=mujoco.MjData(model);d.qpos[:]=state.qpos;d.qvel[:]=state.qvel
 position=noise.get('position',0.);yaw=noise.get('yaw',0.);velocity=noise.get('velocity',0.);yaw_rate=noise.get('yaw_rate',0.)
 d.qpos[[0,1,7,8]]+=rng.normal(0,position,4)
 angle=rng.normal(0,yaw);rotation=np.array([np.cos(angle/2),0,0,np.sin(angle/2)]);quat=d.qpos[3:7].copy();mujoco.mju_mulQuat(d.qpos[3:7],rotation,quat)
 d.qvel[[0,1,6,7]]+=rng.normal(0,velocity,4);d.qvel[5]+=rng.normal(0,yaw_rate)
 return d
