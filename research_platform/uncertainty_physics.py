"""Declared true-plant perturbations; never passed to the controller model."""
import mujoco

def configure_truth(model,cfg):
 if 'object_mass' in cfg:
  body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object');ratio=cfg['object_mass']/model.body_mass[body];model.body_mass[body]*=ratio;model.body_inertia[body]*=ratio
  mujoco.mj_setConst(model,mujoco.MjData(model))
 return model
