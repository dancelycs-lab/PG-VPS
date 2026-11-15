"""Five frozen public sampling optimizers on the exact PG-TAIV cohort.

These are task-adapted algorithm ports, not complete upstream pushing systems.
Development-selected settings are reused without retuning.
"""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_platform.known_servo import configure_servo
from research_platform.polygon_push import make_polygon_model
from research_platform.public_sampling import PublicSampler
from research_platform.public_sampling_extra import ExtraPublicSampler
from research_platform.push_measurement import noisy_observation
from research_platform.tabletop_push import initialize, planar_state
from research_platform.uncertainty_physics import configure_truth


METHODS = ["ps", "mppi", "icem", "cem", "smppi"]


def make_controller(row, settings):
    cls = ExtraPublicSampler if row["mode"] in ["cem", "smppi"] else PublicSampler
    controller = cls(row["estimated_vertices"], row["mode"],
                     row["seed"] + 200000, settings[row["mode"]])
    configure_servo(controller.model, row["cfg"]["servo_kv"])
    return controller


def run(task):
    out, row, settings = task
    cfg = row["cfg"]
    model = configure_servo(
        configure_truth(make_polygon_model(row["vertices"], row["friction"]), cfg),
        cfg["servo_kv"])
    data = initialize(model, row["seed"])
    controller = make_controller(row, settings)
    rng = np.random.default_rng(row["seed"] + 300000)
    states, actions, errors, timing = [], [], [], []
    for _ in range(cfg["steps"]):
        observation = noisy_observation(model, data, rng, cfg["noise"])
        start = time.perf_counter()
        action = controller.action(observation, row["goal"])
        timing.append(time.perf_counter() - start)
        data.ctrl[:] = action
        mujoco.mj_step(model, data, nstep=20)
        q, _ = planar_state(data)
        error = np.r_[.15 + q[0], q[1], q[2]] - row["goal"]
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        errors.append([np.linalg.norm(error[:2]), abs(error[2]),
                       200 * (error[:2] @ error[:2]) + 2 * error[2] ** 2])
        states.append(np.r_[data.qpos, data.qvel])
        actions.append(action.copy())
    errors = np.asarray(errors)
    ok = (errors[:, 0] < .025) & (errors[:, 1] < .15)
    np.savez_compressed(out / (row["name"] + ".npz"), states=states,
                        actions=actions, errors=errors, timing=timing)
    (out / (row["name"] + "_plans.json")).write_text(json.dumps(controller.plan_records))
    return dict(**row, success=bool(np.all(ok[-10:])),
                cost=float(.05 * errors[:, 2].sum()),
                dwell_s=next(((i + 1) * .05 for i in range(9, len(ok))
                              if np.all(ok[i - 9:i + 1])), 20.),
                rollout_steps=controller.rollout_steps,
                plans=controller.plans,
                compute_s=float(np.sum(timing)))

