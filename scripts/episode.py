"""Frozen independent validation of progress-guarded TAIV-MPC."""
import itertools
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_platform.intention_fidelity import BudgetIntentionCapture
from research_platform.direct_pgtaiv_mpc import DirectPGTAIVCapture
from research_platform.pgvps_mpc import PGVPSCapture
from research_platform.known_servo import configure_servo
from research_platform.mgtaiv_mpc import MGTAIVCapture
from research_platform.pgtaiv_mpc import PGTAIVCapture
from research_platform.polygon_push import make_polygon_model
from research_platform.push_measurement import noisy_observation
from research_platform.taiv_mpc import TAIVCapture
from research_platform.tabletop_push import initialize, planar_state
from research_platform.uncertainty_physics import configure_truth


MODES = ["pgtaiv", "fixed65", "fixed95", "direct"]


def make_controller(row):
    arguments = (row["estimated_vertices"], row["seed"] + 200000,
                 row["cfg"]["servo_kv"])
    if row["mode"] == "pgtaiv":
        return PGTAIVCapture(*arguments)
    if row["mode"] == "fixed65":
        return MGTAIVCapture(*arguments)
    if row["mode"] == "fixed95":
        return TAIVCapture(*arguments, simple_faces=4, switch_ratio=.95)
    if row["mode"] == "direct_pg":
        return DirectPGTAIVCapture(*arguments)
    if row["mode"] == "pgvps":
        return PGVPSCapture(*arguments)
    if row["mode"] == "direct65":
        controller = BudgetIntentionCapture(arguments[0], arguments[1],
                                            "direct_verify", arguments[2])
        controller.core.switch_ratio = .65
        controller.core.active = False
        controller.core.update_geometry = True
        controller.core.strategy = "passive"
        return controller
    controller = BudgetIntentionCapture(arguments[0], arguments[1],
                                        "direct_verify", arguments[2])
    controller.core.active = False
    controller.core.update_geometry = True
    controller.core.strategy = "passive"
    return controller


def run(task):
    out, row = task
    cfg = row["cfg"]
    model = configure_servo(
        configure_truth(make_polygon_model(row["vertices"], row["friction"]), cfg),
        cfg["servo_kv"])
    data = initialize(model, row["seed"])
    controller = make_controller(row)
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
                physics_steps=controller.physics_steps,
                feedback_steps=controller.rollout_steps,
                plans=controller.plans,
                applied_updates=sum(record["applied"]
                                    for record in controller.calibration_records),
                relaxed_plans=sum(record.get("progress_relaxed", False)
                                  for record in controller.plan_records))

