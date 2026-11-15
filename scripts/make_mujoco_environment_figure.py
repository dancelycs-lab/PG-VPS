"""Render the actual confirmation plant; annotations never enter physics.

Run with the control environment. MUJOCO_GL defaults to EGL for SSH rendering.
The fixed instance is selected by identifier, independently of its outcome.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_platform.polygon_push import make_polygon_model
from research_platform.known_servo import configure_servo
from research_platform.uncertainty_physics import configure_truth
from research_platform.tabletop_push import initialize


def outline(scene, vertices, height, color, dashed=False):
    """Add nonphysical contour segments to the renderer's scene only."""
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        intervals = [(0., 1.)] if not dashed else [(t, t + .075) for t in np.arange(0., 1., .125)]
        for lo, hi in intervals:
            geom = scene.geoms[scene.ngeom]
            scene.ngeom += 1
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
                               np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                               np.asarray(color, dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0009,
                                 np.r_[a + lo * (b-a), height],
                                 np.r_[a + hi * (b-a), height])


def main():
    protocol = json.loads((ROOT / "results/pgvps_independent_confirmatory/protocol.json").read_text())
    name = "irregular5_19880_0.1_0.25_pgvps"
    row = next(r for r in protocol["tasks"] if r["name"] == name)
    model = configure_servo(configure_truth(
        make_polygon_model(row["vertices"], row["friction"]), row["cfg"]),
        row["cfg"]["servo_kv"])
    data = initialize(model, row["seed"])
    mujoco.mj_forward(model, data)
    # Only render-buffer resolution is changed; dynamics and geometry are intact.
    model.vis.global_.offwidth = 1600
    model.vis.global_.offheight = 1200
    goal = np.asarray(row["goal"])
    c, s = np.cos(goal[2]), np.sin(goal[2])
    target = np.asarray(row["vertices"]) @ np.array([[c, s], [-s, c]]) + goal[:2]
    source = ROOT / "results/pgvps_independent_confirmatory"
    modes = ["pgvps", "direct65", "direct95"]
    episodes = {m: dict(np.load(source / (name.replace("pgvps", m) + ".npz"))) for m in modes}
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [.16, .06, .0]
    cam.distance = .49
    cam.azimuth = 90
    images = []
    times = [0., 5., 10., 20.]
    with mujoco.Renderer(model, height=1000, width=1100) as renderer:
        for t in times:
            if t == 0:
                data.qpos[:], data.qvel[:] = initial_qpos, initial_qvel
            else:
                state = episodes["pgvps"]["states"][round(t / .05) - 1]
                data.qpos[:], data.qvel[:] = state[:model.nq], state[model.nq:]
            mujoco.mj_forward(model, data)
            cam.elevation = -65
            renderer.update_scene(data, camera=cam)
            outline(renderer.scene, target, .002, [0.1, .9, .65, 1])
            images.append(renderer.render().copy())

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 9, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(150/25.4, 125/25.4))
    grid = fig.add_gridspec(3, 4, left=.095, right=.985, top=.94, bottom=.09,
                           height_ratios=[1.05, 1, 1], hspace=.63, wspace=1.25)
    # The screenshot row has tighter spacing than the response panels.
    top = grid[0, :].subgridspec(1, 4, wspace=.035)
    for i, (image, t) in enumerate(zip(images, times)):
        ax = fig.add_subplot(top[0, i])
        ax.imshow(image)
        ax.set_title(f"({chr(97+i)}) $t={t:g}$ s", fontsize=9, pad=4)
        ax.axis("off")
    fig.text(.54, .66, 'PG-VPS replay: orange prism = object; blue sphere = pusher; green outline = target',
             ha='center', fontsize=8)
    axes = [fig.add_subplot(grid[r, c:c+2]) for r in [1, 2] for c in [0, 2]]
    colors = ['#0072B2', '#D55E00', '#009E73']
    styles = ['-', '--', '-.']
    labels = ['PG-VPS', 'Fixed 0.65', 'Fixed 0.95']
    for m, color, style, label in zip(modes, colors, styles, labels):
        ep = episodes[m]
        t = .05 * np.arange(1, len(ep['errors']) + 1)
        axes[0].plot(t, 1000 * ep['errors'][:, 0], style, color=color, lw=1.1, label=label)
        axes[1].plot(t, ep['errors'][:, 1], style, color=color, lw=1.1)
        for j in range(2):
            axes[2+j].step(np.r_[t-.05, t[-1]], np.r_[ep['actions'][:, j], ep['actions'][-1,j]],
                           where='post', ls=style, color=color, lw=.9)
    for ax, limit in zip(axes[:2], [25., .15]):
        ax.axhline(limit, color='0.4', ls=':', lw=.8)
        ax.set_ylim(bottom=0)
    for ax in axes[2:]:
        ax.set_ylim(-.165, .165)
        ax.set_yticks([-.15, 0, .15])
        ax.axhline(0, color='.7', lw=.5)
    for ax, ylabel, title in zip(axes,
            ['Position error (mm)', 'Yaw error (rad)', '$u_x$ (m/s)', '$u_y$ (m/s)'],
            ['(e) Position regulation', '(f) Orientation regulation', '(g) Pusher input: x', '(h) Pusher input: y']):
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=8, labelpad=1)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.tick_params(labelsize=8, pad=2)
        ax.spines[['top', 'right']].set_visible(False)
        for t in times[1:-1]:
            ax.axvline(t, color='.85', lw=.6, zorder=0)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', ncol=1,
                   frameon=False, fontsize=7, handlelength=2.4,
                   borderaxespad=.5, labelspacing=.3)
    from scripts.legend_layout import reserve_legend_space
    reserve_legend_space(fig)
    output = ROOT / "figures/fig_mujoco_environment.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches='tight', metadata={"Title": "MuJoCo closed-loop process: " + name,
                                         "Subject": "Recorded states and inputs; target is a rendering-only overlay."})
    plt.close(fig)
    print(output)
    print("Source:", name, "Snapshot times relative to controller start:", times)


if __name__ == "__main__":
    main()
