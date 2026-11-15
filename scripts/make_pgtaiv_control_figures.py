"""Generate comparison-first PG-VPS figures in a classical control-paper style.

All figures use frozen trajectories.  Representative cases are selected by rules
that are fixed in code and recorded in the output manifest; no trajectory is
selected by visual inspection.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/pgvps_independent_confirmatory"
OUT = ROOT / "figures"
DT = 0.05
POS_TOL = 0.025
YAW_TOL = 0.15
METHODS = ("pgvps", "direct65", "direct95")
LABEL = {"pgvps": "PG-VPS", "direct65": "Fixed 0.65",
         "direct95": "Fixed 0.95"}
COLOR = {"pgvps": "#0072B2", "direct65": "#D55E00", "direct95": "#009E73"}
STYLE = {"pgvps": "-", "direct65": "--", "direct95": "-."}
MARKER = {"pgvps": "o", "direct65": "s", "direct95": "^"}


def load_rows():
    return json.loads((DATA / "summary.json").read_text())


def row_index(rows):
    return {(r["shape"], r["seed"], r["friction"], r["mass"], r["mode"]): r
            for r in rows}


def condition_rows(index, key):
    return {m: index[(*key, m)] for m in METHODS}


def select_cases(rows):
    """Return deterministic common-success, joint-recovery, and guard cases."""
    index = row_index(rows)
    keys = sorted({(r["shape"], r["seed"], r["friction"], r["mass"])
                   for r in rows})
    recovery = next(k for k in keys
                    if index[(*k, "pgvps")]["success"]
                    and not index[(*k, "direct65")]["success"]
                    and not index[(*k, "direct95")]["success"])
    guard = next(k for k in keys
                 if index[(*k, "pgvps")]["success"]
                 and index[(*k, "pgvps")]["relaxed_plans"] > 0
                 and not index[(*k, "direct65")]["success"]
                 and not index[(*k, "direct95")]["success"])
    eligible = []
    for key in keys:
        group = condition_rows(index, key)
        if (all(group[m]["success"] for m in METHODS)
                and group["pgvps"]["cost"] < group["direct95"]["cost"]
                and group["pgvps"]["dwell_s"] < group["direct95"]["dwell_s"]):
            eligible.append((group["direct95"]["cost"] - group["pgvps"]["cost"], key))
    eligible.sort()
    nominal = eligible[(len(eligible) - 1) // 2][1]
    return nominal, recovery, guard, {
        "nominal": ("median fixed-0.95-minus-PG-VPS task-cost improvement among the "
                    "complete common-success set for which PG-VPS also completes earlier"),
        "recovery": ("first lexicographically sorted condition in the frozen cohort "
                     "where PG-VPS succeeds and both fixed-margin rules fail"),
        "guard": ("first lexicographically sorted condition in the frozen cohort where "
                  "PG-VPS activates the relaxed margin and succeeds while both fixed "
                  "margins fail"),
    }


def npz_path(row):
    return DATA / f"{row['name']}.npz"


def plan_path(row):
    return DATA / f"{row['name']}_plans.json"


def load_episode(row):
    with np.load(npz_path(row)) as z:
        return {k: np.array(z[k]) for k in z.files}


def yaw_from_quaternion(q):
    # MuJoCo free-joint quaternion order is w, x, y, z.
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def time_axis(n=400):
    return DT * np.arange(1, n + 1)


def panel_label(ax, text):
    ax.text(-0.13, 1.04, text, transform=ax.transAxes, fontweight="bold")


def method_plot(ax, x, y, mode, **kwargs):
    return ax.plot(x, y, color=COLOR[mode], linestyle=STYLE[mode],
                   linewidth=1.25, label=LABEL[mode], **kwargs)


def save(fig, stem, provenance, width_mm=190):
    from scripts.legend_layout import reserve_legend_space, annotate_results
    reserve_legend_space(fig)
    annotate_results(fig, stem)
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return {"pdf": str(pdf.relative_to(ROOT)), "width_mm": width_mm,
            "provenance": provenance}


def trajectory_figure(rows, nominal, recovery, rules):
    index = row_index(rows)
    fig, axes = plt.subplots(1, 2, figsize=(190 / 25.4, 77 / 25.4),
                             layout="constrained")
    for j, (key, title) in enumerate(((nominal, "Common-success transient"),
                                      (recovery, "Joint-recovery transient"))):
        group = condition_rows(index, key)
        goal = np.asarray(group["pgvps"]["goal"])
        ax = axes[j]
        for mode in METHODS:
            ep = load_episode(group[mode])
            xy = ep["states"][:, :2]
            ax.plot(xy[:, 0], xy[:, 1], color=COLOR[mode], linestyle=STYLE[mode],
                    linewidth=1.35, label=LABEL[mode])
            ax.scatter(xy[-1, 0], xy[-1, 1], s=22, marker=MARKER[mode],
                       facecolor="white", edgecolor=COLOR[mode], linewidth=.9, zorder=4)
        start = load_episode(group["pgvps"])["states"][0, :2]
        ax.scatter(start[0], start[1], s=30, color="black", marker="o", zorder=5,
                   label="Initial pose")
        ax.scatter(goal[0], goal[1], s=65, color="#7A3E9D", marker="*", zorder=5,
                   label="Target pose")
        circle = plt.Circle(goal[:2], POS_TOL, fill=False, color="#7A3E9D",
                            linewidth=.8, linestyle=":")
        ax.add_patch(circle)
        ax.set(xlabel="$x$ position (m)", ylabel="$y$ position (m)", title=title)
        ax.set_aspect("equal", adjustable="box")
        panel_label(ax, "ab"[j])
    axes[0].legend(frameon=False, ncol=2, loc="best")
    return save(fig, "fig_control_trajectories", {
        "source": "results/pgvps_independent_confirmatory/*.npz and summary.json",
        "selection_rules": rules,
        "nominal_key": nominal,
        "recovery_key": recovery,
        "transform": "object-center x-y samples at every 50-ms control update; no smoothing",
        "missing": "none",
    })


def response_figure(rows, nominal, recovery, rules):
    index = row_index(rows)
    fig, axes = plt.subplots(3, 2, figsize=(190 / 25.4, 142 / 25.4),
                             sharex=True, sharey="row", layout="constrained")
    for col, (key, title) in enumerate(((nominal, "Common-success transient"),
                                        (recovery, "Joint-recovery transient"))):
        group = condition_rows(index, key)
        for mode in METHODS:
            errors = load_episode(group[mode])["errors"]
            t = time_axis(len(errors))
            method_plot(axes[0, col], t, 1000 * errors[:, 0], mode)
            method_plot(axes[1, col], t, errors[:, 1], mode)
            method_plot(axes[2, col], t, errors[:, 2], mode)
        axes[0, col].axhline(1000 * POS_TOL, color="#555555", linestyle=":", linewidth=.8)
        axes[1, col].axhline(YAW_TOL, color="#555555", linestyle=":", linewidth=.8)
        axes[0, col].set_title(title)
        axes[2, col].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Position error (mm)")
    axes[1, 0].set_ylabel("Yaw error (rad)")
    axes[2, 0].set_ylabel("Task energy $E_t$")
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    for i, ax in enumerate(axes.flat):
        panel_label(ax, "abcdef"[i])
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper right")
    
    
    return save(fig, "fig_control_responses", {
        "source": "results/pgvps_independent_confirmatory/*.npz and summary.json",
        "selection_rules": rules,
        "nominal_key": nominal,
        "recovery_key": recovery,
        "transform": "saved closed-loop errors at every 50-ms update; no smoothing",
        "thresholds": {"position_m": POS_TOL, "yaw_rad": YAW_TOL},
        "missing": "none",
    })


def held_face(plans, n=400):
    steps = np.array([p["step"] for p in plans], dtype=int)
    faces = np.array([p["selected_face"] for p in plans], dtype=float)
    out = np.full(n, np.nan)
    for i, step in enumerate(steps):
        end = steps[i + 1] if i + 1 < len(steps) else n
        out[step:end] = faces[i]
    return out


def input_switch_figure(rows, guard, rules):
    index = row_index(rows)
    group = condition_rows(index, guard)
    fig, axes = plt.subplots(4, 1, figsize=(190 / 25.4, 150 / 25.4), sharex=True,
                             layout="constrained")
    for mode in METHODS:
        ep = load_episode(group[mode])
        t = time_axis(len(ep["actions"]))
        method_plot(axes[0], t, ep["actions"][:, 0], mode)
        method_plot(axes[1], t, ep["actions"][:, 1], mode)
        method_plot(axes[2], t, ep["errors"][:, 2], mode)
        plans = json.loads(plan_path(group[mode]).read_text())
        method_plot(axes[3], t, held_face(plans, len(t)), mode, drawstyle="steps-post")
        if mode == "pgvps":
            relaxed = [p["step"] for p in plans if p.get("progress_relaxed", False)]
            switched = [p["step"] for p in plans
                        if p["step"] > 0 and p.get("verified", False)
                        and not p.get("incumbent", False)]
            axes[2].scatter(DT * (np.asarray(relaxed) + 1),
                            ep["errors"][relaxed, 2], marker="v", s=21,
                            facecolor="white", edgecolor="#7A3E9D", linewidth=.8,
                            zorder=5, label="PG relaxed replan")
            axes[3].scatter(DT * (np.asarray(switched) + 1),
                            held_face(plans, len(t))[switched], marker="o", s=18,
                            facecolor="white", edgecolor="#7A3E9D", linewidth=.8,
                            zorder=5, label="PG accepted replacement")
    axes[0].set_ylabel("$u_x$ (m/s)")
    axes[1].set_ylabel("$u_y$ (m/s)")
    axes[2].set_ylabel("Task energy $E_t$")
    axes[3].set_ylabel("Selected face")
    axes[3].set_xlabel("Time (s)")
    axes[0].set_ylim(-.165, .165)
    axes[1].set_ylim(-.165, .165)
    axes[0].axhline(.15, color="#666666", linestyle=":", linewidth=.7)
    axes[0].axhline(-.15, color="#666666", linestyle=":", linewidth=.7)
    axes[1].axhline(.15, color="#666666", linestyle=":", linewidth=.7)
    axes[1].axhline(-.15, color="#666666", linestyle=":", linewidth=.7)
    for i, ax in enumerate(axes):
        panel_label(ax, "abcd"[i])
    axes[0].legend(frameon=False, ncol=3, loc="lower left", fontsize=6.5)
    axes[2].legend(frameon=False, loc="upper right")
    axes[3].legend(frameon=False, loc="upper left")
    return save(fig, "fig_control_inputs_switching", {
        "source": "results/pgvps_independent_confirmatory guard-case npz and plan logs",
        "selection_rule": rules["guard"],
        "guard_key": guard,
        "transform": "raw applied controls and errors; face index held between replans",
        "marker_definition": "relaxed PG replans and verified non-incumbent PG replacements",
        "missing": "none",
    })


def sustained_fraction(errors):
    inside = (errors[:, :, 0] <= POS_TOL) & (errors[:, :, 1] <= YAW_TOL)
    out = np.zeros_like(inside, dtype=float)
    for k in range(9, inside.shape[1]):
        out[:, k] = np.all(inside[:, k - 9:k + 1], axis=1)
    return out.mean(axis=0)


def ensemble_figure(rows):
    by_mode = {m: [] for m in METHODS}
    for row in rows:
        by_mode[row["mode"]].append(load_episode(row)["errors"])
    fig, axes = plt.subplots(3, 1, figsize=(190 / 25.4, 132 / 25.4), sharex=True,
                             layout="constrained")
    t = time_axis()
    for mode in METHODS:
        errors = np.stack(by_mode[mode])
        for ax, values, scale in ((axes[0], errors[:, :, 0], 1000),
                                  (axes[1], errors[:, :, 1], 1)):
            q25, med, q75 = np.quantile(values, [.25, .5, .75], axis=0)
            ax.fill_between(t, scale * q25, scale * q75, color=COLOR[mode], alpha=.10,
                            linewidth=0)
            method_plot(ax, t, scale * med, mode)
        method_plot(axes[2], t, 100 * sustained_fraction(errors), mode)
    axes[0].axhline(1000 * POS_TOL, color="#555555", linestyle=":", linewidth=.8)
    axes[1].axhline(YAW_TOL, color="#555555", linestyle=":", linewidth=.8)
    axes[0].set_ylabel("Position error (mm)")
    axes[1].set_ylabel("Yaw error (rad)")
    axes[2].set_ylabel("Sustained-in-tolerance (%)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(0, 102)
    for i, ax in enumerate(axes):
        panel_label(ax, "abc"[i])
    axes[0].legend(frameon=False, ncol=3, loc="lower left", fontsize=6.5)
    return save(fig, "fig_control_ensemble", {
        "source": "all 864 trajectories in results/pgvps_independent_confirmatory",
        "transform": ("pointwise median with interquartile band for error; fraction satisfying "
                      "both tolerances continuously for the preceding ten control samples"),
        "denominator": "288 paired conditions per method at every time sample",
        "missing": "none; failures retained through 20 s",
    })


def aggregate(rows, mode, field, group_field, group_values):
    out = []
    for value in group_values:
        selected = [r for r in rows if r["mode"] == mode and r[group_field] == value]
        if field == "success":
            out.append(100 * np.mean([r[field] for r in selected]))
        else:
            out.append(np.mean([r[field] for r in selected]))
    return out


def robustness_figure(rows):
    cells = ((.1, .25), (.1, .5), (.6, .25), (.6, .5))
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.0), sharex=True, sharey=True,
                             layout="constrained")
    t = time_axis()
    for panel, (ax, (friction, mass)) in enumerate(zip(axes.flat, cells)):
        for mode in METHODS:
            selected = [r for r in rows if r["mode"] == mode
                        and r["friction"] == friction and r["mass"] == mass]
            assert len(selected) == 72
            errors = np.stack([load_episode(r)["errors"] for r in selected])
            method_plot(ax, t, 100*sustained_fraction(errors), mode)
        ax.set_title(rf"$\mu={friction:g}$, $m={mass:g}$ kg")
        ax.set(xticks=[0,5,10,15,20], yticks=[0,25,50,75,100], ylim=(0,102))
        panel_label(ax, "abcd"[panel])
    for ax in axes[-1,:]: ax.set_xlabel("Time (s)")
    for ax in axes[:,0]: ax.set_ylabel("In tolerance (%)")
    axes[0,0].legend(frameon=False, ncol=3)
    return save(fig, "fig_control_robustness", {
        "source": "all 864 independent-confirmation trajectories",
        "transform": "fraction within both tolerances for ten consecutive samples at each of 400 recorded times",
        "sample_size": "72 trials per controller and friction--mass setting; all six polygon orders and twelve seeds",
        "missing": "none; all failures retained through 20 s",
    }, width_mm=178)


def public_tradeoff_figure():
    from scripts.public_diagnostics import main as public_diagnostics
    public_diagnostics(ROOT, OUT)
    return {"files": ["fig_control_public_tradeoff.pdf", "fig_public_responses.pdf", "fig_public_robustness.pdf"], "independent_units": 6, "conditions": 24}


def primary_effect_figure():
    rows = load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.45), layout="constrained")
    t = time_axis()
    endpoints = {}
    for mode in METHODS:
        selected = [r for r in rows if r["mode"] == mode]
        costs, work = [], []
        for row in selected:
            errors = load_episode(row)["errors"]
            costs.append(DT*np.cumsum(errors[:,2]))
            increments = np.zeros(len(t))
            for plan in json.loads(plan_path(row).read_text()):
                step = int(plan["step"])
                if 0 <= step < len(t): increments[step] += plan["physics_steps"]
            cumulative = np.cumsum(increments)
            assert np.isclose(cumulative[-1], row['physics_steps']), row['name']
            work.append(cumulative)
        mean_cost = np.mean(costs, axis=0)
        mean_work = np.mean(work, axis=0)
        method_plot(axes[0], t, mean_cost, mode)
        method_plot(axes[1], t, mean_work/1000, mode)
        endpoints[mode] = dict(mean_cost=float(mean_cost[-1]), mean_physics_steps=float(mean_work[-1]))
    axes[0].set_ylabel("Cumulative task cost")
    axes[1].set_ylabel(r"Prediction steps ($10^3$)")
    for i,ax in enumerate(axes):
        ax.set(xlabel="Time (s)", xlim=(0,20), ylim=(0,None), xticks=[0,5,10,15,20])
        panel_label(ax,"ab"[i])
    axes[0].legend(frameon=False, ncol=3)
    return save(fig,"fig_control_primary_effects",{
        "source": "all independent-confirmation errors and planning logs",
        "transform": "mean cumulative task energy and actual planning substeps over 288 trials per method",
        "samples_per_curve": 400,
        "endpoints": endpoints,
        "missing": "none; failures included",
    }, width_mm=188)


def main():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 7.4, "axes.labelsize": 7.4,
        "axes.titlesize": 8, "legend.fontsize": 6.8, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "axes.linewidth": .65, "lines.linewidth": 1.25,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.transparent": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#D9D9D9", "grid.linewidth": .4,
        "grid.alpha": .7,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    nominal, recovery, guard, rules = select_cases(rows)
    figures = [
        trajectory_figure(rows, nominal, recovery, rules),
        response_figure(rows, nominal, recovery, rules),
        input_switch_figure(rows, guard, rules),
        ensemble_figure(rows),
        robustness_figure(rows),
        primary_effect_figure(),
        public_tradeoff_figure(),
    ]
    manifest = {
        "target": "International Journal of Control, Automation, and Systems PG-VPS experiments",
        "figures": figures,
        "representative_case_policy": "rules fixed before rendering and applied programmatically",
        "integrity": "No trajectory exclusions, smoothing, interpolation, or missing-value imputation.",
        "accessibility": "Method color is redundant with line style and, where applicable, marker.",
    }
    manifest_path = ROOT / "results/pgvps_analysis/control_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
