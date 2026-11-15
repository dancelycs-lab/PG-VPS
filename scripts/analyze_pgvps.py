"""Primary, public-subset, sensitivity, and decision analysis for PG-VPS."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pgvps_analysis"


def condition_key(row):
    return row["seed"], row["shape"], row["friction"], row["mass"]


def cluster_effect(values, rng):
    values = np.asarray(values, dtype=float)
    boot = values[rng.integers(0, len(values), (50000, len(values)))].mean(axis=1)
    signs = np.asarray(list(itertools.product([-1., 1.], repeat=len(values))))
    null = np.mean(signs * values, axis=1)
    return dict(
        difference=float(values.mean()),
        cluster95=np.quantile(boot, [.025, .975]).tolist(),
        sign_flip_p=float(np.mean(np.abs(null) >= abs(values.mean()) - 1e-15)),
        seed_cluster_values=values.tolist())


def summarize(rows):
    return dict(
        n=len(rows), success=int(sum(row["success"] for row in rows)),
        success_rate=float(np.mean([row["success"] for row in rows])),
        cost=float(np.mean([row["cost"] for row in rows])),
        dwell_s=float(np.mean([row["dwell_s"] for row in rows])),
        physics_steps=float(np.mean([row.get("physics_steps",
                                              20 * row.get("rollout_steps", 0))
                                     for row in rows])))


def seed_contrast(mapping, conditions, left, right, field, rng):
    seeds = sorted({condition[0] for condition in conditions})
    values = [np.mean([float(mapping[left][condition][field]) -
                       float(mapping[right][condition][field])
                       for condition in conditions if condition[0] == seed])
              for seed in seeds]
    return cluster_effect(values, rng)


def primary_analysis(rng):
    base = json.loads((ROOT / "results/pgtaiv_extended_confirmatory/summary.json").read_text())
    factorial = json.loads((ROOT / "results/pgtaiv_factorial_confirmatory/summary.json").read_text())
    strict = json.loads((ROOT / "results/pgvps_strict_confirmatory/summary.json").read_text())
    rows = base + factorial + strict
    remap = {"direct_pg": "pgvps", "direct": "direct95"}
    modes = ["pgvps", "direct65", "direct95", "pgtaiv", "fixed65", "fixed95"]
    mapping = {mode: {} for mode in modes}
    for row in rows:
        mode = remap.get(row["mode"], row["mode"])
        if mode in mapping:
            mapping[mode][condition_key(row)] = row
    conditions = sorted(mapping["pgvps"])
    assert len(conditions) == 288
    for mode in modes:
        assert set(mapping[mode]) == set(conditions)
    summary = {mode: summarize([mapping[mode][condition]
                                for condition in conditions])
               for mode in modes}
    contrasts = {}
    for name, baseline in [
            ("pgvps-direct65", "direct65"),
            ("pgvps-direct95", "direct95"),
            ("pgvps-pgtaiv", "pgtaiv")]:
        contrasts[name] = {field: seed_contrast(
            mapping, conditions, "pgvps", baseline, field, rng)
            for field in ["success", "cost", "dwell_s", "physics_steps"]}
    return dict(summary=summary, contrasts=contrasts), mapping, conditions


def holm_adjust(pvalues):
    """Holm-adjust p values while preserving the input mapping."""
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted = {}
    running = 0.
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (count - rank) * pvalues[name])
        adjusted[name] = min(1., running)
    return adjusted


def independent_analysis(rng):
    rows = json.loads((ROOT / "results/pgvps_independent_confirmatory/summary.json").read_text())
    modes = ["pgvps", "direct65", "direct95"]
    mapping = {mode: {condition_key(row): row for row in rows
                      if row["mode"] == mode} for mode in modes}
    conditions = sorted(mapping["pgvps"])
    assert len(conditions) == 288
    for mode in modes:
        assert set(mapping[mode]) == set(conditions)
    summary = {mode: summarize([mapping[mode][condition]
                                for condition in conditions])
               for mode in modes}
    contrasts = {}
    for baseline in ["direct65", "direct95"]:
        name = "pgvps-" + baseline
        contrasts[name] = {field: seed_contrast(
            mapping, conditions, "pgvps", baseline, field, rng)
            for field in ["success", "cost", "dwell_s", "physics_steps"]}
    raw = {name: result["success"]["sign_flip_p"]
           for name, result in contrasts.items()}
    adjusted = holm_adjust(raw)
    for name in contrasts:
        contrasts[name]["success"]["holm_adjusted_p"] = adjusted[name]
    return dict(
        status=("Prospectively frozen after method selection and before simulation; "
                "all seeds are disjoint from earlier cohorts."),
        summary=summary, contrasts=contrasts)


def decision_audit():
    rows = json.loads((ROOT / "results/pgtaiv_factorial_confirmatory/summary.json").read_text())
    selected = [row for row in rows if row["mode"] == "direct_pg"]
    records = []
    for row in selected:
        path = ROOT / "results/pgtaiv_factorial_confirmatory" / (row["name"] + "_plans.json")
        for record in json.loads(path.read_text()):
            if not record["incumbent"]:
                records.append(dict(kind="initial"))
                continue
            proposal = int(record["proposed_index"])
            if proposal == 0:
                records.append(dict(kind="coarse-retain"))
                continue
            fine = {int(index): float(value)
                    for index, value in record["fine_costs"].items()}
            rho = float(record["switch_ratio"])
            c0 = float(record["costs"][0])
            cb = float(record["costs"][proposal])
            j0, jb = fine[0], fine[proposal]
            accepted = int(record["selected_index"]) != 0
            records.append(dict(
                kind="accepted" if accepted else "rejected",
                relaxed=bool(record["progress_relaxed"]),
                ratio_error=abs(cb / c0 - jb / j0),
                fine_slack=rho * j0 - jb,
                rank_reversal=bool((cb - c0) * (jb - j0) < 0.)))
    proposed = [record for record in records
                if record["kind"] in {"accepted", "rejected"}]
    accepted = [record for record in proposed if record["kind"] == "accepted"]
    rejected = [record for record in proposed if record["kind"] == "rejected"]
    errors = np.asarray([record["ratio_error"] for record in proposed])
    return dict(
        episodes=len(selected), total_plans=len(records),
        initial_plans=sum(record["kind"] == "initial" for record in records),
        coarse_retentions=sum(record["kind"] == "coarse-retain" for record in records),
        proposed_switches=len(proposed), accepted_switches=len(accepted),
        rejected_by_fine_verification=len(rejected),
        rejection_fraction=float(len(rejected) / len(proposed)),
        rank_reversals=sum(record["rank_reversal"] for record in proposed),
        relaxed_proposals=sum(record["relaxed"] for record in proposed),
        relaxed_acceptances=sum(record["relaxed"] for record in accepted),
        strict_acceptances=sum(not record["relaxed"] for record in accepted),
        minimum_accepted_fine_slack=float(min(record["fine_slack"]
                                              for record in accepted)),
        maximum_rejected_fine_slack=float(max(record["fine_slack"]
                                              for record in rejected)),
        coarse_fine_ratio_error=dict(
            median=float(np.median(errors)), q95=float(np.quantile(errors, .95)),
            maximum=float(np.max(errors))))


def public_analysis(rng):
    pgvps = json.loads((ROOT / "results/pgvps_public_subset/summary.json").read_text())
    public = json.loads((ROOT / "results/pgtaiv_public_baselines/summary.json").read_text())
    summary = {"pgvps": summarize(pgvps)}
    for mode in ["ps", "mppi", "icem", "cem", "smppi"]:
        summary[mode] = summarize([row for row in public if row["mode"] == mode])
    mapping = {mode: {condition_key(row): row for row in public if row["mode"] == mode}
               for mode in ["ps", "mppi", "icem", "cem", "smppi"]}
    mapping["pgvps"] = {condition_key(row): row for row in pgvps}
    conditions = sorted(mapping["pgvps"])
    contrasts = {}
    for baseline in ["icem", "ps", "mppi", "cem", "smppi"]:
        contrasts["pgvps-" + baseline] = {
            field: seed_contrast(mapping, conditions, "pgvps", baseline,
                                 field, rng)
            for field in ["success", "cost", "dwell_s"]}
    return dict(summary=summary, contrasts=contrasts)


def sensitivity_analysis(rng):
    rows = json.loads((ROOT / "results/pgvps_sensitivity/summary.json").read_text())
    settings = sorted({row["setting"] for row in rows})
    summary = {setting: summarize([row for row in rows
                                   if row["setting"] == setting])
               for setting in settings}
    mapping = {setting: {condition_key(row): row for row in rows
                         if row["setting"] == setting}
               for setting in settings}
    conditions = sorted(mapping["default"])
    contrasts = {}
    for setting in settings:
        if setting == "default":
            continue
        contrasts["default-" + setting] = {
            field: seed_contrast(mapping, conditions, "default", setting,
                                 field, rng)
            for field in ["success", "cost", "dwell_s", "physics_steps"]}
    return dict(summary=summary, contrasts=contrasts)


def export_tables(result):
    """Write readable numerical summaries without manuscript-specific numbering."""
    fields = ["controller", "n", "success", "success_rate", "cost", "dwell_s", "physics_steps"]
    labels = {"pgvps": "PG-VPS", "direct65": "Fixed 0.65", "direct95": "Fixed 0.95",
              "pgtaiv": "Expanded guarded", "fixed65": "Expanded fixed 0.65",
              "fixed95": "Expanded fixed 0.95", "ps": "PS", "mppi": "MPPI",
              "icem": "iCEM", "cem": "CEM", "smppi": "Smooth-MPPI"}
    for study in ["independent_confirmation", "public", "development_factorial", "sensitivity"]:
        with (OUT / (study + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for name, values in result[study]["summary"].items():
                writer.writerow({"controller": labels.get(name, name), **values})
    with (OUT / "decision_audit.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["quantity", "value"])
        for name, value in result["decision_audit"].items():
            if isinstance(value, (int, float, str)):
                writer.writerow([name, value])


def main():
    rng = np.random.default_rng(20261105)
    primary, _, _ = primary_analysis(rng)
    result = dict(
        analysis_status=("Independent evaluation, factorial ablation, public comparisons, and "
                         "parameter sensitivity. Randomized realizations are the "
                         "independent resampling and sign-flip units."),
        independent_confirmation=independent_analysis(rng),
        development_factorial=primary,
        decision_audit=decision_audit(),
        public=public_analysis(rng),
        sensitivity=sensitivity_analysis(rng))
    OUT.mkdir(exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2))
    export_tables(result)
    print(f"Analysis and CSV tables written to {OUT}")


if __name__ == "__main__":
    main()
