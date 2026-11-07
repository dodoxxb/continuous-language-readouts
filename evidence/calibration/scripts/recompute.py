"""CPU falsifiability audit of saved decoder perturbations; no new GPU run.

1. Conditional AUROC compares positive/negative pairs only within the same
   corruption phase and pre-existing relative-RMS magnitude.
2. The untuned geometric threshold ||Delta|| < rho is checked directly for
   observed argmax flips. rho is the saved two-rival first-order proxy, not the
   manuscript's ball-localized Lipschitz certificate.

Every checkpoint/direction and all 264 phase/scale cells are retained. Paired
bootstrap resamples complete sequences, conditional on the fixed checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def auc_parts(rows, key, sign=1):
    ordered = sorted((sign * row[key], int(row["flipped"])) for row in rows)
    p = sum(y for _, y in ordered)
    n = len(ordered) - p
    if not p or not n:
        return 0.0, 0
    i, rank = 0, 0.0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank += (i + j + 1) * .5 * sum(y for _, y in ordered[i:j])
        i = j
    return rank - p * (p + 1) / 2, p * n


def conditional_scores(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["t"], row["relative_rms"]].append(row)
    scores = {}
    denominators = []
    for name, key, sign in [("raw", "raw_margin", -1),
                            ("amplitude", "amplitude_margin", 1),
                            ("radius", "normalized_failure_score", 1)]:
        parts = [auc_parts(block, key, sign) for block in groups.values()]
        denominator = sum(d for _, d in parts)
        scores[name] = sum(n for n, _ in parts) / denominator if denominator else None
        denominators.append(denominator)
    assert len(set(denominators)) == 1
    return scores, denominators[0], sum(auc_parts(block, "normalized_failure_score")[1] > 0 for block in groups.values())


def quantile(values, p):
    values = sorted(values)
    t = (len(values) - 1) * p
    i = int(t)
    return values[i] + (values[min(i + 1, len(values)-1)] - values[i]) * (t-i)


def dump_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    portable_csv = Path(__file__).resolve().parent.parent / "data/heldout_perturbations.csv"
    parser.add_argument("--input-csv", type=Path, default=portable_csv if portable_csv.is_file() else None,
                        help="Portable de-texted held-out CSV, instead of repository JSON files")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--replicas", type=int, default=2000)
    args = parser.parse_args()
    if args.input_csv is None and args.source is None:
        parser.error("provide --input-csv or --source; packaged data/heldout_perturbations.csv was not found")
    out = args.output_dir or (Path(__file__).resolve().parent.parent / "results" if args.input_csv else Path(__file__).resolve().parent)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261011)
    models = ["released"]+[f"s{s}_prob{p}" for p in ["0p2","0p5","0p8","1p0"] for s in [42,43]]+["s42_prob0p8_e5","s43_prob0p8_e5"]
    cells, strata, examples, provenance = [], [], [], {}
    csv_records = None
    if args.input_csv:
        csv_records = list(csv.DictReader(args.input_csv.open()))
        numeric = ["t","relative_rms","raw_margin","perturbation_norm","first_order_radius",
                   "normalized_failure_score","predicted_margin","observed_margin"]
        for row in csv_records:
            for key in numeric:
                row[key] = float(row[key])
            for key in ["sequence","position"]:
                row[key] = int(row[key])
            row["flipped"] = row["flipped"].lower() == "true"
            floor = row.get("rms_floor_applied", "")
            row["rms_floor_applied"] = None if not floor else floor.lower() == "true"
            assert row["split"] == "heldout"
        provenance["data/" + args.input_csv.name] = hashlib.sha256(args.input_csv.read_bytes()).hexdigest()
    for model in models:
        if csv_records is not None:
            records = [dict(r) for r in csv_records if r["model"] == model]
        else:
            path = args.source / model / "sensitivity_rows.json"
            provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            records = [r for r in json.loads(path.read_text()) if r["split"] == "heldout"]
        for r in records:
            r["amplitude_margin"] = r["perturbation_norm"] / max(r["raw_margin"], 1e-12)
        for direction in ["gradient", "random"]:
            rows = [r for r in records if r["direction"] == direction]
            scores, pairs, defined = conditional_scores(rows)
            sequence_ids = sorted(set(r["sequence"] for r in rows))
            below = [r for r in rows if r["normalized_failure_score"] < 1]
            violations = [r for r in below if r["flipped"]]
            # A flip is independently visible in the saved signed original-winner margin.
            assert all(r["observed_margin"] < 0 for r in violations)
            # Below the proxy radius, tested rival linearizations remain positive.
            assert all(r["predicted_margin"] > 0 for r in violations)
            cell = {"model": model, "direction": direction, "n": len(rows),
                    "sequence_clusters": len(sequence_ids), "flips": sum(r["flipped"] for r in rows),
                    "defined_phase_scale_cells": defined, "all_phase_scale_cells": 12,
                    "discordant_label_pairs_within_cells": pairs,
                    "conditional_auc_raw": scores["raw"], "conditional_auc_amplitude": scores["amplitude"],
                    "conditional_auc_radius": scores["radius"],
                    "radius_minus_amplitude": scores["radius"]-scores["amplitude"] if pairs else None,
                    "below_proxy_radius_rows": len(below), "below_proxy_radius_flips": len(violations),
                    "below_proxy_radius_sequence_clusters": len(set(r["sequence"] for r in below)),
                    "violating_sequence_clusters": len(set(r["sequence"] for r in violations)),
                    "minimum_violating_normalized_score": min((r["normalized_failure_score"] for r in violations), default=None)}
            differences = []
            if pairs:
                blocks = [[r for r in rows if r["sequence"] == sid] for sid in sequence_ids]
                for _ in range(args.replicas):
                    resampled = [r for _ in blocks for r in blocks[rng.randrange(len(blocks))]]
                    boot, denominator, _ = conditional_scores(resampled)
                    if denominator:
                        differences.append(boot["radius"]-boot["amplitude"])
            cell["paired_bootstrap_defined"] = len(differences)
            cell["paired_difference_ci_low"] = quantile(differences, .025) if differences else None
            cell["paired_difference_ci_high"] = quantile(differences, .975) if differences else None
            cells.append(cell)
            for t in [1., .8, .6]:
                for magnitude in [.01, .05, .2, .5]:
                    rr = [r for r in rows if r["t"] == t and r["relative_rms"] == magnitude]
                    s, den, _ = conditional_scores(rr)
                    small = [r for r in rr if r["normalized_failure_score"] < 1]
                    strata.append({"model": model, "direction": direction, "t": t, "relative_rms": magnitude,
                                   "n": len(rr), "flips": sum(r["flipped"] for r in rr), "discordant_label_pairs": den,
                                   "auc_raw": s["raw"], "auc_amplitude": s["amplitude"], "auc_radius": s["radius"],
                                   "below_proxy_radius_rows": len(small), "below_proxy_radius_flips": sum(r["flipped"] for r in small)})
            for row in violations:
                examples.append({"model": model, "direction": direction, **{key: row.get(key) for key in [
                    "sequence", "position", "t", "relative_rms", "raw_margin", "perturbation_norm", "first_order_radius",
                    "normalized_failure_score", "predicted_margin", "observed_margin", "rms_floor_applied"]}})
            print(model, direction, scores, f"below-radius={len(violations)}/{len(below)}", flush=True)
    assert len(cells) == 22 and len(strata) == 264
    summary = {}
    for direction in ["gradient", "random"]:
        cc = [r for r in cells if r["direction"] == direction]
        defined = [r for r in cc if r["radius_minus_amplitude"] is not None]
        summary[direction] = {"testable_checkpoints": len(defined),
            "radius_wins": sum(r["radius_minus_amplitude"] > 1e-12 for r in defined),
            "ties": sum(abs(r["radius_minus_amplitude"]) <= 1e-12 for r in defined),
            "radius_losses": sum(r["radius_minus_amplitude"] < -1e-12 for r in defined),
            "paired_intervals_include_zero": sum(r["paired_difference_ci_low"] <= 0 <= r["paired_difference_ci_high"] for r in defined),
            "defined_strata": sum(r["defined_phase_scale_cells"] for r in cc), "total_strata": 132,
            "below_proxy_radius_rows": sum(r["below_proxy_radius_rows"] for r in cc),
            "below_proxy_radius_flips": sum(r["below_proxy_radius_flips"] for r in cc),
            "checkpoint_conditions_with_violations": sum(r["below_proxy_radius_flips"] > 0 for r in cc),
            "sum_of_per_checkpoint_violating_sequence_clusters": sum(r["violating_sequence_clusters"] for r in cc)}
    report = {"protocol": {"analysis": "post hoc CPU analysis of saved held-out perturbations",
         "conditional_auc": "sum of correctly ordered positive-negative pairs (ties one half), divided by sum of pairs, only within identical (t, relative_rms) cells",
         "threshold": "normalized_failure_score < 1, fixed from the first-order radius formula, not tuned",
         "bootstrap": f"{args.replicas} paired sequence-cluster percentile replicates; Python random seed 20261011; fixed checkpoint, unadjusted intervals",
         "missing_strata": "Single-class cells contribute zero discordant pairs and are listed explicitly, not imputed with AUROC.",
         "limits": ["Threshold violations do not contradict a ball-localized Lipschitz theorem: the proxy measures neither uniform neighborhood constants nor all vocabulary rivals.",
             "Gradient direction is constructed using the same local geometry; conditional AUROC does not eliminate this intentional design coupling.",
             "Cross-checkpoint sums are descriptive counts, not independent trial counts or a calibrated risk estimate.",
             "Same generated sample IDs across retrained conditions may be related; sequence-cluster sums across models do not count unique independent documents.",
             "Conditioning fixes relative RMS, not absolute perturbation norm; the amplitude-aware comparator accounts for the latter."]},
         "summary": summary, "checkpoint_direction_cells": cells, "source_sha256": provenance,
         "all_phase_scale_cells_file": "calibration_phase_scale_cells_20261011.csv",
         "all_threshold_violations_file": "calibration_proxy_radius_violations_20261011.csv"}
    (out/"calibration_falsifiability_20261011.json").write_text(json.dumps(report, indent=2)+"\n")
    dump_csv(out/"calibration_conditional_auc_20261011.csv", cells)
    dump_csv(out/"calibration_phase_scale_cells_20261011.csv", strata)
    dump_csv(out/"calibration_proxy_radius_violations_20261011.csv", sorted(examples, key=lambda r:r["normalized_failure_score"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
