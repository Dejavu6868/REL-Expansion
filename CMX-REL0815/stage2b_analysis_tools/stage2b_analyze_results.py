#!/usr/bin/env python3
"""Aggregate Stage2B three-seed results and run the declared mechanism analysis."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


SEEDS = (12345, 23456, 34567)
ARMS = ("rawdepth", "hha", "relplus_local", "relplus_pose")
METRICS = ("miou", "pixel_accuracy", "mean_pixel_accuracy")
COMPARISONS = (
    ("relplus_local_minus_hha", "relplus_local", "hha"),
    ("relplus_pose_minus_relplus_local", "relplus_pose", "relplus_local"),
)


def _run_dir(stage2a_root, stage2b_root, seed, arm):
    if seed == 12345:
        return stage2a_root / arm
    return stage2b_root / "seed_{}".format(seed) / arm


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames=None):
    if not rows:
        raise ValueError("refusing to write an empty CSV: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames if fieldnames is not None else list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def _ratio(numerator, denominator):
    return float("nan") if denominator == 0 else float(numerator) / float(denominator)


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def _sign_pattern(values):
    return "/".join("+" if value > 0 else "-" if value < 0 else "0" for value in values)


def _bh_fdr(p_values):
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _class_metrics(confusion, class_names):
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion matrix shape does not match class names")
    gt = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    rows = []
    for index, class_name in enumerate(class_names):
        rows.append(
            {
                "class": class_name,
                "true_positive_pixels": int(correct[index]),
                "ground_truth_pixels": int(gt[index]),
                "predicted_pixels": int(predicted[index]),
                "precision": _ratio(correct[index], predicted[index]),
                "recall": _ratio(correct[index], gt[index]),
                "iou": _ratio(
                    correct[index], gt[index] + predicted[index] - correct[index]
                ),
            }
        )
    return rows


def _load_per_image(path):
    result = {}
    for row in _read_csv(path):
        name = row["name"]
        if name in result:
            raise ValueError("duplicate per-image key in {}: {}".format(path, name))
        result[name] = {
            "pixel_accuracy": float(row["pixel_accuracy"]),
            "present_class_miou": float(row["present_class_miou"]),
        }
    return result


def _prediction_names(root):
    return {
        str(path.relative_to(root).with_suffix(""))
        for path in root.rglob("*.png")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2a-root", required=True, type=Path)
    parser.add_argument("--stage2b-root", required=True, type=Path)
    parser.add_argument("--gravity-csv", required=True, type=Path)
    parser.add_argument("--gravity-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=17593)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_iou = _read_csv(
        _run_dir(args.stage2a_root, args.stage2b_root, 12345, "rawdepth")
        / "metrics/per_class_iou.csv"
    )
    class_names = [row["class"] for row in first_iou]
    if len(class_names) != 13 or len(set(class_names)) != 13:
        raise ValueError("expected exactly 13 unique evaluator classes")

    aggregate = {}
    run_metric_rows = []
    per_class_rows = []
    confusion_rows = []
    per_class_lookup = {}
    sample_key_sets = []
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = _run_dir(args.stage2a_root, args.stage2b_root, seed, arm)
            metrics_path = run_dir / "metrics/final_metrics.json"
            per_image_path = run_dir / "metrics/per_image_metrics.csv"
            with metrics_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            aggregate[(seed, arm)] = payload
            for metric in METRICS:
                value = float(payload[metric])
                run_metric_rows.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "metric": metric,
                        "value_fraction": value,
                        "value_percent": 100.0 * value,
                    }
                )
            class_rows = _class_metrics(payload["confusion_matrix"], class_names)
            confusion = np.asarray(payload["confusion_matrix"], dtype=np.int64)
            for gt_index, gt_class in enumerate(class_names):
                gt_total = int(np.sum(confusion[gt_index]))
                for pred_index, pred_class in enumerate(class_names):
                    pixels = int(confusion[gt_index, pred_index])
                    confusion_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "ground_truth_class": gt_class,
                            "predicted_class": pred_class,
                            "pixels": pixels,
                            "fraction_of_ground_truth_class": _ratio(pixels, gt_total),
                        }
                    )
            for row in class_rows:
                complete = dict(row)
                complete.update({"seed": seed, "arm": arm})
                per_class_rows.append(complete)
                per_class_lookup[(seed, arm, row["class"])] = row
            image_rows = _read_csv(per_image_path)
            keys = [row["name"] for row in image_rows]
            if len(keys) != args.expected_count or len(set(keys)) != args.expected_count:
                raise ValueError("invalid per-image coverage: {}".format(per_image_path))
            sample_key_sets.append(set(keys))

    reference_keys = sample_key_sets[0]
    if any(keys != reference_keys for keys in sample_key_sets[1:]):
        raise ValueError("the 12 per-image result files do not share identical keys")
    _write_csv(args.output_dir / "run_metrics.csv", run_metric_rows)
    _write_csv(args.output_dir / "per_class_metrics.csv", per_class_rows)
    _write_csv(args.output_dir / "confusion_matrix_long.csv", confusion_rows)

    arm_summary_rows = []
    for arm in ARMS:
        for metric in METRICS:
            values = [float(aggregate[(seed, arm)][metric]) * 100.0 for seed in SEEDS]
            mean, std = _mean_std(values)
            arm_summary_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n_seeds": len(SEEDS),
                    "mean_percent": mean,
                    "std_percent": std,
                    "minimum_percent": min(values),
                    "maximum_percent": max(values),
                    "seed_12345_percent": values[0],
                    "seed_23456_percent": values[1],
                    "seed_34567_percent": values[2],
                }
            )
    _write_csv(args.output_dir / "arm_summary.csv", arm_summary_rows)

    paired_rows = []
    paired_summary_rows = []
    primary_inference = []
    for comparison, left, right in COMPARISONS:
        for metric in METRICS:
            deltas = []
            for seed in SEEDS:
                delta = 100.0 * (
                    float(aggregate[(seed, left)][metric])
                    - float(aggregate[(seed, right)][metric])
                )
                deltas.append(delta)
                paired_rows.append(
                    {
                        "comparison": comparison,
                        "left_arm": left,
                        "right_arm": right,
                        "seed": seed,
                        "metric": metric,
                        "delta_percentage_points": delta,
                    }
                )
            mean, std = _mean_std(deltas)
            paired_summary_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "n_paired_seeds": len(SEEDS),
                    "mean_delta_percentage_points": mean,
                    "std_delta_percentage_points": std,
                    "minimum_delta_percentage_points": min(deltas),
                    "maximum_delta_percentage_points": max(deltas),
                    "sign_pattern_seed_order_12345_23456_34567": _sign_pattern(deltas),
                    "direction_consistent": not (min(deltas) < 0.0 < max(deltas)),
                }
            )
            if metric == "miou":
                test = stats.ttest_1samp(deltas, popmean=0.0)
                critical = float(stats.t.ppf(0.975, df=len(deltas) - 1))
                half_width = critical * std / math.sqrt(len(deltas))
                primary_inference.append(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "n_paired_seeds": len(deltas),
                        "mean_delta_percentage_points": mean,
                        "std_delta_percentage_points": std,
                        "ci95_low_percentage_points": mean - half_width,
                        "ci95_high_percentage_points": mean + half_width,
                        "cohens_dz": mean / std if std > 0.0 else float("nan"),
                        "paired_t_p_advisory": float(test.pvalue),
                        "sign_pattern_seed_order_12345_23456_34567": _sign_pattern(deltas),
                    }
                )
    q_values = _bh_fdr([row["paired_t_p_advisory"] for row in primary_inference])
    for row, q_value in zip(primary_inference, q_values):
        row["bh_fdr_q_advisory"] = float(q_value)
        row["evidence_grade"] = "none_n_equals_3"
        row["allowed_language"] = "descriptive only; no superiority or causal claim"
    _write_csv(args.output_dir / "paired_seed_deltas.csv", paired_rows)
    _write_csv(args.output_dir / "paired_summary.csv", paired_summary_rows)
    _write_csv(args.output_dir / "primary_inference_advisory.csv", primary_inference)

    per_class_delta_rows = []
    per_class_summary_rows = []
    for comparison, left, right in COMPARISONS:
        for class_name in class_names:
            for metric in ("precision", "recall", "iou"):
                deltas = []
                for seed in SEEDS:
                    delta = 100.0 * (
                        per_class_lookup[(seed, left, class_name)][metric]
                        - per_class_lookup[(seed, right, class_name)][metric]
                    )
                    deltas.append(delta)
                    per_class_delta_rows.append(
                        {
                            "comparison": comparison,
                            "class": class_name,
                            "metric": metric,
                            "seed": seed,
                            "delta_percentage_points": delta,
                        }
                    )
                mean, std = _mean_std(deltas)
                per_class_summary_rows.append(
                    {
                        "comparison": comparison,
                        "class": class_name,
                        "metric": metric,
                        "mean_delta_percentage_points": mean,
                        "std_delta_percentage_points": std,
                        "seed_12345_delta_percentage_points": deltas[0],
                        "seed_23456_delta_percentage_points": deltas[1],
                        "seed_34567_delta_percentage_points": deltas[2],
                        "sign_pattern_seed_order_12345_23456_34567": _sign_pattern(deltas),
                    }
                )
    _write_csv(args.output_dir / "per_class_paired_deltas.csv", per_class_delta_rows)
    _write_csv(args.output_dir / "per_class_paired_summary.csv", per_class_summary_rows)

    gravity_rows = _read_csv(args.gravity_csv)
    gravity = {row["name"]: float(row["angular_error_deg"]) for row in gravity_rows}
    if len(gravity) != args.expected_count or set(gravity) != reference_keys:
        raise ValueError("gravity and evaluation sample keys do not match exactly")
    with args.gravity_summary.open(encoding="utf-8") as handle:
        gravity_summary = json.load(handle)
    if gravity_summary.get("status") != "PASS_AREA5_GRAVITY_ERRORS":
        raise ValueError("gravity computation did not pass")

    sorted_names = sorted(reference_keys)
    angles = np.asarray([gravity[name] for name in sorted_names], dtype=np.float64)
    lower_threshold = float(np.quantile(angles, 1.0 / 3.0))
    upper_threshold = float(np.quantile(angles, 2.0 / 3.0))

    def disagreement_group(angle):
        if angle <= lower_threshold:
            return "low"
        if angle <= upper_threshold:
            return "mid"
        return "high"

    joined_rows = []
    spearman_rows = []
    group_rows = []
    joined_lookup = defaultdict(dict)
    for seed in SEEDS:
        local_dir = _run_dir(args.stage2a_root, args.stage2b_root, seed, "relplus_local")
        pose_dir = _run_dir(args.stage2a_root, args.stage2b_root, seed, "relplus_pose")
        local = _load_per_image(local_dir / "metrics/per_image_metrics.csv")
        pose = _load_per_image(pose_dir / "metrics/per_image_metrics.csv")
        if set(local) != reference_keys or set(pose) != reference_keys:
            raise ValueError("Local/Pose per-image keys are incomplete for seed {}".format(seed))
        seed_rows = []
        for name in sorted_names:
            row = {
                "name": name,
                "area": name.split("/", 1)[0],
                "seed": seed,
                "gravity_error_deg": gravity[name],
                "gravity_group": disagreement_group(gravity[name]),
                "local_pixel_accuracy": local[name]["pixel_accuracy"],
                "pose_pixel_accuracy": pose[name]["pixel_accuracy"],
                "pose_minus_local_pixel_accuracy_percentage_points": 100.0
                * (pose[name]["pixel_accuracy"] - local[name]["pixel_accuracy"]),
                "local_present_class_miou": local[name]["present_class_miou"],
                "pose_present_class_miou": pose[name]["present_class_miou"],
                "pose_minus_local_present_class_miou_percentage_points": 100.0
                * (
                    pose[name]["present_class_miou"]
                    - local[name]["present_class_miou"]
                ),
            }
            seed_rows.append(row)
            joined_rows.append(row)
            joined_lookup[name][seed] = row
        for metric in (
            "pose_minus_local_pixel_accuracy_percentage_points",
            "pose_minus_local_present_class_miou_percentage_points",
        ):
            correlation = stats.spearmanr(
                [row["gravity_error_deg"] for row in seed_rows],
                [row[metric] for row in seed_rows],
            )[0]
            spearman_rows.append(
                {
                    "seed": seed,
                    "performance_delta": metric,
                    "n_images": len(seed_rows),
                    "spearman_rho": float(correlation),
                    "interpretation_scope": "descriptive; images are clustered and no p-value is claimed",
                }
            )
        for group in ("low", "mid", "high"):
            subset = [row for row in seed_rows if row["gravity_group"] == group]
            pixel_delta = [
                row["pose_minus_local_pixel_accuracy_percentage_points"] for row in subset
            ]
            miou_delta = [
                row["pose_minus_local_present_class_miou_percentage_points"]
                for row in subset
            ]
            group_rows.append(
                {
                    "seed": seed,
                    "gravity_group": group,
                    "n_images": len(subset),
                    "gravity_mean_deg": float(
                        np.mean([row["gravity_error_deg"] for row in subset])
                    ),
                    "gravity_median_deg": float(
                        np.median([row["gravity_error_deg"] for row in subset])
                    ),
                    "local_pixel_accuracy_percent": 100.0
                    * float(np.mean([row["local_pixel_accuracy"] for row in subset])),
                    "pose_pixel_accuracy_percent": 100.0
                    * float(np.mean([row["pose_pixel_accuracy"] for row in subset])),
                    "pose_minus_local_pixel_accuracy_mean_pp": float(
                        np.mean(pixel_delta)
                    ),
                    "pose_minus_local_pixel_accuracy_std_pp": float(
                        np.std(pixel_delta, ddof=1)
                    ),
                    "local_present_class_miou_percent": 100.0
                    * float(
                        np.mean([row["local_present_class_miou"] for row in subset])
                    ),
                    "pose_present_class_miou_percent": 100.0
                    * float(
                        np.mean([row["pose_present_class_miou"] for row in subset])
                    ),
                    "pose_minus_local_present_class_miou_mean_pp": float(
                        np.mean(miou_delta)
                    ),
                    "pose_minus_local_present_class_miou_std_pp": float(
                        np.std(miou_delta, ddof=1)
                    ),
                }
            )
    _write_csv(args.output_dir / "per_image_mechanism.csv", joined_rows)
    _write_csv(args.output_dir / "spearman_by_seed.csv", spearman_rows)
    _write_csv(args.output_dir / "gravity_group_summary.csv", group_rows)

    tail_rows = []
    tail_counts = {}
    for threshold in (5.0, 15.0, 30.0, 45.0):
        tail_names = {
            name for name in sorted_names if gravity[name] > threshold
        }
        tail_counts[str(int(threshold))] = len(tail_names)
        for seed in SEEDS:
            subset = [
                joined_lookup[name][seed] for name in tail_names
            ]
            tail_rows.append(
                {
                    "gravity_error_greater_than_deg": threshold,
                    "seed": seed,
                    "n_images": len(subset),
                    "fraction_of_area5": len(subset) / float(args.expected_count),
                    "pose_minus_local_pixel_accuracy_mean_pp": float(
                        np.mean(
                            [
                                row[
                                    "pose_minus_local_pixel_accuracy_percentage_points"
                                ]
                                for row in subset
                            ]
                        )
                    ),
                    "pose_minus_local_present_class_miou_mean_pp": float(
                        np.mean(
                            [
                                row[
                                    "pose_minus_local_present_class_miou_percentage_points"
                                ]
                                for row in subset
                            ]
                        )
                    ),
                }
            )
    _write_csv(args.output_dir / "gravity_tail_summary.csv", tail_rows)

    visual_sets = []
    for seed in SEEDS:
        for arm in ("relplus_local", "relplus_pose"):
            root = (
                _run_dir(args.stage2a_root, args.stage2b_root, seed, arm)
                / "visualizations/predictions"
            )
            visual_sets.append(_prediction_names(root))
    visual_pool = set.intersection(*visual_sets)
    if not visual_pool:
        raise ValueError("no common Local/Pose prediction visualization pool")

    per_name = {}
    for name in sorted_names:
        seed_rows = joined_lookup[name]
        per_name[name] = {
            "gravity_error_deg": gravity[name],
            "gravity_group": disagreement_group(gravity[name]),
            "mean_pose_minus_local_present_class_miou_pp": float(
                np.mean(
                    [
                        seed_rows[seed][
                            "pose_minus_local_present_class_miou_percentage_points"
                        ]
                        for seed in SEEDS
                    ]
                )
            ),
        }

    representative_rows = []
    selected = set()
    for group in ("low", "mid", "high"):
        full_group = [value for value in per_name.values() if value["gravity_group"] == group]
        target_angle = float(
            np.median([value["gravity_error_deg"] for value in full_group])
        )
        delta_values = np.asarray(
            [value["mean_pose_minus_local_present_class_miou_pp"] for value in full_group],
            dtype=np.float64,
        )
        angle_scale = max(
            float(np.std([value["gravity_error_deg"] for value in full_group])), 1.0e-12
        )
        delta_scale = max(float(np.std(delta_values)), 1.0e-12)
        candidates = [
            name
            for name in visual_pool
            if per_name[name]["gravity_group"] == group
        ]
        if len(candidates) < 2:
            raise ValueError("visualization pool lacks two samples in group {}".format(group))
        for stratum, quantile in (("lower_quartile", 0.25), ("upper_quartile", 0.75)):
            target_delta = float(np.quantile(delta_values, quantile))

            def score(name):
                value = per_name[name]
                return (
                    abs(value["gravity_error_deg"] - target_angle) / angle_scale
                    + abs(
                        value["mean_pose_minus_local_present_class_miou_pp"]
                        - target_delta
                    )
                    / delta_scale
                )

            available = [name for name in candidates if name not in selected]
            chosen = min(available, key=lambda name: (score(name), name))
            selected.add(chosen)
            row = {
                "name": chosen,
                "gravity_group": group,
                "performance_stratum": stratum,
                "gravity_error_deg": per_name[chosen]["gravity_error_deg"],
                "target_group_median_gravity_error_deg": target_angle,
                "mean_pose_minus_local_present_class_miou_pp": per_name[chosen][
                    "mean_pose_minus_local_present_class_miou_pp"
                ],
                "target_group_delta_quantile_pp": target_delta,
                "selection_score": score(chosen),
            }
            for seed in SEEDS:
                row["seed_{}_pose_minus_local_present_class_miou_pp".format(seed)] = (
                    joined_lookup[chosen][seed][
                        "pose_minus_local_present_class_miou_percentage_points"
                    ]
                )
            representative_rows.append(row)
    _write_csv(args.output_dir / "representative_samples.csv", representative_rows)

    miou_primary = {
        row["comparison"]: row
        for row in paired_summary_rows
        if row["metric"] == "miou"
    }
    reproducible = all(row["direction_consistent"] for row in miou_primary.values())
    summary = {
        "completion_status": "COMPLETE_STAGE2B_NUMERIC_ANALYSIS",
        "scientific_reproducibility_gate": (
            "PASS_PRIMARY_DIRECTION_REPRODUCIBILITY"
            if reproducible
            else "FAIL_PRIMARY_DIRECTION_REPRODUCIBILITY"
        ),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "classes": class_names,
        "per_image_count_per_run": args.expected_count,
        "validated_run_count": len(SEEDS) * len(ARMS),
        "gravity": {
            "count": len(gravity),
            "low_max_deg": lower_threshold,
            "mid_max_deg": upper_threshold,
            "tail_counts_strictly_greater_than_deg": tail_counts,
            "group_rule": "global Area5 tertiles: low <= q1; q1 < mid <= q2; high > q2",
        },
        "visualization_pool_count": len(visual_pool),
        "representative_selection_rule": (
            "within each gravity tertile, choose from the common saved-prediction pool "
            "the closest sample to the full-pool group median angle and the full-pool "
            "lower/upper quartile of three-seed mean Pose-Local present-class mIoU delta"
        ),
        "primary_miou_paired_summary": miou_primary,
        "inference_scope": (
            "n=3 paired seeds; mean/std, exact seed deltas, sign consistency and CI are "
            "primary. p/q values are advisory only. Per-image Spearman is descriptive "
            "because images are clustered and repeated across seeds."
        ),
    }
    _write_json(args.output_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
