#!/usr/bin/env python3
"""Verify this archive offline with Python 3.8+ and only its standard library.

This checks archived evidence and independently recomputes confusion-matrix
metrics. It does not execute the archived model or hash weights/datasets that
are absent from the archive. SHA256SUMS is an integrity inventory, not a
cryptographic signature or independent proof of experimental authenticity.
"""
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


ROOT = Path(__file__).resolve().parent
ARMS = ("rgbd", "hha", "relplus")
LABELS = {"rgbd": "RGBD", "hha": "HHA", "relplus": "RELPlus"}
N = 17593
C = 13
WORLD = 8
PIXELS = 3973620198
TEST_SHA = "b9de196c6c1aa8f9ac37926910af0806ce59b91eb068998711ddbb78eb24423a"
CLASSES = ["beam", "board", "bookcase", "ceiling", "chair", "clutter", "column",
           "door", "floor", "sofa", "table", "wall", "window"]
METRICS = ("mIoU", "pixel_accuracy", "mean_accuracy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError("nonfinite JSON number: " + value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object, parse_constant=invalid_constant)


def integer(value, label):
    if isinstance(value, str):
        require(re.fullmatch(r"[0-9]+", value) is not None, label + ": not integer text")
        value = int(value)
    require(type(value) is int and 0 <= value <= 2 ** 63 - 1,
            label + ": not a nonnegative int64")
    return value


def close(value, expected, label, tolerance=1e-10):
    require(not isinstance(value, bool), label + ": Boolean is not a metric")
    actual = float(value)
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
            label + ": value differs from independent recomputation")


def relative_path(name):
    require(isinstance(name, str) and name != "" and "\\" not in name,
            "invalid inventory filename")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and
            path.as_posix() == name and "." not in path.parts,
            "unsafe inventory filename: " + name)
    return path


def file_set(root):
    result = set()
    require(root.is_dir(), "missing directory: " + str(root))
    for p in root.rglob("*"):
        if "__pycache__" in p.relative_to(root).parts:
            continue
        require(not p.is_symlink(), "archive symlinks are not allowed: " + str(p))
        if p.is_file():
            result.add(p.relative_to(root).as_posix())
        else:
            require(p.is_dir(), "unexpected filesystem entry: " + str(p))
    return result


def verify_inventory(root, name, count=None):
    inventory = {}
    for line in (root / name).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "malformed SHA inventory line: " + name)
        digest, relative = match.groups()
        relative_path(relative)
        require(relative not in inventory and relative != name,
                "duplicate or self-referential SHA entry: " + relative)
        inventory[relative] = digest
    if count is not None:
        require(len(inventory) == count, name + ": wrong inventory count")
    require(file_set(root) == set(inventory) | {name},
            name + ": actual file set does not equal the inventory")
    for relative, expected in inventory.items():
        require(sha(root / relative) == expected, "SHA-256 mismatch: " + relative)
    return inventory


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, "CSV header mismatch: " + str(path))
        rows = list(reader)
    require(all(set(row) == set(fields) and all(v is not None for v in row.values())
                for row in rows), "malformed CSV row: " + str(path))
    return rows


def matrix(value, label, strings=False):
    require(isinstance(value, list) and len(value) == C, label + ": expected 13 rows")
    result = []
    for row in value:
        require(isinstance(row, list) and len(row) == C, label + ": expected 13 columns")
        if not strings:
            require(all(type(v) is int for v in row), label + ": matrix must contain integers")
        result.append([integer(v, label) for v in row])
    require(sum(map(sum, result)) <= PIXELS, label + ": excess pixel count")
    return result


def check_metrics(directory, arm):
    observed = read_json(directory / "metrics.json")
    expected_protocol = ("CMX_RELPLUS_V2_3" if arm == "relplus" else "CMX_S2D_THREE_ARM_V1")
    representation = {"rgbd": "RGBD_UINT8_REPEAT3_SOURCECOMPAT",
                      "hha": "HHA_FROZEN_CACHE_EXECUTABLE_ORDER",
                      "relplus": "RELPLUS_V2_1_OFFLINE480_SOURCECOMPAT"}[arm]
    fixed = {"status": "PASS", "checkpoint_epoch": 200, "evaluation_sample_count": N,
             "class_count": C, "ignore_index": 255, "world_size": WORLD,
             "eval_scale_array": [1], "eval_flip": False, "eval_crop_size": [480, 480],
             "eval_align_corners": False, "metric_unit": "fraction_0_to_1",
             "scientific_metric_reported": True, "integration_protocol_id": expected_protocol,
             "representation_protocol_id": representation, "valid_pixel_count": PIXELS}
    for key, value in fixed.items():
        require(key in observed and type(observed[key]) is type(value) and observed[key] == value,
                str(directory) + ": invalid metric protocol field " + key)
    rows = read_csv(directory / "confusion_matrix.csv", ["true_class"] +
                    ["pred_{}".format(i) for i in range(C)])
    require([integer(r["true_class"], "true_class") for r in rows] == list(range(C)),
            "confusion CSV class order differs")
    hist = matrix([[r["pred_{}".format(i)] for i in range(C)] for r in rows],
                  str(directory), strings=True)
    require(sum(map(sum, hist)) == PIXELS, "total valid pixels differ")
    gt = [sum(row) for row in hist]
    pred = [sum(hist[r][c] for r in range(C)) for c in range(C)]
    require(all(v > 0 for v in gt), "each frozen class must have ground-truth pixels")
    iou = [hist[i][i] / (gt[i] + pred[i] - hist[i][i]) for i in range(C)]
    accuracy = [hist[i][i] / gt[i] for i in range(C)]
    computed = {"mIoU": sum(iou) / C, "pixel_accuracy": sum(hist[i][i] for i in range(C)) / PIXELS,
                "mean_accuracy": sum(accuracy) / C}
    for key, value in computed.items():
        close(observed[key], value, key, 1e-12)
        close(observed[key + "_percent"], value * 100, key + "_percent")
    for key, scale in (("per_class_iou", 1), ("per_class_iou_percent", 100)):
        require(isinstance(observed[key], list) and len(observed[key]) == C, key + ": invalid shape")
        for i in range(C):
            close(observed[key][i], iou[i] * scale, key)
    per_class = read_csv(directory / "per_class_iou.csv",
                         ["class_id", "class_name", "IoU_fraction", "IoU_percent"])
    require([integer(r["class_id"], "class_id") for r in per_class] == list(range(C)) and
            [r["class_name"] for r in per_class] == CLASSES, "per-class CSV order differs")
    for i, row in enumerate(per_class):
        close(row["IoU_fraction"], iou[i], "class IoU", 1e-12)
        close(row["IoU_percent"], iou[i] * 100, "class IoU percent")
    return {"metrics": observed, "matrix": hist, "gt": gt, "iou": iou}


def verify_code(preflight):
    training = ROOT / "code/training"
    bundle = read_json(training / "bundle_manifest.json")
    expected = bundle["files"]
    require(len(expected) == 184 and bundle["source_snapshot_files"] == 174,
            "training bundle count differs")
    require(file_set(training) == set(expected) | {"bundle_manifest.json"},
            "training bundle file set differs")
    require(expected == preflight["source_hashes"], "evaluation used a different training bundle")
    for name, digest in expected.items():
        relative_path(name)
        require(sha(training / name) == digest, "training bundle hash mismatch: " + name)
    eval_root = ROOT / "code/evaluation"
    implementations = preflight["implementation_hashes"]
    require(set(implementations) == {"eval_rank.py", "run_evaluation.py", "input_adapter.py", "audit_eval.py"},
            "unexpected evaluation implementation file set")
    require(file_set(eval_root) == set(implementations) | {"PROTOCOL.json"},
            "evaluation code file set differs")
    for name, digest in implementations.items():
        require(sha(eval_root / name) == digest, "evaluation implementation hash mismatch: " + name)
        require(sha(ROOT / "evidence/evaluation/provenance" / name) == digest,
                "remote evaluation implementation copy differs: " + name)
    require(sha(eval_root / "PROTOCOL.json") == sha(ROOT / "evidence/evaluation/provenance/PROTOCOL.json"),
            "evaluation protocol copy differs")
    suite = read_json(training / "suite.json")
    require(suite == preflight["suite"], "training suite and evaluation preflight differ")
    return suite


def verify_training(preflight, suite):
    completion = read_json(ROOT / "evidence/training/completion.json")
    queue = completion["queue"]
    require(queue == preflight["training"], "training completion differs from evaluation prerequisite")
    require(queue["status"] == "PASS" and queue["phase"] == "train" and
            queue["arms"] == list(ARMS) and queue["active_arm"] is None and
            [r["arm"] for r in queue["completed"]] == list(ARMS), "training queue incomplete")
    require(queue["suite_sha256"] == sha(ROOT / "code/training/suite.json"), "training suite fingerprint differs")
    require(set(completion["arms"]) == set(ARMS), "training arm set differs")
    for arm, summary in zip(ARMS, queue["completed"]):
        row = completion["arms"][arm]
        require(row["status"] == summary and row["exitcode"] == 0,
                arm + ": training summary/exit mismatch")
        require(summary["status"] == "PASS" and summary["epoch"] == 200 and
                summary["global_iteration"] == 189000 and summary["rank_done_count"] == WORLD,
                arm + ": incomplete training endpoint")
        runtime = row["runtime"]
        for key, expected in {"status": "FORMAL_TRAINING_COMPLETED", "epoch": 200,
                              "global_iteration": 189000, "iteration_in_epoch": 945,
                              "author_nan_replacement_count": 0, "world_size": WORLD,
                              "optimizer_step_executed": True}.items():
            require(type(runtime[key]) is type(expected) and runtime[key] == expected,
                    arm + ": invalid training runtime " + key)
        ranks = row["ranks"]
        require(len(ranks) == WORLD and [r["rank"] for r in ranks] == list(range(WORLD)),
                arm + ": missing or duplicate training ranks")
        for rank in ranks:
            require(rank["status"] == "COMPLETED" and rank["arm"] == arm and
                    rank["mode"] == "formal" and rank["epoch"] == 200 and
                    rank["global_iteration"] == 189000 and rank["suite_id"] == suite["suite_id"],
                    arm + ": invalid training rank completion")
        cp = preflight["arms"][arm]["checkpoint"]
        require(cp["path"] == summary["checkpoint"] and cp["size"] == summary["checkpoint_size"] and
                cp["sha256"] == summary["checkpoint_sha256"], arm + ": checkpoint record mismatch")
        require(re.fullmatch(r"[0-9a-f]{64}", cp["sha256"]) is not None and cp["size"] > 0,
                arm + ": invalid checkpoint fingerprint")
        integer(cp["mtime_ns"], "checkpoint mtime")
    return completion


def verify_new_arm(arm, preflight, suite):
    out = ROOT / "evidence/evaluation" / arm
    evidence = preflight["arms"][arm]
    cfg = evidence["config"]
    cp = evidence["checkpoint"]
    require(cfg["class_names"] == CLASSES and cfg["comparison_arm"] == arm,
            arm + ": class order or config arm differs")
    for key, value in suite["shared"].items():
        require(cfg[key] == value, arm + ": training/evaluation config differs: " + key)
    require(evidence["ordered_test_count"] == N and evidence["test_list_sha256"] == TEST_SHA,
            arm + ": preflight test identity differs")
    require((out / "exitcode").read_text().strip() == "0", arm + ": evaluation exit nonzero")
    require(read_json(out / "status.json")["status"] == "PASS", arm + ": evaluation did not pass")
    require(read_json(out / "cleanup.json")["remaining_members"] == [], arm + ": cleanup incomplete")
    result = check_metrics(out / "evaluation", arm)
    require(result["metrics"]["checkpoint"] == cp["path"], arm + ": evaluated checkpoint path differs")
    shards, matrices = [], []
    expected_modules = {"evaluator": "tools/eval_rel_plus_v2_3_full.py",
                        "core": "engine/relplus_evaluator.py", "model": "models/builder.py",
                        "dataset": "dataloader/RGBXDataset.py"}
    for rank in range(WORLD):
        runtime = read_json(out / "rank_{:02d}_runtime.json".format(rank))
        count = len(range(rank, N, WORLD))
        fixed = {"status": "COMPLETED", "rank": rank, "arm": arm, "exitcode": 0,
                 "processed_samples": count, "state_key_count": 837,
                 "checkpoint_payload_epoch": 200, "checkpoint": cp["path"], "amp": False}
        for key, value in fixed.items():
            require(type(runtime[key]) is type(value) and runtime[key] == value,
                    arm + ": rank runtime differs: " + key)
        for key, relative in expected_modules.items():
            require(runtime["loaded_modules"][key] == suite["remote_source_root"] + "/source/" + relative,
                    arm + ": module escaped frozen source: " + key)
        decoder_bn = [r for r in runtime["batch_norm"] if r["name"] == "decode_head.linear_fuse.1"]
        require(len(decoder_bn) == 1 and decoder_bn[0]["eps"] == 1e-5 and
                decoder_bn[0]["type"] == "BatchNorm2d", arm + ": evaluation decoder BN differs")
        report = read_json(out / "evaluation/rank_{:02d}_evaluation.json".format(rank))
        require(report["rank"] == report["local_rank"] == rank and report["world_size"] == WORLD and
                report["sample_count"] == count, arm + ": rank report count/identity differs")
        ids = report["owned_sample_ids"]
        require(len(ids) == count and all(isinstance(v, str) and v and not any(c in v for c in "\r\n") for v in ids),
                arm + ": invalid owned sample IDs")
        shards.append(ids)
        matrices.append(matrix(report["confusion_matrix"], arm + " rank matrix"))
    ids = [shards[i % WORLD][i // WORLD] for i in range(N)]
    require(len(set(ids)) == N, arm + ": duplicated test IDs")
    test_bytes = ("\n".join(ids) + "\n").encode("utf-8")
    require(hashlib.sha256(test_bytes).hexdigest() == TEST_SHA,
            arm + ": reconstructed ordered test.txt hash differs")
    manifest = read_csv(out / "evaluation/evaluation_manifest.csv", ["sample_id", "rank"])
    expected_manifest = [{"sample_id": value, "rank": str(rank)}
                         for rank, shard in enumerate(shards) for value in shard]
    require(manifest == expected_manifest, arm + ": manifest does not match exact rank ownership")
    summed = [[sum(m[r][c] for m in matrices) for c in range(C)] for r in range(C)]
    require(summed == result["matrix"], arm + ": rank matrices do not equal final confusion")
    audit = read_json(out / "integrity_audit.json")
    require(audit["status"] == "PASS" and audit["metrics"] == result["metrics"] and
            audit["gt_histogram"] == result["gt"] and audit["test_source_sha256"] == TEST_SHA and
            audit["checkpoint_sha256"] == cp["sha256"] and
            audit["rank_counts"] == [len(s) for s in shards], arm + ": archived audit differs")
    result["ids"] = ids
    return result


def verify_comparison(new, old, preflight, status):
    comparison = read_json(ROOT / "results/comparison_verified.json")
    remote = read_json(ROOT / "evidence/evaluation/three_arm_comparison.json")
    require(comparison["status"] == "PASS_LOCAL_INDEPENDENT_COMPARISON" and
            comparison["new_evaluation_status"] == "PASS" and
            comparison["single_seed"] == 12345 and comparison["batch_and_lr_changed_together"] is True,
            "local comparison protocol differs")
    require(comparison["old_and_new_gt_histogram_equal"] is True and comparison["gt_histogram"] == new["rgbd"]["gt"] and
            comparison["test_source_sha256"] == TEST_SHA, "local comparison data identity differs")
    require(remote["status"] == "PASS_FIXED_EPOCH200_DESCRIPTIVE_COMPARISON" and
            remote["seed"] == 12345 and remote["training_batch_size"] == 56 and
            remote["training_lr"] == 0.00012 and remote["evaluation_samples_per_arm"] == N and
            remote["class_names"] == CLASSES, "remote comparison protocol differs")
    fields = ["arm", "old_mIoU_percent", "new_mIoU_percent", "delta_mIoU_pp",
              "pixel_accuracy_percent", "mean_accuracy_percent"]
    csv_rows = read_csv(ROOT / "results/new_vs_0915_metrics.csv", fields)
    json_rows = comparison["rows"]
    require(len(csv_rows) == len(json_rows) == len(ARMS), "summary row count differs")
    for arm, csv_row, json_row in zip(ARMS, csv_rows, json_rows):
        require(csv_row["arm"] == json_row["arm"] == LABELS[arm], "summary arm order differs")
        current, prior = new[arm]["metrics"], old[arm]["metrics"]
        expected = {"old_mIoU_percent": prior["mIoU_percent"], "new_mIoU_percent": current["mIoU_percent"],
                    "delta_mIoU_pp": current["mIoU_percent"] - prior["mIoU_percent"],
                    "pixel_accuracy_percent": current["pixel_accuracy_percent"],
                    "mean_accuracy_percent": current["mean_accuracy_percent"]}
        for key, value in expected.items():
            close(csv_row[key], value, "summary CSV " + key)
            close(json_row[key], value, "summary JSON " + key)
        require(status["metrics"][arm] == remote["arms"][arm]["metrics"] == current,
                "aggregate metrics differ from per-arm evidence")
        require(comparison["new_checkpoint_sha256"][arm] == preflight["arms"][arm]["checkpoint"]["sha256"],
                "comparison checkpoint identity differs")
    for a, b in (("relplus", "hha"), ("relplus", "rgbd"), ("hha", "rgbd")):
        pair = a + "-" + b
        for key in METRICS:
            field = key + "_percent"
            value = new[a]["metrics"][field] - new[b]["metrics"][field]
            close(remote["differences_percentage_points"][pair][field], value, "remote pairwise difference")
            close(comparison["new_pairwise_differences_pp"][pair][field], value, "local pairwise difference")
    class_fields = ["class_id", "class_name"]
    for label in LABELS.values():
        class_fields.extend([label + "_new_IoU_percent", label + "_old_IoU_percent", label + "_delta_pp"])
    class_fields.extend(["RELPlus_minus_HHA_pp", "RELPlus_minus_RGBD_pp"])
    rows = read_csv(ROOT / "results/new_vs_0915_per_class.csv", class_fields)
    require(len(rows) == C, "per-class comparison row count differs")
    for i, row in enumerate(rows):
        require(integer(row["class_id"], "class ID") == i and row["class_name"] == CLASSES[i],
                "per-class comparison order differs")
        for arm, label in LABELS.items():
            current, prior = new[arm]["iou"][i] * 100, old[arm]["iou"][i] * 100
            for suffix, value in (("_new_IoU_percent", current), ("_old_IoU_percent", prior),
                                  ("_delta_pp", current - prior)):
                close(row[label + suffix], value, "per-class comparison " + label + suffix)
        close(row["RELPlus_minus_HHA_pp"], (new["relplus"]["iou"][i] - new["hha"]["iou"][i]) * 100,
              "RELPlus minus HHA class difference")
        close(row["RELPlus_minus_RGBD_pp"], (new["relplus"]["iou"][i] - new["rgbd"]["iou"][i]) * 100,
              "RELPlus minus RGBD class difference")
    close(comparison["evaluation_elapsed_seconds"], status["finished_at"] - status["started_at"],
          "evaluation elapsed seconds")
    close(comparison["decoder_BN_eps_eval"], 1e-5, "evaluation BN epsilon", 1e-15)
    close(comparison["decoder_BN_eps_training"], 1e-3, "training BN epsilon", 1e-15)


def verify_provenance(preflight, suite, inventory):
    protocol = read_json(ROOT / "provenance/frozen_protocol.json")
    checkpoints = read_json(ROOT / "provenance/checkpoints.json")
    completeness = read_json(ROOT / "provenance/archive_completeness.json")
    copied = read_json(ROOT / "provenance/copied_files.json")
    source = read_json(ROOT / "provenance/source_provenance.json")
    environment = read_json(ROOT / "provenance/observed_environment.json")
    eval_protocol = read_json(ROOT / "code/evaluation/PROTOCOL.json")
    require(protocol["status"] == "FROZEN_FIXED_EPOCH200_SINGLE_SEED" and
            protocol["suite_id"] == suite["suite_id"] and protocol["training"] == suite["shared"] and
            protocol["budget"] == suite["budget"] and protocol["evaluation"] == eval_protocol,
            "frozen protocol differs from executed configuration")
    require(protocol["actual_optimizer_updates_per_arm"] == 189000 and
            protocol["actual_warmup_updates"] == 9450 and
            protocol["training_completion"] == "evidence/training/completion.json",
            "frozen update budget differs")
    data = protocol["dataset"]
    require(data["test_count"] == N and data["train_count"] == 52903 and data["classes"] == CLASSES and
            data["image_size"] == [480, 480] and data["ignore_index"] == 255 and
            data["test_list_sha256"] == TEST_SHA, "frozen dataset protocol differs")
    require(checkpoints["weights_included"] is False and checkpoints["endpoint"] == "fixed_epoch_200" and
            checkpoints["seed"] == 12345 and set(checkpoints["arms"]) == set(ARMS),
            "checkpoint provenance scope differs")
    startup = ROOT / "evidence/training/startup_20260915/remote/preflight.json"
    startup_preflight = read_json(startup)
    require(sha(startup) == preflight["training"]["prerequisites"]["preflight_sha256"],
            "startup preflight differs from training completion prerequisite")
    require(environment["source"] == startup.relative_to(ROOT).as_posix() and
            environment["environment"] == startup_preflight["environment"],
            "environment provenance differs from recorded preflight")
    for arm in ARMS:
        require(checkpoints["arms"][arm] == preflight["arms"][arm]["checkpoint"],
                arm + ": checkpoint provenance differs from evaluation audit")
        cfg_path = protocol["resolved_config_files"][arm]
        require(cfg_path == "configs/" + arm + ".json", "unexpected resolved configuration path")
        require(read_json(ROOT / cfg_path) == preflight["arms"][arm]["config"] and
                sha(ROOT / cfg_path) == startup_preflight["arms"][arm]["resolved_config_sha256"],
                arm + ": resolved config is not the original training configuration")
        evidence = preflight["arms"][arm]["data_evidence"]
        require(evidence["eval_source"]["sha256"] == TEST_SHA and
                evidence["train_source"]["sha256"] == data["train_list_sha256"],
                arm + ": frozen split hashes differ")
    require(source["remote_training_bundle"] == suite["remote_source_root"] and
            source["remote_source_snapshot"] == suite["remote_source_root"] + "/source" and
            source["remote_training_output"] == suite["output_root"] and
            source["remote_evaluation_output"] == eval_protocol["output_root"] and
            source["source_snapshot_python_files"] == 174 and source["training_bundle_manifest_files"] == 184,
            "source provenance paths/counts differ")
    require(completeness["status"] == "FROZEN_RESULTS_AND_EXECUTION_EVIDENCE" and
            completeness["training_bundle_files"] == 184 and completeness["source_snapshot_python_files"] == 174 and
            completeness["evaluation_evidence_files"] == 102 and
            completeness["evaluation_evidence_manifest_included"] is True and
            completeness["original_copy_file_count"] == len(copied), "archive completeness counts differ")
    for name, record in copied.items():
        relative_path(name)
        require(name in inventory and inventory[name] == record["sha256"],
                "copied-file provenance hash differs: " + name)
    require(not any(PurePosixPath(name).suffix.lower() in (".pth", ".pt", ".ckpt", ".safetensors")
                    for name in inventory), "model weights unexpectedly present despite exclusion statement")


def main():
    inventory = verify_inventory(ROOT, "SHA256SUMS")
    nested = verify_inventory(ROOT / "evidence/evaluation", "EVIDENCE.sha256", 102)
    preflight = read_json(ROOT / "evidence/evaluation/preflight.json")
    require(preflight["status"] == "PASS" and set(preflight["arms"]) == set(ARMS),
            "evaluation preflight incomplete")
    suite = verify_code(preflight)
    verify_training(preflight, suite)
    status = read_json(ROOT / "evidence/evaluation/status.json")
    require(status["status"] == "PASS" and status["completed"] == list(ARMS) and status["active_arm"] is None,
            "evaluation queue incomplete")
    snapshot = read_json(ROOT / "evidence/evaluation/completion_snapshot.json")
    require(snapshot["status"] == "PASS" and snapshot["own_processes_remaining"] == [],
            "evaluation completion snapshot failed")
    new = {arm: verify_new_arm(arm, preflight, suite) for arm in ARMS}
    old = {arm: check_metrics(ROOT / "baseline_0915/results" / LABELS[arm], arm) for arm in ARMS}
    require(all(row["gt"] == new["rgbd"]["gt"] for row in list(new.values()) + list(old.values())),
            "new/old ground-truth class histograms differ")
    require(all(row["ids"] == new["rgbd"]["ids"] for row in new.values()),
            "test ID order differs across arms")
    verify_comparison(new, old, preflight, status)
    verify_provenance(preflight, suite, inventory)
    result = {"status": "PASS_OFFLINE_ARCHIVE_VERIFICATION", "archive_files_hashed": len(inventory),
              "nested_evidence_files_hashed": len(nested), "training_bundle_files": 184,
              "training_completed_ranks": 24, "evaluation_completed_ranks": 24,
              "unique_test_samples_per_arm": N, "reconstructed_test_sha256": TEST_SHA,
              "valid_pixels_per_arm": PIXELS, "old_and_new_gt_equal": True,
              "metrics_independently_recomputed": True,
              "mIoU_percent": {arm: new[arm]["metrics"]["mIoU_percent"] for arm in ARMS},
              "unarchived_weights_or_datasets_rehashed": False,
              "inference_rerun": False}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"status": "FAIL_OFFLINE_ARCHIVE_VERIFICATION", "error": str(error)},
                         ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
