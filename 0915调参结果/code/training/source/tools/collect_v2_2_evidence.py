#!/usr/bin/env python3
"""Collect final V2.2 evidence and assert forbidden actions did not occur."""

import argparse
import filecmp
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", required=True, type=Path)
    parser.add_argument("--original-cmx-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from configs.stanford2d3d_s2d.cmx_mit_b2_rel_plus_v2_2_formal import (
        config as formal,
    )

    core = (
        "camera.py",
        "constants.py",
        "depth.py",
        "encoding.py",
        "generator.py",
        "geometry.py",
        "normal_diagnostics.py",
        "profiles.py",
        "source_helpers.py",
        "stanford_s2d.py",
        "storage.py",
    )
    frozen_core_equal = all(
        filecmp.cmp(
            str(ROOT / "rel_plus" / name),
            str(args.frozen_root / "rel_plus" / name),
            shallow=False,
        )
        for name in core
    )
    ours_models = sorted(
        path.relative_to(ROOT / "models")
        for path in (ROOT / "models").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    reference_models = sorted(
        path.relative_to(args.original_cmx_root / "models")
        for path in (args.original_cmx_root / "models").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    model_tree_equal = ours_models == reference_models and all(
        filecmp.cmp(
            str(ROOT / "models" / relative),
            str(args.original_cmx_root / "models" / relative),
            shallow=False,
        )
        for relative in ours_models
    )
    unwanted = sorted(
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.name in ("__pycache__", ".pytest_cache") or path.suffix == ".pyc"
    )
    evidence = args.evidence_root
    byte_report = _json(evidence / "audit" / "generator_byte_invariant.json")
    audit = _json(evidence / "pilot_cache" / "audit" / "cache_audit_summary.json")
    single = _json(evidence / "pilot_train" / "formal_startup_single.json")
    ddp_mock = _json(evidence / "pilot_train" / "formal_startup_ddp_mock.json")
    evaluator = _json(evidence / "pilot_eval" / "evaluator_smoke_3.json")
    rank_consistency = _json(
        evidence / "formal_eval_smoke" / "rank_consistency.json"
    )
    trace = _json(evidence / "pilot_train" / "three_arm_real_trace.json")
    throughput = _json(
        evidence / "pilot_train" / "dataloader_throughput_1000_final.json"
    )
    fault = _json(evidence / "cache_tools_smoke" / "fault_detection_final.json")
    synthetic_sweep = _json(
        evidence
        / "formal_eval_smoke"
        / "synthetic_sweep"
        / "metrics_epoch200.json"
    )
    rel_count = len(list((evidence / "pilot_cache" / "RELPlus").rglob("*.png")))
    mask_count = len(list((evidence / "pilot_cache" / "ValidMask").rglob("*.png")))
    checkpoint_files = sorted(evidence.rglob("*.pth"))
    non_synthetic_checkpoints = [
        str(path)
        for path in checkpoint_files
        if "synthetic_checkpoints" not in path.parts
        and "missing_checkpoint_fixture" not in path.parts
    ]
    checks = {
        "formal_training_authorized_false": formal.training_authorized is False,
        "full_cache_authorized_false": formal.full_cache_authorized is False,
        "formal_data_ready_false": formal.data_ready is False,
        "frozen_relplus_core_equal": frozen_core_equal,
        "original_cmx_models_equal": model_tree_equal,
        "byte_regression_zero": (
            byte_report["status"] == "PASS"
            and byte_report["changed_pixels"] == 0
            and byte_report["changed_channels"] == 0
            and byte_report["max_difference"] == 0
            and byte_report["real_sample_count"] >= 12
        ),
        "pilot_cache_only_36": rel_count == mask_count == 36,
        "pilot_audit_pass": audit["status"] == "PASS" and audit["failure_count"] == 0,
        "pilot_regeneration_pass": (
            audit["regeneration_count"] == 36
            and audit["regeneration_failure_count"] == 0
        ),
        "cache_fault_detection_pass": fault["status"] == "PASS",
        "single_startup_no_step": (
            single["status"] == "PASS"
            and single["backward_executed"] is True
            and single["optimizer_step_executed"] is False
            and single["checkpoint_written"] is False
            and single["formal_training_started"] is False
        ),
        "ddp_mock_startup_no_step": (
            ddp_mock["status"] == "PASS"
            and ddp_mock["optimizer_step_executed"] is False
            and ddp_mock["checkpoint_written"] is False
        ),
        "throughput_sampler_contract": (
            throughput["status"] == "PASS"
            and throughput["dataset_getitem_full_randperm_calls"] == 0
            and throughput["sampler_randperm_calls_per_epoch"] == 1
        ),
        "evaluator_smoke_not_scientific": (
            evaluator["status"] == "PASS"
            and evaluator["processed_samples"] == 3
            and evaluator["scientific_metric_reported"] is False
        ),
        "rank_confusions_exact": (
            rank_consistency["status"] == "PASS"
            and rank_consistency["single_equals_2_rank"] is True
            and rank_consistency["single_equals_8_rank"] is True
        ),
        "synthetic_sweep_not_scientific": (
            synthetic_sweep["epoch"] == 200
            and synthetic_sweep["scientific_metric_reported"] is False
        ),
        "real_three_arm_trace": (
            trace["status"] == "PASS"
            and trace["trace_count_per_arm"] >= 50
            and trace["mismatch_count"] == 0
            and trace["constructed_trace_copy_used"] is False
        ),
        "no_formal_checkpoint": not non_synthetic_checkpoints,
        "no_full_cache_directory": not (evidence / "formal_cache_forbidden").exists(),
        "source_tree_clean_of_runtime_caches": not unwanted,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "stage": "V2.2 pre-authorization infrastructure",
        "checks": checks,
        "code_root": str(ROOT),
        "frozen_root": str(args.frozen_root),
        "evidence_root": str(evidence),
        "source_file_count": sum(path.is_file() for path in ROOT.rglob("*")),
        "unwanted_source_entries": unwanted,
        "pilot_rel_plus_count": rel_count,
        "pilot_valid_mask_count": mask_count,
        "synthetic_checkpoint_fixture_count": len(checkpoint_files),
        "non_synthetic_checkpoint_files": non_synthetic_checkpoints,
        "full_cache_generated": False,
        "formal_training_started": False,
        "optimizer_step_executed": False,
        "formal_checkpoint_written": False,
        "full_test_evaluation_executed": False,
        "scientific_mIoU_reported": False,
        "file_hash_written": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
