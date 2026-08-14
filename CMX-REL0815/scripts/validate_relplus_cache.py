#!/usr/bin/env python3
"""Fail closed unless the run-local REL+ cache is complete and reproducible."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import time

import cv2


IMAGE_SHAPE = (480, 480, 3)
PIXELS_PER_IMAGE = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]
CHANNEL_ORDER = ("ReD", "EGVIA", "LOA")
MAX_REPORTED_ERRORS = 200
GENERATOR_FILES = (
    "relplus/__init__.py",
    "relplus/geometry.py",
    "relplus/io.py",
    "relplus/representation.py",
    "relplus/spec.py",
    "scripts/prepare_relplus.py",
)
RELPLUS_SPEC = {
    "model_name": "cmx_rel+",
    "config_name": "cmx_rel+",
    "representation_semantics": "REL-default",
    "representation_version": "relplus_rel_default_v2",
    "point_frame": "camera_centered_world_axes",
    "translation_in_red_loa": False,
    "channel_order": ["ReD", "EGVIA", "LOA"],
    "depth_definition": "camera-z; metres=(uint16+1)/512; 65535 invalid",
    "pixel_origin": 1,
    "normal_estimator": "REL square-support algebraic plane fit",
    "normal_radius_native_pixels": 3,
    "alpha_degrees": 45.0,
    "lambda": 0.5,
    "red_height_normalization": "valid-image min-max to uint8",
    "angle_normalization": "zero-to-pi linearly mapped to uint8",
    "invalid_pixel": [255, 255, 255],
    "native_depth_shape": [1080, 1080],
    "cache_shape": [480, 480, 3],
    "cache_resize": "bilinear channels; nearest validity mask",
    "output_dtype": "uint8",
    "intrinsics_usage": "K backprojects camera-z depth into camera coordinates",
    "extrinsics_usage": (
        "R rotates points and normals into world axes; t/C are validated and retained "
        "for provenance but do not enter ReD, EGVIA, or LOA"
    ),
}
RELPLUS_SPEC_SHA256 = hashlib.sha256(
    (json.dumps(RELPLUS_SPEC, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generator_identity(repo_root):
    files = {
        relative: sha256_file(repo_root / relative)
        for relative in GENERATOR_FILES
    }
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n"
    return files, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def read_split(path, split_name, add_error):
    identifiers = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            sample_id = raw_line.strip()
            if not sample_id:
                add_error("{} contains a blank ID at line {}".format(split_name, line_number))
                continue
            pure = PurePosixPath(sample_id)
            if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
                add_error(
                    "{} contains an unsafe ID at line {}: {!r}".format(
                        split_name, line_number, sample_id
                    )
                )
            identifiers.append(sample_id)
    duplicates = len(identifiers) - len(set(identifiers))
    if duplicates:
        add_error("{} contains {} duplicate ID(s)".format(split_name, duplicates))
    return identifiers, duplicates


def load_manifest(path, add_error):
    rows = {}
    row_count = 0
    duplicate_count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                add_error("manifest contains a blank row at line {}".format(line_number))
                continue
            row_count += 1
            try:
                row = json.loads(raw_line)
            except (TypeError, ValueError) as error:
                add_error("manifest line {} is invalid JSON: {}".format(line_number, error))
                continue
            if not isinstance(row, dict):
                add_error("manifest line {} is not an object".format(line_number))
                continue
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                add_error("manifest line {} has no valid sample_id".format(line_number))
                continue
            if sample_id in rows:
                duplicate_count += 1
                add_error("manifest contains duplicate sample_id {!r}".format(sample_id))
                continue
            rows[sample_id] = row
    return rows, row_count, duplicate_count


def validate_image(task):
    sample_id, path, expected_sha = task
    result = {
        "sample_id": sample_id,
        "sha256_match": False,
        "decodable": False,
        "shape_dtype_valid": False,
        "validated": False,
        "error": None,
    }
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = sha256_file(path)
        result["actual_sha256"] = actual_sha
        result["sha256_match"] = (
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and actual_sha == expected_sha.lower()
        )
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("OpenCV could not decode the PNG")
        result["decodable"] = True
        result["actual_shape"] = list(image.shape)
        result["actual_dtype"] = str(image.dtype)
        result["shape_dtype_valid"] = (
            tuple(image.shape) == IMAGE_SHAPE and str(image.dtype) == "uint8"
        )
        result["validated"] = result["sha256_match"] and result["shape_dtype_valid"]
        if not result["sha256_match"]:
            result["error"] = "SHA-256 does not match the manifest"
        elif not result["shape_dtype_valid"]:
            result["error"] = "expected 480x480x3 uint8, got {} {}".format(
                tuple(image.shape), image.dtype
            )
    except Exception as error:  # Per-file errors belong in the complete audit report.
        result["error"] = "{}: {}".format(type(error).__name__, error)
    return result


def validate(args):
    started = time.time()
    run = args.run_dir.resolve()
    reports = run / "data_reports"
    cache = run / "relplus_cache"
    errors = []
    failure_count = [0]

    def add_error(message):
        failure_count[0] += 1
        if len(errors) < MAX_REPORTED_ERRORS:
            errors.append(message)

    data_audit_path = reports / "data_audit.json"
    with open(data_audit_path, "r", encoding="utf-8") as handle:
        data_audit = json.load(handle)
    spec_path = reports / "relplus_representation_spec.json"
    with open(spec_path, "r", encoding="utf-8") as handle:
        representation_spec = json.load(handle)
    actual_spec_sha256 = sha256_file(spec_path)
    generator_path = Path(__file__).resolve().with_name("prepare_relplus.py")
    actual_generator_sha256 = sha256_file(generator_path)
    repo_root = Path(__file__).resolve().parents[1]
    actual_generator_files_sha256, actual_generator_bundle_sha256 = generator_identity(
        repo_root
    )
    representation_semantics_valid = True
    representation_generator_valid = True

    def check_representation(condition, message):
        nonlocal representation_semantics_valid
        if not condition:
            representation_semantics_valid = False
            add_error(message)

    def check_generator(condition, message):
        nonlocal representation_generator_valid
        if not condition:
            representation_generator_valid = False
            add_error(message)

    check_representation(
        representation_spec == RELPLUS_SPEC,
        "REL+ representation spec does not match REL-default",
    )
    check_representation(
        actual_spec_sha256 == RELPLUS_SPEC_SHA256,
        "REL+ representation spec SHA-256 mismatch",
    )
    for key in (
        "representation_semantics",
        "representation_version",
        "point_frame",
        "translation_in_red_loa",
    ):
        check_representation(
            data_audit.get(key) == RELPLUS_SPEC[key],
            "data_audit {} does not match REL-default".format(key),
        )
    check_representation(
        data_audit.get("representation_spec_sha256") == RELPLUS_SPEC_SHA256,
        "data_audit representation_spec_sha256 mismatch",
    )
    check_representation(
        data_audit.get("representation_generator_sha256") == actual_generator_sha256,
        "data_audit representation_generator_sha256 mismatch",
    )
    check_generator(
        data_audit.get("representation_generator_files_sha256")
        == actual_generator_files_sha256,
        "data_audit representation generator file hashes mismatch",
    )
    check_generator(
        data_audit.get("representation_generator_bundle_sha256")
        == actual_generator_bundle_sha256,
        "data_audit representation generator bundle SHA-256 mismatch",
    )
    dataset_root = Path(data_audit["dataset_root"]).resolve()
    train_path = dataset_root / "train.txt"
    test_path = dataset_root / "test.txt"

    train_ids, train_duplicates = read_split(train_path, "train split", add_error)
    test_ids, test_duplicates = read_split(test_path, "test split", add_error)
    train_set = set(train_ids)
    test_set = set(test_ids)
    overlap = train_set & test_set
    if overlap:
        add_error("train/test splits overlap by {} ID(s)".format(len(overlap)))
    expected_ids = train_set | test_set

    if data_audit.get("train_count") != len(train_ids):
        add_error("data_audit train_count does not match train.txt")
    if data_audit.get("test_count") != len(test_ids):
        add_error("data_audit test_count does not match test.txt")
    if data_audit.get("overlap_count") != len(overlap):
        add_error("data_audit overlap_count does not match the splits")
    if data_audit.get("train_sha256") != sha256_file(train_path):
        add_error("data_audit train_sha256 does not match train.txt")
    if data_audit.get("test_sha256") != sha256_file(test_path):
        add_error("data_audit test_sha256 does not match test.txt")

    manifest_path = reports / "cache_manifest.jsonl"
    manifest, manifest_count, manifest_duplicates = load_manifest(manifest_path, add_error)
    manifest_ids = set(manifest)
    if manifest_ids != expected_ids:
        add_error(
            "manifest ID set mismatch: missing={} extra={}".format(
                len(expected_ids - manifest_ids), len(manifest_ids - expected_ids)
            )
        )

    cache_is_symlink = cache.is_symlink()
    if cache_is_symlink:
        add_error("run-local REL+ cache must not be a symlink")
    cache_tree_symlink_free = not cache_is_symlink
    if cache.is_dir() and not cache_is_symlink:
        for descendant in cache.rglob("*"):
            if descendant.is_symlink():
                cache_tree_symlink_free = False
                add_error("REL+ cache contains symlink: {}".format(descendant))
    png_paths = list(cache.rglob("*.png")) if cache.is_dir() and not cache_is_symlink else []
    png_ids = set()
    for path in png_paths:
        relative = path.relative_to(cache).as_posix()
        png_ids.add(relative[:-4])
    if len(png_ids) != len(png_paths):
        add_error("cache contains duplicate canonical PNG paths")
    if png_ids != expected_ids:
        add_error(
            "PNG ID set mismatch: missing={} extra={}".format(
                len(expected_ids - png_ids), len(png_ids - expected_ids)
            )
        )

    manifest_valid_pixels = 0
    manifest_invalid_pixels = 0
    status_counts = {"generated": 0, "skipped": 0}
    manifest_rows_valid = True
    for sample_id, row in manifest.items():
        status = row.get("status")
        if status not in status_counts:
            manifest_rows_valid = False
            add_error("manifest {!r} has invalid status {!r}".format(sample_id, status))
        else:
            status_counts[status] += 1
        valid_pixels = row.get("valid_pixels")
        invalid_pixels = row.get("invalid_pixels")
        if (
            not isinstance(valid_pixels, int)
            or isinstance(valid_pixels, bool)
            or not isinstance(invalid_pixels, int)
            or isinstance(invalid_pixels, bool)
            or valid_pixels < 0
            or invalid_pixels < 0
            or valid_pixels + invalid_pixels != PIXELS_PER_IMAGE
        ):
            manifest_rows_valid = False
            add_error("manifest {!r} has inconsistent pixel counts".format(sample_id))
        else:
            manifest_valid_pixels += valid_pixels
            manifest_invalid_pixels += invalid_pixels
        for key, expected in (
            ("representation_version", RELPLUS_SPEC["representation_version"]),
            ("representation_spec_sha256", RELPLUS_SPEC_SHA256),
            ("representation_generator_sha256", actual_generator_sha256),
            (
                "representation_generator_bundle_sha256",
                actual_generator_bundle_sha256,
            ),
        ):
            if row.get(key) != expected:
                manifest_rows_valid = False
                if key.startswith("representation_generator"):
                    representation_generator_valid = False
                else:
                    representation_semantics_valid = False
                add_error("manifest {!r} {} mismatch".format(sample_id, key))

    with open(reports / "relplus_statistics.json", "r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    statistics_counts_consistent = True

    def check_statistics(condition, message):
        nonlocal statistics_counts_consistent
        if not condition:
            statistics_counts_consistent = False
            add_error(message)

    sample_count = len(expected_ids)
    check_statistics(statistics.get("sample_count") == sample_count, "statistics sample_count mismatch")
    check_statistics(
        statistics.get("generated_count") == status_counts["generated"],
        "statistics generated_count mismatch",
    )
    check_statistics(
        statistics.get("skipped_count") == status_counts["skipped"],
        "statistics skipped_count mismatch",
    )
    check_statistics(
        status_counts["generated"] + status_counts["skipped"] == sample_count,
        "manifest status counts do not cover the full split",
    )
    check_statistics(
        statistics.get("invalid_pixels") == manifest_invalid_pixels,
        "statistics invalid_pixels mismatch",
    )
    total_pixels = manifest_valid_pixels + manifest_invalid_pixels
    expected_invalid_ratio = (
        float(manifest_invalid_pixels) / total_pixels if total_pixels else 0.0
    )
    ratio = statistics.get("invalid_ratio")
    check_statistics(
        isinstance(ratio, (int, float))
        and math.isfinite(float(ratio))
        and math.isclose(float(ratio), expected_invalid_ratio, rel_tol=0.0, abs_tol=1e-15),
        "statistics invalid_ratio mismatch",
    )
    check_statistics(
        statistics.get("channel_order") == list(CHANNEL_ORDER),
        "statistics channel_order mismatch",
    )
    for key, expected in (
        ("normal_radius_native_pixels", RELPLUS_SPEC["normal_radius_native_pixels"]),
        ("alpha_degrees", RELPLUS_SPEC["alpha_degrees"]),
        ("lambda", RELPLUS_SPEC["lambda"]),
    ):
        check_statistics(
            statistics.get(key) == expected,
            "statistics {} mismatch".format(key),
        )
    for key in (
        "representation_semantics",
        "representation_version",
        "point_frame",
        "translation_in_red_loa",
    ):
        matches = statistics.get(key) == RELPLUS_SPEC[key]
        check_statistics(matches, "statistics {} mismatch".format(key))
        if not matches:
            representation_semantics_valid = False
    for key, expected in (
        ("representation_spec_sha256", RELPLUS_SPEC_SHA256),
        ("representation_generator_sha256", actual_generator_sha256),
    ):
        matches = statistics.get(key) == expected
        check_statistics(matches, "statistics {} mismatch".format(key))
        if not matches:
            representation_semantics_valid = False
    generator_files_match = (
        statistics.get("representation_generator_files_sha256")
        == actual_generator_files_sha256
    )
    check_statistics(
        generator_files_match,
        "statistics representation generator file hashes mismatch",
    )
    check_generator(
        generator_files_match,
        "statistics representation generator file hashes mismatch",
    )
    generator_bundle_matches = (
        statistics.get("representation_generator_bundle_sha256")
        == actual_generator_bundle_sha256
    )
    check_statistics(
        generator_bundle_matches,
        "statistics representation generator bundle SHA-256 mismatch",
    )
    check_generator(
        generator_bundle_matches,
        "statistics representation generator bundle SHA-256 mismatch",
    )
    all_cache_entries_generated = (
        status_counts["generated"] == sample_count and status_counts["skipped"] == 0
    )
    check_statistics(
        all_cache_entries_generated,
        "REL-default cache must be generated fresh with no skipped entries",
    )
    for channel in CHANNEL_ORDER:
        channel_stats = statistics.get(channel)
        check_statistics(
            isinstance(channel_stats, dict)
            and channel_stats.get("valid_pixels") == manifest_valid_pixels,
            "statistics {} valid_pixels mismatch".format(channel),
        )

    cv2.setNumThreads(1)
    tasks = []
    for sample_id in sorted(expected_ids):
        row = manifest.get(sample_id, {})
        tasks.append((sample_id, cache / (sample_id + ".png"), row.get("sha256")))

    sha_matches = 0
    decodable = 0
    shape_dtype_valid = 0
    validated_files = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(validate_image, tasks), 1):
            sha_matches += int(result["sha256_match"])
            decodable += int(result["decodable"])
            shape_dtype_valid += int(result["shape_dtype_valid"])
            validated_files += int(result["validated"])
            if result["error"]:
                add_error("{}: {}".format(result["sample_id"], result["error"]))
            if args.progress_every and (index % args.progress_every == 0 or index == len(tasks)):
                print(
                    "cache validation {}/{} validated={}".format(
                        index, len(tasks), validated_files
                    ),
                    flush=True,
                )

    split_unique = train_duplicates == 0 and test_duplicates == 0
    split_disjoint = not overlap
    manifest_set_matches = manifest_ids == expected_ids and manifest_duplicates == 0
    png_set_matches = png_ids == expected_ids and len(png_ids) == len(png_paths)
    all_sha256_match = sha_matches == sample_count
    all_decodable = decodable == sample_count
    all_shape_dtype_valid = shape_dtype_valid == sample_count
    passed = (
        failure_count[0] == 0
        and split_unique
        and split_disjoint
        and manifest_set_matches
        and png_set_matches
        and manifest_rows_valid
        and statistics_counts_consistent
        and representation_semantics_valid
        and representation_generator_valid
        and cache_tree_symlink_free
        and all_cache_entries_generated
        and validated_files == sample_count
    )
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.time() - started,
        "run_dir": str(run),
        "dataset_root": str(dataset_root),
        "train_count": len(train_ids),
        "test_count": len(test_ids),
        "sample_count": sample_count,
        "manifest_count": manifest_count,
        "png_count": len(png_paths),
        "cache_is_symlink": cache_is_symlink,
        "cache_tree_symlink_free": cache_tree_symlink_free,
        "validated_files": validated_files,
        "sha256_matches": sha_matches,
        "decodable_files": decodable,
        "shape_dtype_valid_files": shape_dtype_valid,
        "split_unique": split_unique,
        "split_disjoint": split_disjoint,
        "manifest_set_matches": manifest_set_matches,
        "png_set_matches": png_set_matches,
        "manifest_rows_valid": manifest_rows_valid,
        "statistics_counts_consistent": statistics_counts_consistent,
        "representation_semantics_valid": representation_semantics_valid,
        "representation_semantics": RELPLUS_SPEC["representation_semantics"],
        "representation_version": RELPLUS_SPEC["representation_version"],
        "representation_spec_sha256": actual_spec_sha256,
        "representation_generator_sha256": actual_generator_sha256,
        "representation_generator_files_sha256": actual_generator_files_sha256,
        "representation_generator_bundle_sha256": actual_generator_bundle_sha256,
        "representation_generator_valid": representation_generator_valid,
        "all_cache_entries_generated": all_cache_entries_generated,
        "all_sha256_match": all_sha256_match,
        "all_decodable": all_decodable,
        "all_shape_dtype_valid": all_shape_dtype_valid,
        "manifest_valid_pixels": manifest_valid_pixels,
        "manifest_invalid_pixels": manifest_invalid_pixels,
        "failure_count": failure_count[0],
        "errors_truncated": failure_count[0] > len(errors),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    report_path = args.run_dir.resolve() / "data_reports" / "cache_validation.json"
    exit_path = args.run_dir.resolve() / "status" / "cache_validation.exitcode"
    try:
        report = validate(args)
    except Exception as error:
        report = {
            "status": "failed",
            "exit_code": 1,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_dir": str(args.run_dir.resolve()),
            "failure_count": 1,
            "errors_truncated": False,
            "errors": ["{}: {}".format(type(error).__name__, error)],
            "sample_count": 0,
            "manifest_count": 0,
            "png_count": 0,
            "validated_files": 0,
            "all_sha256_match": False,
            "all_decodable": False,
            "all_shape_dtype_valid": False,
        }
    atomic_write_json(report_path, report)
    atomic_write_text(exit_path, "{}\n".format(report["exit_code"]))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
