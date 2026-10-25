#!/usr/bin/env python3
"""CPU checks before polar crops from ``make_polar_crops.py`` go into training.

``geometry``: the cross-projection check of ``cross_projection.py`` on random
training-area panoramas, at the Phase 0 setting (pitch -20/0/20, FOV 60, 480 px,
panorama 1024x2048) and at polar pitches (-80/-60/60/80). It keeps every failure
message, i.e. which quantity (height, radius, EGVIA angle, LOA) missed its 2 cm /
2 degree median, which the Phase 0 report did not.

``reproduce``: for random training pinholes, find the rotation between the dataset
world frame and the frame of the panorama taken at the same camera location (yaw
grid, then a Nelder-Mead refine of yaw and two tilt angles, matching ranges),
render the pinhole's own view from the panorama with ``make_polar_crops.render``'s
sampling, and compare with the dataset frame. It also reports the panorama's tilt
against the dataset gravity, which ``make_polar_crops.py`` assumes is small.

``sheet``: a contact sheet of real crops from a crop folder, label colours over RGB.

Pre-set rules (``decide_*``): a reproduced frame passes when its depth median
relative error is <= 1% and its stored labels agree on >= 95% of the pixels
labelled in both; ``reproduce`` passes when >= 90% of frames pass. The level
assumption holds when the tilt p95 is <= 1 degree. RGB differences are reported
only (the dataset frames were not rendered from the panoramas, so exposure may
differ).
"""

import argparse
import csv
import json
import multiprocessing
from pathlib import Path

import cv2
import numpy as np

import cross_projection as cp
import eval_pano_transfer as ev
import make_polar_crops as mpc


PHASE0_PITCHES = (-20.0, 0.0, 20.0)
POLAR_PITCHES = (-80.0, -60.0, 60.0, 80.0)
GEOMETRY_YAWS = (0.0, 90.0, 180.0, 270.0)
DEPTH_REL_ERROR_MAX = 0.01
LABEL_AGREEMENT_MIN = 0.95
FRAMES_PASSING_MIN = 0.90
TILT_P95_MAX_DEG = 1.0
FIT_PIXELS = 4000
RELATIVE_ERROR_CLIP = 0.2
MIRROR_X = np.diag([-1.0, 1.0, 1.0])
_JOB = {}


# ------------------------------------------------------------ geometry


def geometry_panorama(sample):
    height, width = _JOB["erp_size"]
    raw = cv2.imread(str(sample["depth"]), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16:
        raise ValueError("{} is not a uint16 depth map".format(sample["depth"]))
    raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)
    result = {"sample_id": sample["sample_id"]}
    for name, pitches in (("phase0", PHASE0_PITCHES), ("polar", POLAR_PITCHES)):
        report = cp.check_panorama(
            raw, _JOB["frozen"], yaws=GEOMETRY_YAWS, pitches=pitches, size=480,
            fov=_JOB["fov"], thresholds=cp.DEFAULT_THRESHOLDS,
        )
        result[name] = {
            "status": report["status"],
            "failures": report["failures"],
            "erp_estimated_gravity_rotation_deg": report["erp_estimated_gravity_rotation_deg"],
            "crops": [{"yaw_deg": c["yaw_deg"], "pitch_deg": c["pitch_deg"],
                       "compared_pixels": c["compared_pixels"], "raw": c["raw"]} for c in report["crops"]],
        }
    return result


def summarize_geometry(results):
    summary = {}
    for name in ("phase0", "polar"):
        failing = {}
        medians = {}
        for result in results:
            for crop in result[name]["crops"]:
                for quantity, stats in crop["raw"].items():
                    medians.setdefault(quantity, []).append(stats["median"])
            for message in result[name]["failures"]:
                quantity = message.split(": ", 1)[1].split(" median", 1)[0]
                failing[quantity] = failing.get(quantity, 0) + 1
        summary[name] = {
            "failing_panoramas": sum(r[name]["status"] == "FAIL" for r in results),
            "panoramas": len(results),
            "failing_crops_by_quantity": failing,
            "crop_median_by_quantity_p50_p95": {
                q: None if any(v is None for v in values)
                else [float(np.percentile(values, 50)), float(np.percentile(values, 95))]
                for q, values in medians.items()
            },
        }
    return summary


# ------------------------------------------------------------ reproduce


def rotation(yaw_deg, tilt_x_deg, tilt_y_deg):
    a, b, c = np.radians([yaw_deg, tilt_x_deg, tilt_y_deg])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    ry = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
    return rz @ rx @ ry


def range_cost(range_erp, directions, ranges, world_to_erp):
    rows, columns = cp.erp_indices_for_directions(directions @ world_to_erp.T, *range_erp.shape)
    sampled = range_erp[rows, columns]
    error = np.where(sampled > 0, np.abs(sampled - ranges) / ranges, RELATIVE_ERROR_CLIP)
    return float(np.mean(np.minimum(error, RELATIVE_ERROR_CLIP)))


def fit_world_to_erp(range_erp, directions, ranges):
    """Rotation (times an optional x mirror) taking dataset-world directions into the panorama frame."""
    from scipy.optimize import minimize

    best = None
    for mirror_name, mirror in (("none", np.eye(3)), ("x", MIRROR_X)):
        grid = [(range_cost(range_erp, directions, ranges, rotation(yaw, 0, 0) @ mirror), yaw)
                for yaw in np.arange(0.0, 360.0, 0.5)]
        cost, yaw = min(grid)
        refined = minimize(
            lambda p: range_cost(range_erp, directions, ranges, rotation(*p) @ mirror),
            x0=[yaw, 0.0, 0.0], method="Nelder-Mead",
            options={"initial_simplex": [[yaw, 0, 0], [yaw + 0.5, 0, 0], [yaw, 0.5, 0], [yaw, 0, 0.5]],
                     "xatol": 0.01, "fatol": 1e-6, "maxiter": 400},
        )
        candidate = (min(cost, refined.fun), mirror_name,
                     refined.x if refined.fun <= cost else np.array([yaw, 0.0, 0.0]), mirror)
        if best is None or candidate[0] < best[0]:
            best = candidate
    cost, mirror_name, params, mirror = best
    return rotation(*params) @ mirror, {"cost": cost, "mirror": mirror_name,
                                        "yaw_deg": float(params[0]),
                                        "tilt_x_deg": float(params[1]), "tilt_y_deg": float(params[2])}


def panorama_for(stanford_root, row):
    stem = "camera_{}_{}_frame_equirectangular_domain".format(row["camera"], row["room"])
    pano = Path(stanford_root) / row["area"] / "pano"
    return {
        "sample_id": "{}/{}".format(row["area"], stem),
        "rgb": pano / "rgb" / (stem + "_rgb.png"),
        "depth": pano / "depth" / (stem + "_depth.png"),
        "semantic": pano / "semantic" / (stem + "_semantic.png"),
    }


def reproduce_frame(row):
    frozen = _JOB["frozen_loader"]
    pano = panorama_for(_JOB["stanford_root"], row)
    result = {"sample_id": row["sample_id"], "panorama": pano["sample_id"]}
    if not all(pano[k].is_file() for k in ("rgb", "depth", "semantic")):
        return dict(result, status="NO_PANORAMA")
    raw, camera, _ = frozen["load_canonical_frame"](
        row["depth_path"], row["camera_metadata_path"], dataset_profile=frozen["profile"]
    )
    z_pin, valid_pin = frozen["decode_depth"](raw)
    size = raw.shape[0]
    k_json = np.asarray(camera.K_json, dtype=np.float64)
    world_from_camera = np.asarray(camera.R_world_to_camera, dtype=np.float64).T
    rays_camera, rays_world = cp.crop_rays(size, k_json, world_from_camera)
    ranges = z_pin * np.linalg.norm(rays_camera, axis=2)

    range_erp = cp.decode_erp_depth(cv2.imread(str(pano["depth"]), cv2.IMREAD_UNCHANGED))
    rgb_erp = cv2.imread(str(pano["rgb"]), cv2.IMREAD_COLOR)
    label_erp = mpc.stored_labels(cv2.imread(str(pano["semantic"]), cv2.IMREAD_COLOR), _JOB["lookup"])
    pick = np.flatnonzero(valid_pin.ravel())
    if pick.size < 100:
        return dict(result, status="NO_DEPTH")
    pick = np.random.default_rng(0).choice(pick, min(FIT_PIXELS, pick.size), replace=False)
    world_to_erp, fit = fit_world_to_erp(
        range_erp, rays_world.reshape(-1, 3)[pick], ranges.ravel()[pick].astype(np.float64)
    )
    up = world_to_erp @ np.array([0.0, 0.0, 1.0])

    erp_dirs = rays_world @ world_to_erp.T
    rows, columns = cp.erp_indices_for_directions(erp_dirs, *range_erp.shape)
    range_crop = range_erp[rows, columns]
    both = valid_pin & (range_crop > 0)
    depth_error = np.abs(range_crop - ranges)[both] / ranges[both]
    rgb_crop = mpc.sample_rgb(rgb_erp, erp_dirs)
    rows, columns = cp.erp_indices_for_directions(erp_dirs, *label_erp.shape)
    label_crop = label_erp[rows, columns]

    rgb_pin = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
    label_pin = cv2.imread(row["label_path"], cv2.IMREAD_GRAYSCALE)
    if rgb_pin.shape[:2] != (size, size) or label_pin.shape != (size, size):
        raise ValueError("{}: RGB/label are not {}x{}".format(row["sample_id"], size, size))
    labelled = (label_pin != 0) & (label_pin != 255) & (label_crop != 0)
    rgb_diff = np.abs(rgb_pin.astype(np.float64) - rgb_crop.astype(np.float64))
    correlation = [float(np.corrcoef(rgb_pin[..., c].ravel(), rgb_crop[..., c].ravel())[0, 1]) for c in range(3)]
    depth_median = float(np.median(depth_error)) if depth_error.size else None
    agreement = float(np.mean(label_pin[labelled] == label_crop[labelled])) if labelled.any() else None
    passed = (depth_median is not None and depth_median <= DEPTH_REL_ERROR_MAX
              and agreement is not None and agreement >= LABEL_AGREEMENT_MIN)
    if _JOB["examples"] is not None and _JOB["example_ids"].get(row["sample_id"]) is not None:
        agree = np.where(labelled, np.where(label_pin == label_crop, 255, 0), 128).astype(np.uint8)
        side = np.concatenate([rgb_pin, rgb_crop, cv2.cvtColor(agree, cv2.COLOR_GRAY2BGR)], axis=1)
        name = "{:02d}_{}.png".format(_JOB["example_ids"][row["sample_id"]], row["sample_id"].replace("/", "__"))
        cv2.imwrite(str(_JOB["examples"] / name), side)
    return dict(
        result,
        status="PASS" if passed else "FAIL",
        fit=fit,
        tilt_deg=float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))),
        pinhole_pitch_deg=float(np.degrees(np.arcsin(np.clip(world_from_camera[2, 2], -1.0, 1.0)))),
        depth_relative_error_p50=depth_median,
        depth_relative_error_p95=float(np.percentile(depth_error, 95)) if depth_error.size else None,
        depth_compared_pixels=int(both.sum()),
        label_agreement=agreement,
        label_compared_pixels=int(labelled.sum()),
        rgb_mean_abs_diff=float(rgb_diff.mean()),
        rgb_channel_correlation=correlation,
    )


def decide_reproduce(results):
    checked = [r for r in results if r["status"] in ("PASS", "FAIL")]
    tilts = [r["tilt_deg"] for r in checked]
    passing = sum(r["status"] == "PASS" for r in checked)
    share = passing / len(checked) if checked else 0.0

    def p(key, q):
        values = [r[key] for r in checked if r[key] is not None]
        return float(np.percentile(values, q)) if values else None

    return {
        "rule": "frame: depth p50 rel error <= {:g} and label agreement >= {:g}; "
                "overall: >= {:g} of frames; level: tilt p95 <= {:g} deg".format(
                    DEPTH_REL_ERROR_MAX, LABEL_AGREEMENT_MIN, FRAMES_PASSING_MIN, TILT_P95_MAX_DEG),
        "frames": len(results),
        "frames_checked": len(checked),
        "frames_without_panorama_or_depth": len(results) - len(checked),
        "frames_passing": passing,
        "reproduce": "PASS" if checked and share >= FRAMES_PASSING_MIN else "FAIL",
        "level_assumption": "OK" if tilts and np.percentile(tilts, 95) <= TILT_P95_MAX_DEG else "ALIGN_NEEDED",
        "mirror_fits": sorted({r["fit"]["mirror"] for r in checked}),
        "tilt_deg_p50_p95_max": [p("tilt_deg", 50), p("tilt_deg", 95), p("tilt_deg", 100)],
        "depth_relative_error_p50_median": p("depth_relative_error_p50", 50),
        "label_agreement_p5_p50": [p("label_agreement", 5), p("label_agreement", 50)],
        "rgb_mean_abs_diff_p50_p95": [p("rgb_mean_abs_diff", 50), p("rgb_mean_abs_diff", 95)],
    }


# ---------------------------------------------------------------- sheet


def contact_sheet(crop_root, count, columns, tile):
    report = json.loads((crop_root / "crops.json").read_text(encoding="utf-8"))
    crops = report["crops"]
    chosen = [crops[i] for i in np.linspace(0, len(crops) - 1, min(count, len(crops))).astype(int)]
    palette = np.random.default_rng(7).integers(40, 255, (mpc.NUM_STORED_CLASSES, 3)).astype(np.uint8)
    palette[0] = 0
    tiles = []
    for crop in chosen:
        rgb = cv2.imread(str(crop_root / "RGB" / (crop["sample_id"] + ".png")), cv2.IMREAD_COLOR)
        label = cv2.imread(str(crop_root / "Label" / (crop["sample_id"] + ".png")), cv2.IMREAD_GRAYSCALE)
        blend = np.where((label > 0)[..., None], (0.6 * rgb + 0.4 * palette[label]).astype(np.uint8), rgb)
        both = cv2.resize(np.concatenate([rgb, blend], axis=1), (2 * tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(both, "pitch {:+.0f} fov {:.0f}".format(crop["pitch_deg"], crop["hfov_deg"]), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(both)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    grid = [np.concatenate(tiles[i: i + columns], axis=1) for i in range(0, len(tiles), columns)]
    return np.concatenate(grid, axis=0)


# ----------------------------------------------------------------- main


def run(function, items, workers):
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            return pool.map(function, items, chunksize=1)
    return [function(item) for item in items]


def read_train_rows(manifest):
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "train"
                and row["area"] in mpc.TRAIN_AREAS]


def import_loader(source_root):
    cp.import_frozen_source(source_root)
    from rel_plus.depth import decode_stanford_s2d_depth
    from rel_plus.profiles import STANFORD_S2D_PROFILE
    from rel_plus.stanford_s2d import load_canonical_frame

    return {"load_canonical_frame": load_canonical_frame, "profile": STANFORD_S2D_PROFILE,
            "decode_depth": decode_stanford_s2d_depth}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("geometry", "reproduce", "sheet"))
    parser.add_argument("--output", required=True, type=Path, help="JSON report (sheet: PNG)")
    parser.add_argument("--stanford-root", type=Path, help="geometry, reproduce")
    parser.add_argument("--semantic-labels", type=Path, help="reproduce")
    parser.add_argument("--manifest", type=Path, help="reproduce: frozen full_manifest.csv")
    parser.add_argument("--crop-root", type=Path, help="sheet: output of make_polar_crops.py")
    parser.add_argument("--areas", nargs="+", default=list(mpc.TRAIN_AREAS), help="geometry")
    parser.add_argument("--source-root", type=Path, default=cp.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--count", type=int, default=None,
                        help="panoramas (geometry, default 10), frames (reproduce, default 100), crops (sheet, 24)")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--fov", type=float, default=60.0, help="geometry; Phase 0 used 60")
    parser.add_argument("--erp-size", type=int, nargs=2, default=(1024, 2048), help="geometry")
    parser.add_argument("--examples", type=int, default=8, help="reproduce: side-by-side PNGs to write")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    if args.command == "sheet":
        if args.crop_root is None:
            parser.error("sheet needs --crop-root")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output), contact_sheet(args.crop_root, args.count or 24, 4, 240))
        print(args.output)
        return 0

    if args.stanford_root is None:
        parser.error("{} needs --stanford-root".format(args.command))
    _JOB["stanford_root"] = args.stanford_root
    if args.command == "geometry":
        _JOB.update(frozen=cp.import_frozen_source(args.source_root), fov=args.fov, erp_size=tuple(args.erp_size))
        if set(args.areas) - set(mpc.TRAIN_AREAS):
            parser.error("only training areas are allowed")
        panoramas = ev.list_panoramas(args.stanford_root, args.areas)
        chosen = rng.choice(len(panoramas), min(args.count or 10, len(panoramas)), replace=False)
        results = run(geometry_panorama, [panoramas[i] for i in sorted(chosen)], args.workers)
        report = {"settings": {"fov_deg": args.fov, "crop_size": 480, "erp_size": list(args.erp_size),
                               "yaws": GEOMETRY_YAWS, "phase0_pitches": PHASE0_PITCHES,
                               "polar_pitches": POLAR_PITCHES, "thresholds": cp.DEFAULT_THRESHOLDS},
                  "summary": summarize_geometry(results), "panoramas": results}
    else:
        if args.manifest is None or args.semantic_labels is None:
            parser.error("reproduce needs --manifest and --semantic-labels")
        rows = read_train_rows(args.manifest)
        chosen = [rows[i] for i in sorted(rng.choice(len(rows), min(args.count or 100, len(rows)), replace=False))]
        examples = args.output.with_name(args.output.stem + "_examples") if args.examples else None
        if examples is not None:
            examples.mkdir(parents=True, exist_ok=True)
        _JOB.update(frozen_loader=import_loader(args.source_root), lookup=ev.label_lookup(args.semantic_labels),
                    examples=examples,
                    example_ids={row["sample_id"]: i for i, row in enumerate(chosen[: args.examples])})
        results = run(reproduce_frame, chosen, args.workers)
        report = {"decision": decide_reproduce(results), "frames": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"] if args.command == "geometry" else report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
