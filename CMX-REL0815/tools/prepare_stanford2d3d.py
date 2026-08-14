#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time
import warnings
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from tqdm import tqdm


AREAS = ("area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b", "area_6")
TRAIN_AREAS = {"area_1", "area_2", "area_3", "area_4", "area_6"}
TEST_AREAS = {"area_5a", "area_5b"}
EXPECTED_COUNTS = {
    "area_1": 10327,
    "area_2": 15714,
    "area_3": 3704,
    "area_4": 13268,
    "area_5a": 6261,
    "area_5b": 11332,
    "area_6": 9890,
}
SUFFIXES = {
    "rgb": "_domain_rgb.png",
    "depth": "_domain_depth.png",
    "semantic": "_domain_semantic.png",
    "pose": "_domain_pose.json",
}
OUTPUT_DIRS = {
    "rgb": "RGB",
    "depth": "Depth16",
    "semantic": "Label",
    "pose": "Pose",
}
EXPECTED_CLASSES = [
    "<UNK>", "beam", "board", "bookcase", "ceiling", "chair", "clutter",
    "column", "door", "floor", "sofa", "table", "wall", "window",
]

_GET_HHA = None


def atomic_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(str(tmp), str(path))


def atomic_image(path, image, compression=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
    if not cv2.imwrite(str(tmp), image, params):
        raise RuntimeError("failed to write {}".format(tmp))
    os.replace(str(tmp), str(path))


def load_label_lut(labels_path):
    labels = json.loads(Path(labels_path).read_text())
    classes = []
    class_ids = []
    for label in labels:
        class_name = label.split("_", 1)[0]
        if class_name not in classes:
            classes.append(class_name)
        class_ids.append(classes.index(class_name))
    if classes != EXPECTED_CLASSES:
        raise ValueError("unexpected semantic classes: {}".format(classes))
    return np.asarray(class_ids, dtype=np.uint8), classes


def parse_member(member_name, expected_area):
    parts = PurePosixPath(member_name).parts
    if len(parts) != 4 or parts[0] != expected_area or parts[1] != "data":
        return None
    modality = parts[2]
    suffix = SUFFIXES.get(modality)
    filename = parts[3]
    if suffix is None or not filename.endswith(suffix):
        return None
    return modality, filename[:-len(suffix)]


def decode(payload, mode):
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), mode)
    if image is None:
        raise ValueError("OpenCV could not decode an archive member")
    return image


def save_rgb(payload, path, overwrite):
    if path.exists() and not overwrite:
        return False
    rgb = decode(payload, cv2.IMREAD_COLOR)
    if rgb.shape != (1080, 1080, 3) or rgb.dtype != np.uint8:
        raise ValueError("unexpected RGB shape or dtype: {} {}".format(rgb.shape, rgb.dtype))
    atomic_image(path, cv2.resize(rgb, (480, 480), interpolation=cv2.INTER_LINEAR))
    return True


def save_depth(payload, depth_path, raw_path, overwrite):
    if depth_path.exists() and raw_path.exists() and not overwrite:
        return False
    depth = decode(payload, cv2.IMREAD_UNCHANGED)
    if depth.shape != (1080, 1080) or depth.dtype != np.uint16:
        raise ValueError("unexpected depth shape or dtype: {} {}".format(depth.shape, depth.dtype))
    if overwrite or not depth_path.exists():
        atomic_bytes(depth_path, payload)
    if overwrite or not raw_path.exists():
        # The source uses 1/512 m units and 65535 for missing pixels. Adding one
        # wraps missing pixels to zero; the high byte is a linear 8-bit sensor-range encoding.
        raw_depth = ((depth + np.uint16(1)) >> np.uint16(8)).astype(np.uint8)
        raw_depth = cv2.resize(raw_depth, (480, 480), interpolation=cv2.INTER_NEAREST)
        atomic_image(raw_path, raw_depth)
    return True


def save_label(payload, path, label_lut, overwrite):
    if path.exists() and not overwrite:
        return False
    semantic = decode(payload, cv2.IMREAD_COLOR)
    if semantic.shape != (1080, 1080, 3) or semantic.dtype != np.uint8:
        raise ValueError("unexpected semantic shape or dtype: {} {}".format(semantic.shape, semantic.dtype))
    indices = (
        semantic[:, :, 0].astype(np.uint32)
        + semantic[:, :, 1].astype(np.uint32) * 256
        + semantic[:, :, 2].astype(np.uint32) * 65536
    )
    label = np.zeros(indices.shape, dtype=np.uint8)
    valid = indices < len(label_lut)
    label[valid] = label_lut[indices[valid]]
    label = cv2.resize(label, (480, 480), interpolation=cv2.INTER_NEAREST)
    atomic_image(path, label)
    return True


def extract_area(tar_path, output_root, area, label_lut, overwrite, sample_filter):
    found = {modality: set() for modality in SUFFIXES}
    written = {modality: 0 for modality in SUFFIXES}
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parsed = parse_member(member.name, area)
            if parsed is None:
                continue
            modality, sample = parsed
            if sample_filter and sample != sample_filter:
                continue
            if sample in found[modality]:
                raise ValueError("duplicate {} sample: {}/{}".format(modality, area, sample))
            found[modality].add(sample)

            extension = ".json" if modality == "pose" else ".png"
            output_path = output_root / OUTPUT_DIRS[modality] / area / (sample + extension)
            raw_path = output_root / "RawDepth" / area / (sample + ".png")
            already_done = output_path.exists() and not overwrite
            if modality == "depth":
                already_done = already_done and raw_path.exists()
            if already_done:
                continue

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError("could not read {}".format(member.name))
            payload = source.read()
            if modality == "rgb":
                changed = save_rgb(payload, output_path, overwrite)
            elif modality == "depth":
                changed = save_depth(payload, output_path, raw_path, overwrite)
            elif modality == "semantic":
                changed = save_label(payload, output_path, label_lut, overwrite)
            else:
                json.loads(payload.decode("utf-8"))
                atomic_bytes(output_path, payload)
                changed = True
            written[modality] += int(changed)

    reference = found["rgb"]
    mismatches = {
        modality: {
            "missing": sorted(reference - samples),
            "extra": sorted(samples - reference),
        }
        for modality, samples in found.items()
        if samples != reference
    }
    if mismatches:
        raise ValueError("modality mismatch in {}: {}".format(area, mismatches))
    if not sample_filter and len(reference) != EXPECTED_COUNTS[area]:
        raise ValueError(
            "{} has {} samples, expected {}".format(area, len(reference), EXPECTED_COUNTS[area])
        )
    return sorted(reference), written


def write_split(path, entries):
    path.write_text("\n".join(entries) + "\n")


def extract_dataset(args):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    label_lut, classes = load_label_lut(args.semantic_labels)
    (output_root / "class_mapping.json").write_text(json.dumps(
        {"stored_ids": dict(enumerate(classes)), "loader_transform": "stored label - 1; 0 becomes 255"},
        indent=2,
    ) + "\n")

    manifest = {"areas": {}, "classes": classes, "sample_filter": args.sample}
    area_samples = {}
    for area in args.areas:
        tar_path = Path(args.tar_root) / (area + "_no_xyz.tar")
        if not tar_path.is_file():
            raise FileNotFoundError(tar_path)
        samples, written = extract_area(
            tar_path, output_root, area, label_lut, args.overwrite, args.sample,
        )
        area_samples[area] = samples
        manifest["areas"][area] = {"samples": len(samples), "written": written}
        print("{} samples={} written={}".format(area, len(samples), written), flush=True)

    if tuple(args.areas) == AREAS and args.sample is None:
        train = [area + "/" + sample for area in AREAS if area in TRAIN_AREAS for sample in area_samples[area]]
        test = [area + "/" + sample for area in AREAS if area in TEST_AREAS for sample in area_samples[area]]
        if len(train) != 52903 or len(test) != 17593:
            raise ValueError("unexpected split sizes: train={} test={}".format(len(train), len(test)))
        write_split(output_root / "train.txt", train)
        write_split(output_root / "test.txt", test)
        manifest["train_count"] = len(train)
        manifest["test_count"] = len(test)

    (output_root / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def init_hha(depth2hha_root):
    global _GET_HHA
    cv2.setNumThreads(0)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    sys.path.insert(0, depth2hha_root)
    from getHHA import getHHA
    _GET_HHA = getHHA


def make_hha(task):
    depth_path, pose_path, output_path = map(Path, task)
    start = time.perf_counter()
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.shape != (1080, 1080) or depth.dtype != np.uint16:
        raise ValueError("invalid depth {}".format(depth_path))
    pose = json.loads(pose_path.read_text())
    camera_matrix = np.asarray(pose["camera_k_matrix"], dtype=np.float64)
    depth_m = (depth + np.uint16(1)).astype(np.float64) / 512.0
    hha = _GET_HHA(camera_matrix, depth_m, depth_m)
    hha = cv2.resize(hha, (480, 480), interpolation=cv2.INTER_LINEAR)
    atomic_image(output_path, hha)
    return time.perf_counter() - start


def git_commit(path):
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=False,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.strip()


def generate_hha(args):
    output_root = Path(args.output_root)
    tasks = []
    total_depth = 0
    for depth_path in sorted((output_root / "Depth16").glob("area_*/*.png")):
        total_depth += 1
        relative = depth_path.relative_to(output_root / "Depth16")
        pose_path = output_root / "Pose" / relative.with_suffix(".json")
        output_path = output_root / "HHA" / relative
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        if args.overwrite or not output_path.is_file():
            tasks.append((str(depth_path), str(pose_path), str(output_path)))
    expected_total = 1 if args.sample else 70496
    if total_depth != expected_total:
        raise ValueError("found {} depth images, expected {}".format(total_depth, expected_total))

    start = time.perf_counter()
    worker_seconds = 0.0
    if tasks:
        with mp.Pool(args.workers, initializer=init_hha, initargs=(args.depth2hha_root,)) as pool:
            for seconds in tqdm(pool.imap_unordered(make_hha, tasks, chunksize=1), total=len(tasks)):
                worker_seconds += seconds

    final_count = sum(1 for _ in (output_root / "HHA").glob("area_*/*.png"))
    if final_count != expected_total:
        raise ValueError("generated {} HHA images, expected {}".format(final_count, expected_total))
    reference_root = Path(args.depth2hha_root)
    manifest = {
        "generated": len(tasks),
        "final_count": final_count,
        "workers": args.workers,
        "wall_seconds": time.perf_counter() - start,
        "worker_seconds": worker_seconds,
        "algorithm_root": str(reference_root.resolve()),
        "algorithm_commit": git_commit(reference_root),
        "method": "Depth2HHA at 1080x1080 using camera_k_matrix, then bilinear resize to 480x480",
        "depth_units": "source uint16 plus one, divided by 512 metres",
    }
    (output_root / "hha_generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "hha"))
    parser.add_argument(
        "--tar-root", default="/data/zhuzhaoziao/datasets/Stanford2D3D/no_xyz",
    )
    parser.add_argument(
        "--output-root", default="/data/zhuzhaoziao/cmx/datasets/Stanford2D3D_480",
    )
    parser.add_argument(
        "--semantic-labels",
        default="/data/zhuzhaoziao/cmx/raw/reference_repos/2D-3D-Semantics/assets/semantic_labels.json",
    )
    parser.add_argument(
        "--depth2hha-root",
        default="/data/zhuzhaoziao/cmx/raw/reference_repos/Depth2HHA-python",
    )
    parser.add_argument("--areas", nargs="+", choices=AREAS, default=list(AREAS))
    parser.add_argument("--sample", help="Prepare one basename for a non-production sanity check")
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.stage == "extract":
        extract_dataset(args)
    else:
        if args.sample is None and tuple(args.areas) != AREAS:
            raise ValueError("HHA production requires the complete extracted dataset")
        generate_hha(args)


if __name__ == "__main__":
    main()
