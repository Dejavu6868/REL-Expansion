#!/usr/bin/env python3
"""Create the small offline fixture by executing the audited source functions."""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_sources(authoritative_root, perspective_root, hha_file):
    authoritative_root = Path(authoritative_root)
    if (authoritative_root / "third_party/rel_original/rel.py").is_file():
        sys.path.insert(0, str(authoritative_root))
        rel_module = importlib.import_module("third_party.rel_original.rel")
    elif (authoritative_root / "rel.py").is_file():
        package_root = authoritative_root.parent.parent
        sys.path.insert(0, str(package_root))
        rel_module = importlib.import_module("third_party.rel_original.rel")
    else:
        raise OSError("REL authoritative root does not contain rel.py")

    hha_spec = importlib.util.spec_from_file_location("utils.hha_util", str(hha_file))
    hha_module = importlib.util.module_from_spec(hha_spec)
    sys.modules[hha_spec.name] = hha_module
    hha_spec.loader.exec_module(hha_module)
    rgbd_path = Path(perspective_root) / "utils/rgbd_util.py"
    rgbd_spec = importlib.util.spec_from_file_location("source_rgbd_util", str(rgbd_path))
    rgbd_module = importlib.util.module_from_spec(rgbd_spec)
    rgbd_spec.loader.exec_module(rgbd_module)
    return rel_module, rgbd_module, hha_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-root", required=True)
    parser.add_argument("--perspective-root", required=True)
    parser.add_argument("--hha-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rel_module, rgbd_module, hha_module = load_sources(
        args.authoritative_root, args.perspective_root, args.hha_file
    )

    rotation_initial = np.array([0.0, 0.0, -1.0])
    rotation_final = np.array([0.3, -0.4, -0.8660254037844386])
    rotation_expected = hha_module.getRMatrix(rotation_initial, rotation_final)
    rotation_points = np.arange(3 * 4 * 3, dtype=np.float64).reshape(3, 4, 3) / 10.0
    rotated_points_expected = hha_module.rotatePC(rotation_points, rotation_expected)

    rows, columns = np.indices((9, 11), dtype=np.float64)
    normal_depth_m = (1.5 + 0.01 * columns + 0.02 * rows).astype(np.float32)
    normal_missing = np.zeros((9, 11), dtype=bool)
    normal_missing[1, 2] = True
    normal_k_json = np.array(
        [[80.0, 0.0, 5.5], [0.0, 82.0, 4.5], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    helper_k = normal_k_json.copy()
    helper_k[0, 2] += 0.5
    helper_k[1, 2] += 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        normal_expected, normal_offset_expected = rgbd_module.computeNormalsSquareSupport(
            normal_depth_m, normal_missing, 2, 1, helper_k,
            np.ones(normal_depth_m.shape, dtype=np.float32)
        )

    erows, ecolumns = np.indices((5, 8), dtype=np.float32)
    encoding_points_cm = np.stack(
        [0.2 + ecolumns, 0.3 + erows, 1.0 + 0.1 * ecolumns + 0.2 * erows], axis=-1
    )
    encoding_normals = np.stack(
        [0.2 + 0.01 * ecolumns, -0.3 + 0.02 * erows, np.ones_like(erows)], axis=-1
    )
    encoding_normals /= np.linalg.norm(encoding_normals, axis=-1, keepdims=True)
    depth_marker = np.ones((5, 8), dtype=np.float32)
    depth_marker[0, 0] = 0.0
    encoding_valid = depth_marker != 0.0
    rel_module.processDepthImage_ERP = lambda _depth, _missing: (
        encoding_points_cm.copy(), encoding_normals.copy(), [0.0, 0.0]
    )
    encoding_expected = rel_module.getREL(depth_marker, alpha=45.0, lam=0.5)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        rotation_initial=rotation_initial,
        rotation_final=rotation_final,
        rotation_expected=rotation_expected,
        rotation_points=rotation_points,
        rotated_points_expected=rotated_points_expected,
        normal_depth_m=normal_depth_m,
        normal_missing=normal_missing,
        normal_k_json=normal_k_json,
        normal_expected=normal_expected,
        normal_offset_expected=normal_offset_expected,
        encoding_points_cm=encoding_points_cm,
        encoding_normals=encoding_normals,
        encoding_valid=encoding_valid,
        encoding_expected=encoding_expected,
    )
    print("created {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
