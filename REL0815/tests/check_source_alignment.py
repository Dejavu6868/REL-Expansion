#!/usr/bin/env python3
"""Confirm that extracted numerical function bodies match their sources."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = Path("/home/zhuzhaoziao/RELPlus/REL-SF4PASS-reference")
COMPATIBILITY_HHA = Path("/data/bxh_copy/Pano_MA_Seg/utils/hha_util.py")


def function_body(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    body = function.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def compare(source, extracted, functions):
    for name in functions:
        assert function_body(source, name) == function_body(extracted, name), name
        print(f"function_body_equal={name}")


def main():
    compare(
        REFERENCE_ROOT / "getREL.py",
        PROJECT_ROOT / "rel_original" / "rel.py",
        ("getImage", "getREL"),
    )
    compare(
        REFERENCE_ROOT / "utils" / "rgbd_util.py",
        PROJECT_ROOT / "rel_original" / "rgbd_util.py",
        (
            "processDepthImage_ERP",
            "getPointCloud_ERP",
            "computeNormalsSquareSupport_ERP",
        ),
    )
    compare(
        COMPATIBILITY_HHA,
        PROJECT_ROOT / "rel_original" / "hha_util.py",
        (
            "filterItChopOff",
            "mutiplyIt",
            "invertIt",
            "getRMatrix",
            "rotatePC",
            "getGDir",
            "getGDirHelper",
        ),
    )
    print("source_function_alignment=PASS")


if __name__ == "__main__":
    main()
