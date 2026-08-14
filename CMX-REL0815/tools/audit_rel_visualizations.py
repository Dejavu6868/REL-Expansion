import argparse
import csv
from pathlib import Path

import cv2


def audit(root, output):
    rows = []
    for path in sorted(root.glob("*/**/*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("visualization cannot be read: {}".format(path))
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "height": image.shape[0],
                "width": image.shape[1],
                "channels": 1 if image.ndim == 2 else image.shape[2],
                "min": int(image.min()),
                "max": int(image.max()),
                "mean": float(image.mean()),
                "std": float(image.std()),
                "all_black": bool(image.max() == 0),
                "all_white": bool(image.min() == 255),
            }
        )
    if not rows:
        raise ValueError("no PNG visualizations found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.root, args.output)
    print("VISUAL_FILE_COUNT={}".format(len(rows)))
    print("ALL_BLACK_COUNT={}".format(sum(row["all_black"] for row in rows)))
    print("ALL_WHITE_COUNT={}".format(sum(row["all_white"] for row in rows)))


if __name__ == "__main__":
    main()
