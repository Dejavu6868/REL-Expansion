"""Checks for the black-RGB pixel count and the strict 8-class bounds."""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import black_pixel_check as bp  # noqa: E402
import eval_pano_transfer as ev  # noqa: E402


def confusion_of(label, prediction):
    valid = label < 13
    return np.bincount(13 * label[valid].astype(np.int64) + prediction[valid], minlength=169).reshape(13, 13)


def test_bounds_hold_the_exact_rescore():
    rng = np.random.default_rng(0)
    for trial in range(60):
        label = rng.integers(0, 14, 4000)
        label[label == 13] = 255
        # predictions right about half the time, so tp, fn and fp are all nonzero
        prediction = np.where(rng.random(4000) < 0.5, np.minimum(label, 12), rng.integers(0, 13, 4000))
        black = rng.random(4000) < rng.choice([0.002, 0.05, 0.3])
        before = confusion_of(label, prediction)
        after = confusion_of(np.where(black, 255, label), prediction)
        black_counts = np.bincount(label[black & (label < 13)], minlength=13)
        s = bp.strict8_bounds(before, black_counts)
        exact = bp.strict8_bounds(after, np.zeros(13))["now"]
        assert np.allclose(bp.strict8_bounds(before, np.zeros(13))["now"], s["now"])
        assert (s["low"] <= exact + 1e-9).all() and (exact <= s["high"] + 1e-9).all(), trial
        assert s["mIoU_low"] - 1e-9 <= np.nanmean(exact[s["present_after"]]) <= s["mIoU_high"] + 1e-9, trial
        assert s["mIoU_high"] <= np.nanmean(s["high"][s["present_after"]]) + 1e-9
        # every black pixel predicted right: the low end is the exact rescore
        right = np.where(black & (label < 13), label, prediction)
        exact_right = bp.strict8_bounds(confusion_of(np.where(black, 255, label), right), np.zeros(13))["now"]
        assert np.isclose(bp.strict8_bounds(confusion_of(label, right), black_counts)["mIoU_low"],
                          np.nanmean(exact_right[s["present_after"]]))


def test_no_black_pixels_give_a_zero_width_interval():
    rng = np.random.default_rng(1)
    confusion = rng.integers(0, 1000, (13, 13))
    s = bp.strict8_bounds(confusion, np.zeros(13))
    assert np.array_equal(s["low"], s["now"]) and np.array_equal(s["high"], s["now"])


def test_band_rows_follow_the_eval_rule():
    for height in (8, 16, 1024, 2048):
        band = bp.band_of_rows(height)
        label = np.zeros((height, 4), np.uint8)
        hist_info = lambda n, p, l: (confusion_of(l.ravel(), p.ravel()),)  # noqa: E731
        per_band = ev.elevation_band_confusions(13, label, label, hist_info)
        assert per_band[:, 0, 0].tolist() == (np.bincount(band, minlength=6) * 4).tolist()


def make_dataset(root, count=2, height=16, width=32, seed=2):
    rng = np.random.default_rng(seed)
    names = ["<UNK>_0"] + [name + "_x" for name in ev.EXPECTED_CLASSES[1:]]  # index i -> class i
    labels_path = root / "semantic_labels.json"
    labels_path.write_text(json.dumps(names))
    folder = root / "area_5a" / "pano"
    for sub in ("rgb", "depth", "semantic"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    expected = []
    for i in range(count):
        stem = "camera_{}_frame_equirectangular_domain".format(i)
        index = rng.integers(0, 14, (height, width))  # 0 = <UNK>
        bgr = np.stack([index % 256, index // 256, np.zeros_like(index)], axis=2).astype(np.uint8)
        rgb = rng.integers(1, 256, (height, width, 3)).astype(np.uint8)
        rgb[-3:] = 0                                   # black bottom rows
        rgb[0, :5] = 0                                 # and a few on top
        rgb[5, 7] = (0, 0, 9)                          # dark but not black
        cv2.imwrite(str(folder / "rgb" / (stem + "_rgb.png")), rgb)
        cv2.imwrite(str(folder / "semantic" / (stem + "_semantic.png")), bgr)
        cv2.imwrite(str(folder / "depth" / (stem + "_depth.png")), np.ones((height, width), np.uint16))
        expected.append((index, (rgb == 0).all(axis=2)))
    return labels_path, expected


def test_count_end_to_end_and_bounds_find_the_label_size(tmp_path):
    labels_path, expected = make_dataset(tmp_path)
    out = tmp_path / "out"
    assert bp.main(["count", "--stanford-root", str(tmp_path), "--semantic-labels", str(labels_path),
                    "--output", str(out), "--areas", "area_5a", "--label-size", "16", "32", "--label-size", "8", "16"]) == 0
    counts = np.load(out / "black_pixel_counts.npz")
    summary = json.loads((out / "black_pixel_summary.json").read_text())

    # native size: straight from the arrays that were written
    label = np.concatenate([index for index, _ in expected]) - 1   # class id 0..12, <UNK> -> -1
    black = np.concatenate([mask for _, mask in expected])
    assert counts["16x32_labelled"].sum() == (label >= 0).sum()
    assert counts["16x32_black_labelled"].sum(0).tolist() == np.bincount(label[black & (label >= 0)], minlength=13).tolist()
    assert counts["16x32_black_any"].sum() == black.sum() == 2 * (3 * 32 + 5)
    # bottom 3 of 16 rows are elevation below -60 (rows 11..15 are -33.75..-78.75); row 0 is the top band
    assert counts["16x32_black_any"].tolist() == [10, 0, 0, 0, 2 * 32, 2 * 2 * 32]
    assert summary["sizes"]["16x32"]["black_any"] == 202
    rows = list(csv.DictReader((out / "per_panorama.csv").open()))
    assert [int(r["black_any"]) for r in rows] == [101, 101]

    # bounds pick the size whose ground-truth counts match, per band
    rng = np.random.default_rng(3)
    for key, size in (("16x32", (16, 32)), ("8x16", (8, 16))):
        band_confusion = np.zeros((6, 13, 13), np.int64)
        band = bp.band_of_rows(size[0])
        for index, _ in expected:
            small = cv2.resize(index.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
            lab = np.where(small == 0, 255, small.astype(np.int64) - 1)
            pred = rng.integers(0, 13, lab.shape)
            for b in range(6):
                band_confusion[b] += confusion_of(lab[band == b], pred[band == b])
        assert bp.match_size(band_confusion, counts) == key
        assert bp.match_size(band_confusion.sum(0), counts) == key
    path = tmp_path / "run.npy"
    np.save(path, band_confusion)
    assert bp.main(["bounds", "--counts", str(out / "black_pixel_counts.npz"), "--output",
                    str(tmp_path / "b.csv"), "r=" + str(path)]) == 0
    table = list(csv.DictReader((tmp_path / "b.csv").open()))
    assert [r["band"] for r in table] == ["all"] + bp.BANDS
    assert table[2]["width"] == "0.00"   # no black pixels in the 60..25 band
    assert table[-1]["mIoU_low"] == "nan"  # at 8x16 the bottom row is all black


def test_eval_ignore_black_rgb_masks_the_same_pixels_as_the_counter(tmp_path):
    labels_path, expected = make_dataset(tmp_path, count=1)
    sample = ev.list_panoramas(tmp_path, ["area_5a"])[0]
    for size in ((16, 32), (8, 16)):
        mask = ev.black_rgb_mask(sample["rgb"], size)
        direct = cv2.resize(expected[0][1].astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        assert np.array_equal(mask, direct.astype(bool))
