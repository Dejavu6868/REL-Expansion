"""Canonical full-image evaluator plumbing for CMX-REL+ v2.1 S2D."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from utils.metric import hist_info
from utils.transforms import normalize


@dataclass(frozen=True)
class PreparedEvalSample:
    rgb: torch.Tensor
    modal_x: torch.Tensor
    label: np.ndarray
    valid_mask: np.ndarray
    sample_id: str


def _as_uint8_hwc3(value, name):
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("{} must be uint8 HxWx3".format(name))
    return array


def _normalized_bchw(array, mean, std):
    chw = normalize(array, mean, std).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(chw, dtype=np.float32))[None]


def prepare_eval_sample(sample, config):
    if getattr(config, "x_mode", None) != "rel_plus_v2_1":
        raise ValueError("canonical REL+ evaluator requires x_mode=rel_plus_v2_1")
    if getattr(config, "eval_flip", None) is not False:
        raise ValueError("REL+ evaluator requires eval_flip=False")
    if list(getattr(config, "eval_scale_array", [])) != [1]:
        raise ValueError("REL+ S2D evaluator requires eval_scale_array=[1]")

    rgb = _as_uint8_hwc3(sample["data"], "RGB")
    modal_x = _as_uint8_hwc3(sample["modal_x"], "REL+")
    label = np.asarray(sample["label"])
    valid_mask = np.asarray(sample.get("modal_x_valid_mask"))
    if label.shape != rgb.shape[:2] or modal_x.shape[:2] != rgb.shape[:2]:
        raise ValueError("RGB, REL+ and label shapes must match")
    if valid_mask.shape != rgb.shape[:2]:
        raise ValueError("REL+ evaluation valid mask shape mismatch")
    crop_size = tuple(int(value) for value in config.eval_crop_size)
    if crop_size != rgb.shape[:2]:
        raise ValueError(
            "full-image S2D evaluation requires sample shape {} but found {}".format(
                crop_size, rgb.shape[:2]
            )
        )
    return PreparedEvalSample(
        rgb=_normalized_bchw(rgb, config.norm_mean, config.norm_std),
        modal_x=_normalized_bchw(
            modal_x, config.norm_mean, config.norm_std
        ),
        label=np.ascontiguousarray(label),
        valid_mask=np.ascontiguousarray(valid_mask, dtype=bool),
        sample_id=str(sample["fn"]),
    )


def evaluate_prepared_sample(
    network, prepared, *, class_num, ignore_index, device
):
    del ignore_index  # hist_info already excludes labels outside [0, class_num).
    network.eval()
    with torch.no_grad():
        logits = network(
            prepared.rgb.to(device), prepared.modal_x.to(device)
        )
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError("CMX evaluator expected BCHW logits with batch size one")
    if tuple(logits.shape[2:]) != prepared.label.shape:
        raise ValueError("prediction/logit shape does not match label")
    finite = bool(torch.isfinite(logits).all())
    if not finite:
        raise FloatingPointError("CMX evaluator produced non-finite logits")
    prediction = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    hist, labeled, correct = hist_info(class_num, prediction, prepared.label)
    return {
        "prediction": prediction,
        "hist": hist.astype(np.int64),
        "labeled": int(labeled),
        "correct": int(correct),
        "logits_finite": finite,
        "diagnostic_mask_passed_to_model": False,
    }


def merge_confusion_matrices(matrices):
    values = [np.asarray(matrix, dtype=np.int64) for matrix in matrices]
    if not values:
        raise ValueError("at least one confusion matrix is required")
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("confusion matrix shapes must match")
    return np.sum(np.stack(values, axis=0), axis=0, dtype=np.int64)


def aggregate_rank_evaluations(rank_reports):
    reports = list(rank_reports)
    if not reports:
        raise ValueError("at least one rank evaluation report is required")
    confusion = merge_confusion_matrices(
        report["confusion_matrix"] for report in reports
    )
    sample_count = sum(int(report["sample_count"]) for report in reports)
    owned = [
        sample_id
        for report in reports
        for sample_id in report["owned_sample_ids"]
    ]
    if len(owned) != sample_count or len(set(owned)) != sample_count:
        raise ValueError("rank ownership is incomplete or duplicated")
    return confusion, sample_count, owned


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    true_pixels = np.diag(confusion).astype(np.float64)
    ground_truth = confusion.sum(axis=1, dtype=np.int64).astype(np.float64)
    predicted = confusion.sum(axis=0, dtype=np.int64).astype(np.float64)
    union = ground_truth + predicted - true_pixels
    iou = np.divide(
        true_pixels,
        union,
        out=np.full_like(true_pixels, np.nan),
        where=union > 0,
    )
    class_accuracy = np.divide(
        true_pixels,
        ground_truth,
        out=np.full_like(true_pixels, np.nan),
        where=ground_truth > 0,
    )
    total = int(confusion.sum())
    return {
        "mIoU": float(np.nanmean(iou)),
        "pixel_accuracy": float(true_pixels.sum() / total) if total else 0.0,
        "mean_accuracy": float(np.nanmean(class_accuracy)),
        "per_class_iou": iou.tolist(),
        "valid_pixel_count": total,
    }


def save_prediction_pair(prediction, sample_id, output_dir, class_colors):
    prediction = np.asarray(prediction)
    if prediction.ndim != 2:
        raise ValueError("prediction must be HxW")
    output_dir = Path(output_dir)
    relative = Path(str(sample_id) + ".png")
    raw_path = output_dir / "raw" / relative
    color_path = output_dir / "color" / relative
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    color_path.parent.mkdir(parents=True, exist_ok=True)
    raw_uint8 = prediction.astype(np.uint8)
    if not cv2.imwrite(str(raw_path), raw_uint8):
        raise RuntimeError("failed to save raw prediction {}".format(raw_path))

    palette = list(np.asarray(class_colors, dtype=np.uint8).reshape(-1))
    palette = (palette + [0] * 768)[:768]
    color = Image.fromarray(raw_uint8, mode="P")
    color.putpalette(palette)
    color.save(str(color_path))
    return {"raw": raw_path, "color": color_path}
