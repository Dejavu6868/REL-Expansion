"""Pin2Pan adaptation pieces ported from Trans4PASS (adaptations/, commit 758f2301).

* ``feat_kl_loss``: ``utils/loss.py``, vectorised and for any class count.
* ``FCDiscriminator``: ``model/discriminator.py``, unchanged.
* ``class_thresholds``: the per-class rule of ``gen_pseudo_label.py``.
* ``PrototypeMemory``: the memory of ``train_mpa.py`` and ``model/memory.py``,
  kept identical on every rank, with the update fixed (see the class).
* ``TargetPanoramas``: cached target panoramas cut into full-height crops at a
  random yaw, so every crop holds both poles.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255
CONFIDENCE_LEVELS = 65535  # confidences are stored as round(p * 65535) in uint16
KL_TEMPERATURE = 20.0
KL_ALPHA = 0.9


def batch_class_means(feats, labels, num_classes):
    """Per-class mean feature over labelled pixels; labels already at feature size, 255 ignored."""
    channels = feats.shape[1]
    flat = feats.detach().permute(0, 2, 3, 1).reshape(-1, channels)
    labels = labels.reshape(-1)
    valid = labels != IGNORE_INDEX
    counts = torch.bincount(labels[valid], minlength=num_classes)
    sums = flat.new_zeros(num_classes, channels).index_add_(0, labels[valid], flat[valid])
    return sums / counts.clamp(min=1)[:, None], counts > 0


def labels_at(labels, size):
    return F.interpolate(labels[:, None].float(), size, mode="nearest")[:, 0].long()


def feat_kl_loss(feats, labels, memory):
    """Trans4PASS ``feat_kl_loss``: pull labelled pixels' features toward their class prototype.

    Same loss as the original (temperature 20, alpha 0.9, ``KLDivLoss`` mean
    plus cross-entropy to the prototype's arg-max). As there, an unlabelled pixel's
    target is its own feature. Returns the loss and this batch's class means
    (no gradient) and which classes occurred.
    """
    batch, channels, height, width = feats.shape
    labels = labels_at(labels, (height, width))
    flat = feats.permute(0, 2, 3, 1).reshape(-1, channels)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != IGNORE_INDEX
    prototypes = memory[torch.where(valid, flat_labels, torch.zeros_like(flat_labels))]
    target = torch.where(valid[:, None], prototypes, flat)
    target = target.view(batch, height, width, channels).permute(0, 3, 1, 2)
    kl = F.kl_div(
        F.log_softmax(feats / KL_TEMPERATURE, dim=1),
        F.softmax(target / KL_TEMPERATURE, dim=1),
        reduction="mean",
    )
    loss = kl * (KL_ALPHA * KL_TEMPERATURE ** 2) + F.cross_entropy(
        feats, target.argmax(dim=1)
    ) * (1.0 - KL_ALPHA)
    means, present = batch_class_means(feats, labels, memory.shape[0])
    return loss, means, present


class FCDiscriminator(nn.Module):
    """Output-space discriminator of AdaptSegNet, as used by Trans4PASS."""

    def __init__(self, num_classes, ndf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4):
            x = self.leaky_relu(conv(x))
        return self.classifier(x)


def quantize_confidence(probability):
    return np.rint(np.asarray(probability, dtype=np.float64) * CONFIDENCE_LEVELS).astype(np.uint16)


def class_thresholds(histograms, cap=0.9):
    """Trans4PASS rule from per-class histograms of quantized confidence.

    For each class, the confidence at sorted index ``round(n * 0.5)`` among the
    ``n`` pixels predicted as that class; 0 for a class never predicted; capped at 0.9.
    """
    thresholds = np.zeros(len(histograms), dtype=np.float64)
    for class_id, histogram in enumerate(histograms):
        count = int(histogram.sum())
        if count:
            index = int(np.round(count * 0.5))
            level = np.searchsorted(np.cumsum(histogram), index, side="right")
            thresholds[class_id] = level / CONFIDENCE_LEVELS
    return np.minimum(thresholds, cap)


def apply_thresholds(prediction, confidence, thresholds):
    """Keep a pixel when its confidence reaches its predicted class's threshold; else 255."""
    keep = confidence.astype(np.float64) / CONFIDENCE_LEVELS >= thresholds[prediction]
    return np.where(keep, prediction, IGNORE_INDEX).astype(np.uint8)


def all_reduce_sum(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


class PrototypeMemory:
    """Class prototypes for ``feat_kl_loss``, identical on every rank.

    Trans4PASS ``train_mpa.py`` stores each batch's class means and every 100
    iterations moves the memory 0.1% toward ``np.mean(buffer)``. That mean has
    no axis, so all channels get one scalar; the buffer is never cleared; and a
    class missing from the buffer is pulled toward zero. Here each update uses
    the per-channel mean of the batch means gathered on all ranks since the last
    update, only for classes that occurred, and then clears the buffer.
    """

    def __init__(self, memory, momentum=0.999, update_every=100):
        self.memory = memory
        self.momentum = momentum
        self.update_every = update_every
        self.sums = torch.zeros_like(memory)
        self.counts = torch.zeros(memory.shape[0], dtype=memory.dtype, device=memory.device)

    def collect(self, means, present):
        self.sums[present] += means[present]
        self.counts += present.to(self.counts.dtype)

    def maybe_update(self, iteration):
        """Trans4PASS timing: after the batches of iteration 100, 200, ... (0-based)."""
        if iteration == 0 or iteration % self.update_every:
            return False
        sums = all_reduce_sum(self.sums.clone())
        counts = all_reduce_sum(self.counts.clone())
        seen = counts > 0
        centres = sums[seen] / counts[seen, None]
        self.memory[seen] = self.memory[seen] * self.momentum + centres * (1.0 - self.momentum)
        self.sums.zero_()
        self.counts.zero_()
        return True


def initial_memory(source_batches, target_batches, predict, num_classes, channels, device):
    """Trans4PASS ``init_memory``: mean over batches of each class's mean feature on
    correctly predicted pixels, per domain, then the average of the two domains.

    ``predict(batch)`` returns (features, logits) without gradients. Batches are
    summed over all ranks. A class seen in one domain only takes that domain's mean.
    """
    domains = []
    for batches in (source_batches, target_batches):
        sums = torch.zeros(num_classes, channels, device=device)
        counts = torch.zeros(num_classes, device=device)
        for batch in batches:
            feats, logits = predict(batch)
            label = batch["label"].to(device)
            prediction = logits.argmax(dim=1)
            correct = torch.where(prediction == label, label, torch.full_like(label, IGNORE_INDEX))
            means, present = batch_class_means(feats, labels_at(correct, feats.shape[2:]), num_classes)
            sums[present] += means[present]
            counts += present.float()
        all_reduce_sum(sums)
        all_reduce_sum(counts)
        domains.append((sums / counts.clamp(min=1)[:, None], counts))
    (source, source_counts), (target, target_counts) = domains
    weights = torch.stack([source_counts > 0, target_counts > 0]).float()
    memory = (source * weights[0, :, None] + target * weights[1, :, None]) / weights.sum(0).clamp(min=1)[:, None]
    return memory, source_counts.long().tolist(), target_counts.long().tolist()


class TargetPanoramas(torch.utils.data.Dataset):
    """Cached panoramas (``build_target_cache.py``), cut into full-height crops at a random yaw.

    ``label_root`` holds pseudo-labels (``gen_pseudo_labels.py``); without it
    every label is 255. The cache's ground truth (``Label/``) is never read here.
    """

    def __init__(self, cache_root, sample_ids, crop_width, normalize, label_root=None):
        self.cache_root = Path(cache_root)
        self.sample_ids = list(sample_ids)
        self.crop_width = crop_width
        self.normalize = normalize
        self.label_root = None if label_root is None else Path(label_root)

    def __len__(self):
        return len(self.sample_ids)

    @staticmethod
    def read(path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        return image

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        rgb = self.read(self.cache_root / "RGB" / (sample_id + ".png"))
        modal_x = self.read(self.cache_root / "X" / (sample_id + ".png"))
        if self.label_root is None:
            label = np.full(rgb.shape[:2], IGNORE_INDEX, dtype=np.uint8)
        else:
            label = self.read(self.label_root / (sample_id + ".png"))
        width = rgb.shape[1]
        if not self.crop_width <= width:
            raise ValueError("crop width {} exceeds the panorama width {}".format(self.crop_width, width))
        columns = (random.randrange(width) + np.arange(self.crop_width)) % width  # yaw wraps around
        tensor = lambda image: torch.from_numpy(
            np.ascontiguousarray(self.normalize(image[:, columns]).transpose(2, 0, 1), dtype=np.float32)
        )
        return {
            "data": tensor(rgb),
            "modal_x": tensor(modal_x),
            "label": torch.from_numpy(np.ascontiguousarray(label[:, columns])).long(),
            "fn": sample_id,
        }
