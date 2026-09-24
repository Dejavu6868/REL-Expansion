"""Original ERP-REL functions extracted from SrtaEstrella/REL-SF4PASS.

The numerical statements and their order are kept from ``getREL.py``.  The
only packaging change is the relative import of the extracted ERP utilities.
"""

import cv2
import numpy as np

from .rgbd_util import processDepthImage_ERP


def getImage(file_path, ds="Stanford2D3DPano"):
    """Load and normalize depth exactly as the public source does."""
    D = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)

    if ds == "Stanford2D3DPano" or ds == "ToF-360":
        D += 1
        D = D.astype(np.float32) / 512.0
    elif ds == "Matterport":
        D = D.astype(np.float32) / 4000.0
    else:
        raise ValueError(f"Unknown dataset: {ds}")

    return D


def getREL(D, alpha=45, lam=0.5):
    """Calculate the original ERP-REL three-channel uint8 array."""
    missingMask = D == 0

    pcRot, N, rot = processDepthImage_ERP(D * 100, missingMask)

    h, w = D.shape

    u = np.arange(w)
    phi = (u / w) * 2 * np.pi - np.pi
    cos_theta = np.cos(phi)
    sin_theta = np.sin(phi)

    cos_theta = cos_theta[np.newaxis, :]
    sin_theta = sin_theta[np.newaxis, :]

    hcos = N[:, :, 0] * cos_theta - N[:, :, 1] * sin_theta
    hcos = np.nan_to_num(hcos, nan=0)
    hcos = np.clip(hcos, -1.0, 1.0)
    HA = (np.arccos(hcos) * 180 / np.pi).astype(np.uint8)

    RD = np.hypot(pcRot[:, :, 0], pcRot[:, :, 1])
    RD_min = RD.min()
    RD_max = RD.max()
    if RD_max > RD_min:
        RD = (RD - RD_min) * 255.0 / (RD_max - RD_min)
    RD = np.clip(RD, 0, 255).astype(np.uint8)

    h_val = pcRot[:, :, 2]
    hmin = np.percentile(h_val, 1)
    hmax = np.percentile(h_val, 99)
    if hmax > hmin:
        h_val = (h_val - hmin) * 255.0 / (hmax - hmin)
    h_val = np.clip(h_val, 0, 255).astype(np.float32)

    N_z = -N[:, :, 2]
    N_z = np.clip(N_z, -1.0, 1.0)
    angle = (np.arccos(N_z, dtype=np.float32) / np.pi) * 255.0
    angle = np.clip(angle, 0, 255).astype(np.float32)

    angle_threshold = alpha * 255.0 / 180.0
    is_horizontal = (angle <= angle_threshold) | (
        angle >= 255.0 - angle_threshold
    )
    angle[~is_horizontal] = lam * angle[~is_horizontal] + (
        1 - lam
    ) * h_val[~is_horizontal]

    I = np.stack([angle, HA, RD], axis=2).astype(np.float32)
    I = np.nan_to_num(I, nan=255.0)
    I[missingMask, :] = 255.0
    I = np.clip(I, 0, 255).astype(np.uint8)

    return I
