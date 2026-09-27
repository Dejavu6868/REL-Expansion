# --*-- coding:utf-8 --*--
import numpy as np

from .hha_util import *

np.seterr(divide="ignore", invalid="ignore")


def processDepthImage_ERP(D, missingMask):
    """
    D: depth image in 'centimetres'
    missingMask: a mask
    """
    pc = getPointCloud_ERP(D)

    normalParam_patchSize = np.array([2, 10])
    N, _ = computeNormalsSquareSupport_ERP(
        pc, missingMask, normalParam_patchSize[0], np.ones(D.shape)
    )

    # Compute the direction of gravity
    angleThresh = np.array(
        [15, 5]
    )  # threshold to estimate the direction of the gravity
    gIter = np.array([5, 5])
    g0 = np.array([0, 0, -1])
    gDir = getGDir(N, angleThresh, gIter, g0)
    alpha = -np.rad2deg(np.arcsin(gDir[0]))
    beta = -np.rad2deg(np.arctan(gDir[1] / gDir[2]))
    rot = [round(alpha, 2), round(beta, 2)]

    R = getRMatrix(g0.T, gDir)
    NRot = rotatePC(N, R.T)
    pcRot = rotatePC(pc, R.T)

    return pcRot, NRot, rot


def getPointCloud_ERP(D):
    """
    Convert the ERP panoramic depth map to a 3D point cloud.
    Z: Depth map (in centimeters), with shape (H, W).
    """
    h, w = D.shape

    # Generate u and v coordinates
    u = np.arange(w)[np.newaxis, :]
    v = np.arange(h)[:, np.newaxis]

    # Calculate phi and theta
    phi = (u / w) * 2 * np.pi - np.pi  # Longitude range [-π, π]
    theta = (0.5 - v / h) * np.pi  # Latitude range [-π/2, π/2]

    # Calculate trigonometric function values
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)

    # Calculate point cloud coordinates
    x = D * cos_theta * sin_phi
    y = D * cos_theta * cos_phi
    z = D * sin_theta

    # Create point cloud
    pc = np.stack([x, y, z], axis=2)

    return pc


def computeNormalsSquareSupport_ERP(pc_raw, missingMask, R, superpixels):
    """
    Clip out a 2R+1 x 2R+1 window at each point and estimate
    the normal from points within this window. In case the window
    straddles more than a single superpixel, only take points in the
    same superpixel as the centre pixel.

    Input:
        pc: 3D point cloud (HxWx3) in meters.
        missingMask: Boolean mask indicating missing data (HxW).
        R: Radius of clipping.
        superpixels: Superpixel map to define boundaries that should not be straddled.
    """

    pc = pc_raw.copy()
    X, Y, Z = pc[:, :, 0], pc[:, :, 1], pc[:, :, 2]
    XYZf = pc

    # Handle missing values
    X[missingMask] = np.nan
    Y[missingMask] = np.nan
    Z[missingMask] = np.nan

    eps = 1e-4
    Z[Z == 0] = eps

    one_Z = np.expand_dims(1 / Z, axis=2)
    one = np.ones_like(Z)
    one[np.isnan(Z)] = 1

    X_Z = X / Z
    Y_Z = Y / Z
    ZZ = Z * Z
    X_ZZ = np.expand_dims(X / ZZ, axis=2)
    Y_ZZ = np.expand_dims(Y / ZZ, axis=2)

    # Build matrix
    AtARaw = np.stack(
        [X_Z**2, X_Z * Y_Z, X_Z, Y_Z**2, Y_Z, one], axis=2
    )

    AtbRaw = np.stack([X_ZZ[..., 0], Y_ZZ[..., 0], one_Z[..., 0]], axis=2)

    # with clipping
    combined = np.concatenate((AtARaw, AtbRaw), axis=2)
    AtA = filterItChopOff(combined, R, superpixels)
    Atb = AtA[:, :, AtARaw.shape[2] :]
    AtA = AtA[:, :, : AtARaw.shape[2]]

    AtA_1, detAtA = invertIt(AtA)
    N = mutiplyIt(AtA_1, Atb)

    # Normalize normals
    divide_fac = np.linalg.norm(N, axis=2, keepdims=True)
    divide_fac[divide_fac == 0] = 1e-4  # Avoid division by zero

    # Normalize normals
    N = N / divide_fac
    b = -detAtA / divide_fac[..., 0]

    # Adjust normal directions
    SN = np.sign(N[:, :, 2])
    SN[SN == 0] = 1
    SN_3d = SN[..., np.newaxis]
    N *= SN_3d
    b *= SN

    # Ensure consistent normal directions
    dot_product = np.sum(N * XYZf, axis=2)
    sn = np.sign(dot_product)
    sn[np.isnan(sn)] = 1
    sn[sn == 0] = 1
    sn_3d = sn[..., np.newaxis]
    N *= sn_3d
    b *= sn

    return N, b
