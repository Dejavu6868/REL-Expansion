# --*-- coding:utf-8 --*--
import numpy as np
import cv2

'''
helper function
'''
def filterItChopOff(f, r, sp):
    f = np.nan_to_num(f, copy=False)
    H, W, d = f.shape
    B = np.ones((2 * r + 1, 2 * r + 1), dtype=np.float32)

    min_sp = cv2.erode(sp, B, iterations=1)
    max_sp = cv2.dilate(sp, B, iterations=1)
    edge_mask = np.logical_or(min_sp != sp, max_sp != sp)
    edge_indices = np.where(edge_mask)

    # If no edge points, return filtered result directly
    if len(edge_indices[0]) == 0:
        return cv2.filter2D(f, -1, B, borderType=cv2.BORDER_CONSTANT)

    sp_expanded = np.pad(sp, ((r, r), (r, r)), mode="constant", constant_values=-1)

    delta = np.zeros_like(f)
    
    # Process edge points
    for x, y in zip(*edge_indices):
        # Extract neighborhood
        neighborhood_sp = sp_expanded[x : x + 2 * r + 1, y : y + 2 * r + 1]
        mask = (neighborhood_sp != sp[x, y])[r:-r, r:-r]

        # Calculate valid region
        x_start = max(0, x - r)
        x_end = min(H, x + r + 1)
        y_start = max(0, y - r)
        y_end = min(W, y + r + 1)

        # Extract neighborhood and apply mask
        neighborhood_f = f[x_start:x_end, y_start:y_end, :]
        valid_mask = mask[:neighborhood_f.shape[0], :neighborhood_f.shape[1]]
        valid_mask_3d = np.repeat(valid_mask[:, :, np.newaxis], d, axis=2)

        # Calculate delta
        delta[x, y, :] = np.sum(neighborhood_f[valid_mask_3d].reshape(-1, d), axis=0)

    fFilt = cv2.filter2D(f, -1, B, borderType=cv2.BORDER_CONSTANT)
    return fFilt - delta


def mutiplyIt(AtA_1, Atb):
    # Use NumPy vectorized operations
    result = np.zeros_like(Atb)
    
    # Matrix multiplication
    result[..., 0] = AtA_1[..., 0] * Atb[..., 0] + AtA_1[..., 1] * Atb[..., 1] + AtA_1[..., 2] * Atb[..., 2]
    result[..., 1] = AtA_1[..., 1] * Atb[..., 0] + AtA_1[..., 3] * Atb[..., 1] + AtA_1[..., 4] * Atb[..., 2]
    result[..., 2] = AtA_1[..., 2] * Atb[..., 0] + AtA_1[..., 4] * Atb[..., 1] + AtA_1[..., 5] * Atb[..., 2]
    
    return result


def invertIt(AtA):
    # Calculate inverse matrix
    a = AtA[..., 0]
    b = AtA[..., 1]
    c = AtA[..., 2]
    d = AtA[..., 3]
    e = AtA[..., 4]
    f_val = AtA[..., 5]

    # Preallocate memory
    H, W, _ = AtA.shape
    AtA_1 = np.empty((H, W, 6))
    
    # Calculate inverse matrix elements
    AtA_1[..., 0] = d * f_val - e * e
    AtA_1[..., 1] = -b * f_val + c * e
    AtA_1[..., 2] = b * e - c * d
    AtA_1[..., 3] = a * f_val - c * c
    AtA_1[..., 4] = -a * e + b * c
    AtA_1[..., 5] = a * d - b * b

    # Calculate determinant
    detAta = a * AtA_1[..., 0] + b * AtA_1[..., 1] + c * AtA_1[..., 2]
    return AtA_1, detAta


def getRMatrix(yi, yf):
    """
    getRMatrix: Generate a rotation matrix that
                if yf is a scalar, rotates about axis yi by yf degrees
                if yf is an axis, rotates yi to yf in the direction given by yi x yf
    Input: yi is an axis 3x1 vector
           yf could be a scalar of axis

    """
    if np.isscalar(yf):
        ax = yi / np.linalg.norm(yi)  # norm(A) = max(svd(A))
        phi = yf
    else:
        yi = yi / np.linalg.norm(yi)
        yf = yf / np.linalg.norm(yf)
        ax = np.cross(yi.T, yf.T).T
        ax = ax / np.linalg.norm(ax)
        # find angle of rotation
        phi = np.degrees(np.arccos(np.dot(yi.T, yf)))

    if abs(phi) > 0.1:
        phi = phi * (np.pi / 180)
        ax = ax.flatten()
        s_hat = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = (
            np.eye(3) + np.sin(phi) * s_hat + (1 - np.cos(phi)) * np.dot(s_hat, s_hat)
        )  # dot???
    else:
        R = np.eye(3)
    return R


def rotatePC(pc, R):
    """
    Calibration of gravity direction
    """
    if np.allclose(R, np.eye(3)):
        return pc
    return np.einsum("ij,klj->kli", R, pc)

def getGDir(N, angleThresh, iter, g0):
    """
    Compute the direction of gravity
    N: normal field
    iter: number of 'big' iterations
    """
    g = g0.copy()
    for i in range(len(angleThresh)):
        thresh = np.radians(angleThresh[i])
        g = getGDirHelper(N, g, thresh, iter[i])
    return g


def getGDirHelper(N, g0, thresh, num_iter):
    """
    Compute gravity direction
    """
    # Process point cloud
    nn = np.moveaxis(N, -1, 0).reshape(3, -1)
    valid = ~np.isnan(nn[0])
    nn = nn[:, valid]
    
    # Early termination check
    if nn.shape[1] < 10:
        return g0
    
    gDir = g0.copy()
    
    # Precompute trigonometric values
    cos_thresh = np.cos(thresh)
    sin_thresh = np.sin(thresh)
    
    # Iterate with adaptive termination
    for i in range(num_iter):
        # Calculate dot product
        sim = np.dot(gDir, nn)
        
        # Calculate mask
        indF = np.abs(sim) > cos_thresh
        indW = np.abs(sim) < sin_thresh
        
        # Avoid empty matrix
        if np.sum(indF) < 5 or np.sum(indW) < 5:
            break
        
        # Extract valid points
        NF = nn[:, indF]
        NW = nn[:, indW]
        
        # Calculate matrix
        A = np.dot(NW, NW.T) - np.dot(NF, NF.T)
        
        # Eigenvalue decomposition
        try:
            # Eigenvalue decomposition
            w, v = np.linalg.eigh(A)
            # Select eigenvector corresponding to minimum eigenvalue
            min_eig_idx = np.argmin(w)
            new_gDir = v[:, min_eig_idx]
            
            # Ensure direction consistency
            if np.dot(gDir, new_gDir) < 0:
                new_gDir = -new_gDir
            
            # Check convergence
            if np.linalg.norm(gDir - new_gDir) < 1e-3:
                break
            
            gDir = new_gDir
        except:
            break
    
    return gDir
