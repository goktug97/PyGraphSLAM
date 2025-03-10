# Original code: https://github.com/ClayFlannigan/icp
# Modifications:
# Reject pairs that have greater distance than 1 meters
# Weights https://arxiv.org/abs/2007.07627
# Handle different length scans

import numpy as np
from sklearn.neighbors import NearestNeighbors


def welsch_weight(weights, p=0.4):
    return np.exp(-weights * weights / (2 * np.square(p)))


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    weights = np.linalg.norm(A - B, axis=1)
    weights = welsch_weight(weights)

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T * weights, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    """
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    T_acc = init_pose if init_pose is not None else np.identity(m + 1)
    src = np.dot(T_acc, src)

    prev_error = 0

    for i in range(max_iterations):
        distances, indices = nearest_neighbor(src.T, dst.T)

        # Reject pairs that have more than 1 meter distance between them
        close_match_mask = distances < 1.0

        # Need at least 3 points for a meaningful transform
        if np.sum(close_match_mask) < 3:
            continue

        matched_src = src[:m, close_match_mask].T
        matched_dst = dst[:m, indices[close_match_mask]].T

        T, _, _ = best_fit_transform(matched_src, matched_dst)

        # update the current source
        src = np.dot(T, src)

        T_acc = np.dot(T, T_acc)

        # check error
        mean_error = np.mean(distances[close_match_mask])
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return T_acc, distances, i
