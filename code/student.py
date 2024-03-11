import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    # TODO: calculate M

    n = points2d.shape[0]
    data_matrix = np.zeros((2 * n, 11))
    coefficient_vector = np.zeros((2 * n, 1))

    for i in range(0, n):
        X = points3d[i, 0]
        Y = points3d[i, 1]
        Z = points3d[i, 2]
        u = points2d[i, 0]
        v = points2d[i, 1]

        data_matrix[(2 *
                     i), :] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z]
        data_matrix[(2 * i) +
                    1, :] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z]
        coefficient_vector[(2 * i)] = u
        coefficient_vector[(2 * i) + 1] = v

    M, residual, rank, s = np.linalg.lstsq(data_matrix, coefficient_vector, rcond=None)
    M = np.append(M, 1)
    M = np.reshape(M, (3, 4))

    return M, residual


def normalize_coordinates(Points):
    """
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param Points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the transformation
    matrix
    """
    n = Points.shape[0]
    u = np.copy(Points[:, 0])
    v = np.copy(Points[:, 1])

    # Calculate offset matrix
    c_u = np.mean(u)
    c_v = np.mean(v)

    offset_matrix = np.array([[1, 0, -c_u], [0, 1, -c_v], [0, 0, 1]])

    # Calculate scale matrix
    s = 1 / np.std([[u - c_u], [v - c_v]])

    scale_matrix = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])

    # Calculate transformation matrix
    T = scale_matrix @ offset_matrix

    # Normalize points using transformation matrix
    for i in range(0, n):
        norm = T @ np.transpose([u[i], v[i], 1])
        u[i] = norm[0]
        v[i] = norm[1]

    return np.column_stack((u, v)), T


def estimate_fundamental_matrix_unnormalizedpoints(Points1, Points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will constrain a point to lie
    along a line within the second image - the epipolar line. Fitting a
    fundamental matrix to a set of points will try to minimize the error of
    all points to their respective epipolar lines. The residual can be computed 
    as the difference from the known geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the sum of the squared Euclidean error in the estimation
    """
    # Alternate solution
    n = Points2.shape[0]

    # Points1_norm, T1 = normalize_coordinates(Points1)
    # Points2_norm, T2 = normalize_coordinates(Points2)

    u = np.copy(Points1[:, 0])
    v = np.copy(Points1[:, 1])
    u_prime = np.copy(Points2[:, 0])
    v_prime = np.copy(Points2[:, 1])

    # Create data matrix
    data_matrix = np.array([
        u_prime * u, u_prime * v, u_prime, v_prime * u, v_prime * v, v_prime,
        u, v,
        np.ones((n))
    ])
    data_matrix = np.transpose(data_matrix)

    # Get system matrix using svd
    U, S, Vh = np.linalg.svd(data_matrix)

    # Get column of V coresp to the smallest singular value for full rank F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we take the last
    # row instead of the last column Vh is sorted in descending order of the
    # size of the eigenvalues
    full_F = Vh[-1, :]

    # Reshape column to 3x3 so we have the right dimension for F
    full_F = np.reshape(full_F, (3, 3))

    # Reduce rank to get final F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we don't have to
    # transpose it here for the matrix multiplication to produce F_matrix
    U, S, Vh = np.linalg.svd(full_F)

    # S is sorted in descending order of the size of the eigenvalues
    # Set the smallest singular value to zero.
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    # Calculate the residual as the euclidean norm (sum of squared error among all reprojections)
    residual = np.sum(np.square(data_matrix @ F_matrix.flatten()))

    # RESIDUAL CAN ALSO BE CALCULATED AS:
    dist = np.zeros((n))
    for j in range(0, n):
        homMatch1 = np.append(Points1[j, :], [1])
        homMatch2 = np.append(Points2[j, :], [1])
        dist[j] = np.abs(homMatch2 @ F_matrix @ np.transpose(homMatch1))
    residual = np.sum(np.square(dist))

    return F_matrix, residual

def estimate_fundamental_matrix(Points1, Points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will constrain a point to lie
    along a line within the second image - the epipolar line. Fitting a
    fundamental matrix to a set of points will try to minimize the error of
    all points to their respective epipolar lines. The residual can be computed 
    as the difference from the known geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the sum of the squared Euclidean error in the estimation
    """
    # Alternate solution
    n = Points2.shape[0]

    Points1_norm, T1 = normalize_coordinates(Points1)
    Points2_norm, T2 = normalize_coordinates(Points2)

    u = np.copy(Points1_norm[:, 0])
    v = np.copy(Points1_norm[:, 1])
    u_prime = np.copy(Points2_norm[:, 0])
    v_prime = np.copy(Points2_norm[:, 1])

    # Create data matrix
    data_matrix = np.array([
        u_prime * u, u_prime * v, u_prime, v_prime * u, v_prime * v, v_prime,
        u, v,
        np.ones((n))
    ])
    data_matrix = np.transpose(data_matrix)

    # Get system matrix using svd
    U, S, Vh = np.linalg.svd(data_matrix)

    # Get column of V coresp to the smallest singular value for full rank F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we take the last
    # row instead of the last column Vh is sorted in descending order of the
    # size of the eigenvalues
    full_F = Vh[-1, :]

    # Reshape column to 3x3 so we have the right dimension for F
    full_F = np.reshape(full_F, (3, 3))

    # Reduce rank to get final F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we don't have to
    # transpose it here for the matrix multiplication to produce F_matrix
    U, S, Vh = np.linalg.svd(full_F)

    # S is sorted in descending order of the size of the eigenvalues
    # Set the smallest singular value to zero.
    S[-1] = 0
    F_matrix_norm = U @ np.diagflat(S) @ Vh

    # Calculate the residual as the euclidean norm (sum of squared error among all reprojections)
    residual = np.sum(np.square(data_matrix @ F_matrix_norm.flatten()))

    # Adjust back to original coordinates
    F_matrix = np.transpose(T2) @ F_matrix_norm @ T1

    # RESIDUAL CAN ALSO BE CALCULATED AS FOLLOWS:
    # dist = np.zeros((n))
    # for j in range(0, n):
    #     homMatch1 = np.append(Points1[j, :], [1])
    #     homMatch2 = np.append(Points2[j, :], [1])
    #     dist[j] = np.abs(homMatch2 @ F_matrix @ np.transpose(homMatch1))
    # residual = np.sum(np.square(dist))

    return F_matrix, residual

def estimate_fundamental_matrix_2(Points1, Points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will constrain a point to lie
    along a line within the second image - the epipolar line. Fitting a
    fundamental matrix to a set of points will try to minimize the error of
    all points to their respective epipolar lines. The residual can be computed 
    as the difference from the known geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the sum of the squared Euclidean error in the estimation
    """
    ##################
    # Your code here #
    ##################
    # ALTERNATE SOLUTION
    norm_Points1, T1 = normalize_coordinates(Points1)
    norm_Points2, T2 = normalize_coordinates(Points2)

    u = np.copy(norm_Points1[:, 0])
    v = np.copy(norm_Points1[:, 1])
    u_prime = np.copy(norm_Points2[:, 0])
    v_prime = np.copy(norm_Points2[:, 1])

    # Create data matrix
    data_matrix = np.array([
        u_prime * u, u_prime * v, u_prime, v_prime * u, v_prime * v, v_prime,
        u, v
    ])
    data_matrix = np.transpose(data_matrix)

    M, residual, rank, s = np.linalg.lstsq(data_matrix, -np.ones_like(u), rcond=None)
    full_F = np.append(M, 1).reshape(3, 3)

    # Reduce rank to get final F
    # Note: np.linalg.svd returns the transpose of V (Vh), so we don't have to
    # transpose it here. for the matrix multiplication to produce F_matrix
    U, S, Vh = np.linalg.svd(full_F)

    # S is sorted in descending order of the size of the eigenvalues
    S[-1] = 0
    F_matrix = U @ np.diagflat(S) @ Vh

    # Adjust back to original coordinates
    F_matrix = np.transpose(T2) @ F_matrix @ T1

    # Compute residual on final F_matrix
    residual = np.linalg.norm( Points2 - F_matrix @ Points1 )

    # The 2-norm (Euclidean) error _is_ the largest singular value from the SVD decomposition
    # As the singular values are ordered large to small, the residual error is simply:
    residual = S[0]

    return F_matrix, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points.
    
    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the sum of the squared Euclidean error in the estimation induced by best_Fmatrix upon the inlier set

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    ##################
    # Your code here #
    ##################

    # Your RANSAC loop should contain a call to your 'estimate_fundamental_matrix()'

    num_pts = 8  # number of point correspondances per iteration
    threshold = 0.002  # threshold for distance metric of inliers
    n = matches1.shape[0]
    inlier_best_count = 0
    inlier_best = np.zeros((n))
    best_Fmatrix = np.zeros((3, 3))
    random.seed(0)  # seed the same as student template

    # RANSAC loop. We store the fundamental matrix and number of inliers in
    # each iteration, and keep the best one.
    for i in range(0, num_iters):
        # Pick num_pts random pts from correspondances in [1, n]
        cur_rand = random.sample(range(n), num_pts)
        points1 = matches1[cur_rand, :]
        points2 = matches2[cur_rand, :]

        # Calculate the fundamental matrix using pt 2 work
        cur_F, residual = estimate_fundamental_matrix(points1, points2)

        # Use distance metric to find inliers. For a given correspondence x to
        # x', x'Fx = 0. So our metric refers to how far from zero our result
        # is. Store inliers
        dist = np.zeros((n))
        for j in range(0, n):
            homMatch1 = np.append(matches1[j, :], [1])
            homMatch2 = np.append(matches2[j, :], [1])
            dist[j] = np.abs(homMatch2 @ cur_F @ np.transpose(homMatch1))

        inliers = dist <= threshold
        inlier_count = np.sum(inliers)
        inlier_counts.append(inlier_count)
        # Compute inlier residual as sum of L2 norm
        inlier_residual = np.sum(np.square(dist[inliers]))
        inlier_residuals.append( inlier_residual )
        

        # Compare result to previous F and number of inliers. Replace best F
        # with the current one if it has a larger number of inliers.
        if (inlier_count > inlier_best_count):
            inlier_best_count = inlier_count
            inlier_best = inliers
            best_Fmatrix = cur_F
            best_inlier_residual = inlier_residual
        
    # Take inliers from matched points
    best_inliers1 = matches1[inlier_best, :]
    best_inliers2 = matches2[inlier_best, :]

    return best_Fmatrix, best_inliers1, best_inliers2, best_inlier_residual


def matches_to_3d(points1, points2, M1, M2, threshold=1.0):
    """
    Given two sets of corresponding 2D points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq().

    You may find that some 3D points have high residual/error, in which case you 
    can return a subset of the 3D points that lie within a certain threshold.
    In this case, also return subsets of the initial points2d_1, points2d_2 that
    correspond to this new inlier set. You may modify the default value of threshold above.

    N is the input number of point correspondences
    M is the output number of 3D points / inlier point correspondences; M could equal N.

    :param points2d_1: [N x 2] points from image1
    :param points2d_2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :param threshold: scalar value representing the maximum allowed residual for a solved 3D point

    :return points3d_inlier: [M x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points2d_1 and points2d_2
    :return points2d_1_inlier: [M x 2] points as subset of inlier points from points2d_1
    :return points2d_2_inlier: [M x 2] points as subset of inlier points from points2d_2
    """
    N = np.shape(points1)[0]
    points3d = np.zeros((N, 3))
    residuals = np.zeros(N,)
    for idx, (pt1, pt2) in enumerate(zip(points1, points2)):
        A = [
            [M1[0, 0] - pt1[0] * M1[2, 0], M1[0, 1] - pt1[0] * M1[2, 1],
             M1[0, 2] - pt1[0] * M1[2, 2]],
            [M1[1, 0] - pt1[1] * M1[2, 0], M1[1, 1] - pt1[1] * M1[2, 1],
             M1[1, 2] - pt1[1] * M1[2, 2]],
            [M2[0, 0] - pt2[0] * M2[2, 0], M2[0, 1] - pt2[0] * M2[2, 1],
             M2[0, 2] - pt2[0] * M2[2, 2]],
            [M2[1, 0] - pt2[1] * M2[2, 0], M2[1, 1] - pt2[1] * M2[2, 1],
             M2[1, 2] - pt2[1] * M2[2, 2]]
        ]
        b = [
            pt1[0] * M1[2, 3] - M1[0, 3],
            pt1[1] * M1[2, 3] - M1[1, 3],
            pt2[0] * M2[2, 3] - M2[0, 3],
            pt2[1] * M2[2, 3] - M2[1, 3]
        ]

        x, residual, _, _ = np.linalg.lstsq(A, b, rcond=None)
        residuals[idx] = residual
        points3d[idx] = x

    # Initial random values for 3D points
    mask = (residuals < threshold)
    points3d_inlier = points3d[mask]
    points2d_1_inlier = points1[mask] # only modify if using threshold
    points2d_2_inlier = points2[mask] # only modify if using threshold

    return points3d_inlier, points2d_1_inlier, points2d_2_inlier

#/////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    plt.figure(1, figsize = (8, 8))
    plt.subplot(211)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(212)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()
    plt.show()
