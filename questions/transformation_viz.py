import numpy as np
import matplotlib.pyplot as plt
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--translation',
#         action='store_true',
#         help='If true, then the transformation matrix should be 2x3. And in this case you should pad the starting points with a row of ones'
#     )
#     return parser.parse_args()


def calculate_matrix(starting_points, end_points):
    """
    This function should calculate the transformation matrix using the given
    starting and ending points. We recommend using the least squares solution
    to solve for the transformation matrix. See the handout, Q1 of the written
    questions, or the lecture slides for how to set up these equations.
    If we are using an affine transformation, then the transformation matrix
    should be 2x3. And in this case you should pad the starting points with a
    row of ones If we are not using an affine transformation (no translation),
    then the transformation matrix should be 2x2.

    :param starting_points: 2xN array of points in the starting image
    :param end_points: 2xN array of points in the ending image
    :return: 2X2 or 2X3 matrix M such that M * starting_points = end_points
    """
    #### start student code ####
    # we are solving for the matrix M such that M * starting_points = end_points
    # we can rewrite this as M * starting_points - end_points = 0
    ## TODO 1: transform the point coordinates to the A matrix and b vector
    ## for question 1(d)
    ## for affine transformation, you will be dealing with a transformation matrix of 6 parameters
    ## think about how to change the shape of A to accomodate the extra parameters
    A = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    b = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])

    ## TODO 2: solve for the least squares solution (use the np.linalg.lstsq function)
    x, residual = np.linalg.lstsq(A, b, rcond=5)[:2]
    ## TODO 3: reshape the x vector into a square matrix
    x = x.reshape(2, 2)
    return x, residual


def transform(starting_points, transformation_matrix):
    return transformation_matrix @ starting_points


def main():
    starting_points = np.array([[1, 1.5, 2, 2.5], [1, 0.5, 1, 2]])

    # TODO: use the primed coordinates from the written hw.
    # See how we do this for starting points to do it for end_points.
    end_points = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

    ## fill in your computation here
    transformation_matrix, residual = calculate_matrix(starting_points, end_points)
    print(f"The residual of your transformation is {residual}")

    transformed_points = transform(starting_points, transformation_matrix)
    print(transformed_points)

    fig, ax = plt.subplots()

    # plot the transformed results
    # the points transformed by your matrix wound not perfectly match the end points
    # this is expected because the result returned by calculate_matrix is the least squares solution, which of course comes with residuals
    ax.fill(
        starting_points[0],
        starting_points[1],
        color="blue",
        alpha=0.5,
        label="Starting Points",
    )
    ax.fill(end_points[0], end_points[1], alpha=0.5, color="red", label="End Points")
    # ax.fill(default_transformed_points[0], default_transformed_points[1], color='orange', alpha=0.5, label=default_transformation_names[default_transformation_idx])
    ax.fill(
        transformed_points[0],
        transformed_points[1],
        color="green",
        alpha=0.5,
        label="Your Transformation",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
