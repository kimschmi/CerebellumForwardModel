import numpy as np


def compute_end_effector_positions(arm_length, theta0, theta1):
    """Compute end-effector positions of 2-joint planar arm for different joint angles.

    End-effector positions are computed according to
    x = l_0 * cos(theta0) + l_1 * cos (theta0 + theta1)
    y = l_0 * sin(theta0) + l_1 * sin(theta0 + theta1)

    :param arm_length: length of arm segments [l_0, l_1]
    :param theta0: angles of first joint in radians
    :param theta1: angles of second joint in radians
    :return: end-effector positions x, y
    """

    transform_x = np.zeros((theta0.size, 2))
    transform_x[:, 0] = np.cos(theta0)
    transform_x[:, 1] = np.cos(theta0 + theta1)
    transform_y = np.zeros((theta0.size, 2))
    transform_y[:, 0] = np.sin(theta0)
    transform_y[:, 1] = np.sin(theta0 + theta1)
    x = np.dot(transform_x, arm_length)
    y = np.dot(transform_y, arm_length)
    return x, y
