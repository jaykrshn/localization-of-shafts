import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def preprocess_annot(annot_path, image_shape):
    """Load image annotation as a numpy array

    Args:
        annot_path (path): Path of image annotation
        image_shape (tuple): Shape of the image

    Returns:
        numpy.ndarray: Annotation as numpy array
    """
    annot = np.loadtxt(annot_path)
    # print(image_shape)
    # print(annot.shape)
    # annot = np.delete(annot, obj=[], axis=0)

    return annot


def filter_annot(annot_path, check_params):

    annot = np.loadtxt(annot_path)
    # print(image_shape)
    # print(annot.shape)
    # annot = np.delete(annot, obj=[], axis=0)

    return annot


def draw_annot(image, annot):
    """Plot the center point and represent the angle using a line
    TODO: Remove line_len_b option. Use x_0, y_0 instead.

    Args:
        image (numpy.ndarray): image to plot
        annot (numpy.ndarray): image annotation
    """
    x_0, y_0 = annot[:, 1], annot[:, 2]
    # gamma has a range (0, 2π)
    gamma = annot[:, 3]
    # phi has a range (0, π)
    phi = annot[:, 4] / (math.pi)

    plt.figure(figsize=[6, 6])
    plt.imshow(image)
    plt.scatter(x_0, y_0, color="r", s=40, marker="o")

    line_len = 25
    cmap = cm.get_cmap("bwr")
    for num in range(annot.shape[0]):
        x_1 = x_0[num] - line_len * math.cos(gamma[num])
        y_1 = y_0[num] - line_len * math.sin(gamma[num])
        color_factor = phi[num]
        plt.plot([x_1, x_0[num]], [y_1, y_0[num]], c=cmap(color_factor))

    # plt.axis("off")
    plt.xticks(np.arange(0, 128, 8))
    plt.yticks(np.arange(0, 128, 8))
    plt.grid()
    plt.show()
