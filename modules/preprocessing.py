import math
import matplotlib.pyplot as plt
from matplotlib import cm


def draw_annot(image, annot):
    """Plot the center point and represnt the angle using a line

    Args:
        image (numpy.ndarray): image to plot
        annot (numpy.ndarray): image annotation
    """
    x_0, y_0 = annot[:, 1], annot[:, 2]
    gamma = annot[:, 3]
    phi = annot[:, 4] / (math.pi)

    fig = plt.figure(figsize=[6, 6])
    plt.imshow(image)
    plt.scatter(x_0, y_0, color="r", s=75, marker="o")

    line_len_a = 20
    line_len_b = 5
    cmap = cm.get_cmap("bwr")
    for num in range(annot.shape[0]):
        x_1 = x_0[num] - line_len_a * math.cos(gamma[num])
        y_1 = y_0[num] - line_len_a * math.sin(gamma[num])
        x_2 = x_0[num] + line_len_b * math.cos(gamma[num])
        y_2 = y_0[num] + line_len_b * math.sin(gamma[num])

        color_factor = phi[num]
        plt.plot([x_1, x_2], [y_1, y_2], c=cmap(color_factor))
        # plt.plot([x_1, x_2], [y_1, y_2], c= "r")

    plt.axis("off")
    plt.show()
