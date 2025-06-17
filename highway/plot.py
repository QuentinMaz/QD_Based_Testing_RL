import io
from typing import List, Union

import matplotlib
import numpy as np
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap, ListedColormap, Normalize
from PIL import Image

POINT_SIZE = 5

FAILURE_COLOR = "xkcd:fire engine red"
PASS_COLOR = "xkcd:shamrock green"
BLUE_COLOR = "xkcd:deep sky blue"

MARKER_SIZE_LEGEND = 75.0
LABEL_FONTSIZE = 15


def plot_line(data):
    """
    Displays the values as a line.

    Returns:
        figure and its axis.
    """
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(data)), data, color=BLUE_COLOR)
    return fig, ax


def plot_summary(forces, rewards, failures):
    """
    Displays on the three plots:
        - The force distribution.
        - The reward distribution.
        - The pass and failed tests.

    Returns:
        figure and its axes.
    """
    fig, axs = plt.subplots(ncols=3, figsize=(15, 5))
    ax = axs[0]
    ax.scatter(forces[:, 0], forces[:, 1], color=BLUE_COLOR, s=POINT_SIZE)
    ax = axs[1]
    im = ax.scatter(forces[:, 0], forces[:, 1], c=rewards, s=POINT_SIZE)
    fig.colorbar(im, ax=ax, orientation="vertical", label="Rewards")
    ax = axs[2]

    ax.scatter(
        forces[failures, 0],
        forces[failures, 1],
        color=FAILURE_COLOR,
        s=POINT_SIZE,
        label="Failures",
    )
    ax.scatter(
        forces[~failures, 0],
        forces[~failures, 1],
        color=PASS_COLOR,
        s=POINT_SIZE,
        label="Pass",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=2,
        framealpha=1.0,
        prop={"size": 13},
        # labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("lightgray")
    legend_frame.set_edgecolor("black")
    for handle in legend.legend_handles:
        handle.set_sizes([MARKER_SIZE_LEGEND])

    fig.tight_layout()
    return fig, axs


def plot_correlation(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    return fig, ax


# https://stackoverflow.com/a/59222785
def figure_to_buffer(figure):
    """
    Saves a (matplotlib) Figure to a buffer and closes the Figure.
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(figure)
    # changes the stream position to the start of the file
    # not sure why it is useful nor if it is used or necessary
    buffer.seek(0)
    return buffer


def figure_to_image_array(figure) -> np.ndarray:
    """
    Converts a (matplotlib) Figure into a Numpy array using PIL.
    Note that the Figure is closed in the process.

    Returns:
        - image_arr (np.ndarray): Image of the figure of shape (height, width, rgba_color).
    """
    buffer = figure_to_buffer(figure)
    image = Image.open(buffer)
    image_arr = np.array(image)

    image.close()
    buffer.close()

    return image_arr


def plot_points_with_oracle_results(points: np.ndarray, values: np.ndarray):
    """
    Displays on a single plot 2d points colored by `values`.

    Returns:
        figure and its axis.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(
        points[values, 0],
        points[values, 1],
        color=FAILURE_COLOR,
        s=POINT_SIZE,
        label="Failures",
    )
    ax.scatter(
        points[~values, 0],
        points[~values, 1],
        color=PASS_COLOR,
        s=POINT_SIZE,
        label="Pass",
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=2,
        framealpha=1.0,
        prop={"size": 13},
        # labelspacing=1.0, handletextpad=1.0, borderpad=0.55, borderaxespad=0.7)
    )
    legend_frame = legend.get_frame()
    legend_frame.set_facecolor("lightgray")
    legend_frame.set_edgecolor("black")
    for handle in legend.legend_handles:
        handle.set_sizes([MARKER_SIZE_LEGEND])
    return fig, ax


def get_colors_from_cmap(n: int = 2, cmap: Union[str, Colormap] = plt.cm.jet):
    """
    Return a list of `n` RGBA colors by sampling evenly in `cmap` (default to "jet").
    The color values are floating points between 0 and 1.
    """
    if isinstance(cmap, str):
        cmap: Colormap = plt.cm.get_cmap(cmap)
    return [cmap(i) for i in np.linspace(0, 1, n)]


def save_gif(images: List[np.ndarray], filepath: str, duration: float = 30.0):
    """Saves a sequence of images as `.gif` file. **No input checking nor exception handling are implemented**."""
    if len(images) == 0:
        return
    imgs = [Image.fromarray(img) for img in images]
    imgs[0].save(filepath, save_all=True, append_images=imgs[1:], duration=duration)
