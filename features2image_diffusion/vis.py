#! /usr/bin/env python3
# vim:fenc=utf-8

from jaxtyping import Float
from jo3mnist.vis import to_img
import matplotlib.pyplot as plt
import numpy as np


def tabulate_generations(
    generations: Float[np.ndarray, "num_generations 28 28"],
    original_img=None
):
    # Determine the shape of the table of images.
    num_images = len(generations)
    if original_img is not None:
        num_images += 1
    rows = int(np.sqrt(num_images))
    cols = num_images // rows
    if rows * cols < num_images:
        cols += 1
    fig, axs = plt.subplots(rows, cols, facecolor="gray")

    # Place the images in the table.
    for i, generation in enumerate(generations):
        r = i // cols
        c = i % cols
        if len(generation.shape) == 3:
            generation = (generation + 2.1008) / (2.64 + 2.1008)
            axs[r, c].imshow(np.transpose(generation, (1, 2, 0)))
        else:
            axs[r, c].imshow(to_img(generation))
        axs[r, c].set_axis_off()
    if original_img is not None:
        axs[-1, -1].imshow(to_img(original_img))
        axs[-1, -1].set_axis_off()

    return fig, axs
