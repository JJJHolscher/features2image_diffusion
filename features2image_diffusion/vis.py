#! /usr/bin/env python3
# vim:fenc=utf-8

from jaxtyping import Float
from jo3mnist.vis import to_img
import torch
import matplotlib.pyplot as plt
import numpy as np


def old_evaluation_section():
    n_sample = 4*n_classes
    for w_i, w in enumerate(ws_test):
        x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

        # append some real images at bottom, order by class also
        x_real = torch.Tensor(x_gen.shape).to(device)
        for k in range(n_classes):
            for j in range(int(n_sample/n_classes)):
                try:
                    idx = torch.squeeze((c == k).nonzero())[j]
                except:
                    idx = 0
                x_real[k+(j*n_classes)] = x[idx]

        x_all = torch.cat([x_gen, x_real])
        grid = make_grid(x_all*-1 + 1, nrow=10)
        save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
        print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

        if ep % 5 == 0 or ep == int(n_epoch-1):
            # gif of images evolving over time, based on x_gen_store
            fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
            def animate_diff(i, x_gen_store):
                print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                plots = []
                for row in range(int(n_sample/n_classes)):
                    for col in range(n_classes):
                        axs[row, col].clear()
                        axs[row, col].set_xticks([])
                        axs[row, col].set_yticks([])
                        # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                        plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
                return plots
            ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
            ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
            print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")


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
        axs[r, c].imshow(to_img(generation))
        axs[r, c].set_axis_off()
    if original_img is not None:
        axs[-1, -1].imshow(to_img(original_img))
        axs[-1, -1].set_axis_off()

    return fig, axs
