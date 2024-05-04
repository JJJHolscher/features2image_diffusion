from pathlib import Path

import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset

from jo3mnist.imagenet import ImageNet


def load_data(
    *, feature_dir, image_dir, batch_size, image_size, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    dataset = ImageNetSet(
        feature_dir,
        image_dir,
        image_size=image_size,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class ImageNetSet(Dataset):
    def __init__(self, features_path, imagenet_path, image_size=224):
        self.imagenet = ImageNet(imagenet_path, image_size=image_size)
        self.paths = [p for p in Path(features_path).glob("[0-9]*.npy")]

    def __getitem__(self, i):
        features = jnp.load(self.paths[i], allow_pickle=False).flatten()
        image, label = self.imagenet[i]
        return features, image, label

    def __len__(self):
        return len(self.paths)
