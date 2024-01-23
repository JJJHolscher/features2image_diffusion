#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm
from jo3mnist.vis import to_img
from jo3util.warning import todo
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST



class MnistFeaturesSet(Dataset):
    """The MNIST set that also yields the features associated with the image.

    When getting an item from this, it returns `features, image, label`
    """

    def __init__(self, features_path, mnist_path, train=True, transform=None):
        self.mnist = MNIST(
            mnist_path, train=train, download=True, transform=transform
        )

        train = "train" if train else "test"
        todo("Make sure file names are numeric before the .npy.")
        paths = Path(features_path).glob(train + "/[0-9]*.npy")
        self.paths = [p for p in paths]

        assert len(self.paths) == len(self.mnist)

    def __getitem__(self, i):
        features = np.load(self.paths[i], allow_pickle=False)
        image, label = self.mnist[i]
        return features, image, label

    def __len__(self):
        return len(self.paths)


def load_mnist_with_features(
    features_path: Union[Path, str],
    mnist_path: Union[Path, str],
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """ Load train and test dataloaders that yield features, image and label.
    """
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = MnistFeaturesSet(
        features_path, mnist_path, train=True, transform=normalise_data
    )
    test_set = MnistFeaturesSet(
        features_path, mnist_path, train=False, transform=normalise_data
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader


def loader_to_dataframe(data_loader, img_dir="./res/mnist/img"):
    data_len = len(data_loader)
    img_dir = Path(img_dir)
    if not img_dir.exists():
        print("[0/1] storing mnist images")
        img_dir.mkdir(parents=True, exist_ok=True)
        iterator = tqdm(enumerate(data_loader), total=data_len)
        for i, (_, images, _) in iterator:
            for j, img in enumerate(images):
                index = i * len(images) + j
                to_img(img).save(img_dir / f"{index}.png")

    # Empty the data_loader iterator into tensors.
    print("[1/1] loading features and labels")
    all_features = []
    all_labels = []
    iterator = tqdm(enumerate(data_loader), total=data_len)
    for i, (features, _, labels) in iterator:
        all_features.append(torch.Tensor(features))
        all_labels.append(torch.Tensor(labels))

    dataframe = {"label": torch.concatenate(all_labels)}

    # Create a column per feature.
    all_features = torch.concatenate(all_features)
    feature_len = all_features.shape[-1]
    for f in range(feature_len):
        dataframe[f"f{f}"] = all_features[:, f]

    return pd.DataFrame(dataframe)
