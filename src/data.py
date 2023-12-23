#! /usr/bin/env python3
# vim:fenc=utf-8

from pathlib import Path
from jo3util.warning import todo
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST


class MnistFeaturesSet(Dataset):

    def __init__(self, features_path, mnist_path, train=True, transform=None):
        self.mnist = MNIST(
            mnist_path,
            train=train,
            download=True,
            transform=transform
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
    features_path,
    mnist_path,
    batch_size,
    shuffle=True
):
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_set = MnistFeaturesSet(
        features_path,
        mnist_path,
        train=True,
        transform=normalise_data
    )
    test_set = MnistFeaturesSet(
        features_path,
        mnist_path,
        train=False,
        transform=normalise_data
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return train_loader, test_loader
