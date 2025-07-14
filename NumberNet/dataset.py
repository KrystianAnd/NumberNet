import torch
from torchvision import datasets, transforms


def mnist_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def svhn_transform():
    return transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_mnist_dataset():
    train = datasets.MNIST(
        root="data", train=True, download=True, transform=mnist_transform()
    )
    test = datasets.MNIST(
        root="data", train=False, download=True, transform=mnist_transform()
    )
    return train, test


def get_svhn_dataset():
    train = datasets.SVHN(
        root="data", split="train", download=True, transform=svhn_transform()
    )
    test = datasets.SVHN(
        root="data", split="test", download=True, transform=svhn_transform()
    )

    train.labels = torch.tensor([0 if label == 10 else label for label in train.labels])
    test.labels = torch.tensor([0 if label == 10 else label for label in test.labels])

    return train, test
