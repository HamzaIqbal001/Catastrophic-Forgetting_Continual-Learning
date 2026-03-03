import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import random


def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def split_dataset(dataset):
    task_A_indices = []
    task_B_indices = []

    for idx, (_, label) in enumerate(dataset):
        if label <= 4:
            task_A_indices.append(idx)
        else:
            task_B_indices.append(idx)

    return Subset(dataset, task_A_indices), Subset(dataset, task_B_indices)


def create_replay_buffer(dataset, buffer_size):
    indices = random.sample(range(len(dataset)), buffer_size)
    return Subset(dataset, indices)