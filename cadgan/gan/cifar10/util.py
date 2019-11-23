"""
Utility functions for Cifar10 dataset.
"""
import cadgan.glo as glo
import cadgan.util as util
import numpy as np
import torch
import torchvision
from torchvision import transforms


def label_class_list():
    """
    Return a list of tuples [(a, b)]
    where a is a class index (0-9), and b is a string describing the class.
    """
    cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    cifar10_class_inds = list(zip(range(10), cifar10_classes))
    return cifar10_class_inds


def load_cifar10_dataset(train=True):
    """
    Load and return the CIFAR10 dataset as an instance of 
    torchvision.datasets.cifar.CIFAR10.
    """
    # load data
    trdata_folder = glo.data_file("cifar10")
    trdata = torchvision.datasets.CIFAR10(
        trdata_folder,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                #                            transforms.Normalize((0.1307,), (0.3081,))
            ]
        ),
    )
    return trdata


def load_cifar10_class_subsets(classes, train=True, device="cpu", dtype=torch.float, minmax=(0, 1.0)):
    """
    Load and return a subset of the CIFAR10 dataset, including only the
    specified classes.

    - classes: a list of class indices (as returned by label_class_list()
    - minmax: transform the range of pixel intensity to satisfy this range.
    """
    trdata = load_cifar10_dataset(train)
    # ntr = trdata.train_data.shape[0]

    # Pick out the specified classes
    # numpy arrays
    X = trdata.train_data
    Y = np.array(trdata.train_labels)

    # filter data according to the chosen classes
    tr_inds = [Y[i] in classes for i in range(len(Y))]
    Xtr = X[tr_inds]
    Ytr = Y[tr_inds]
    Xtr = util.linear_range_transform(Xtr, (0, 255), minmax)
    Ytr = util.linear_range_transform(Ytr, (0, 255), minmax)

    TXtr = torch.tensor(Xtr.transpose(0, 3, 1, 2), device=device, dtype=dtype)
    TYtr = torch.tensor(Ytr, device=device, dtype=dtype)
    Tr = torch.utils.data.TensorDataset(TXtr, TYtr)
    return Tr
