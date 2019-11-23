"""
Utility functions for Mnist dataset.
"""

import numpy as np
import torch


def pt_sample_by_labels(data, label_counts):
    """
    data: a dataset such that data[i][0] is a point, and data[i][1] is an
        integer label.
    label_counts: a list of tuples of two values (A, B), where A is a label,
        and B is the count.
    """
    list_selected = []
    labels = np.array([data[i][1] for i in range(len(data))])
    for label, count in label_counts:
        inds = np.where(labels == label)[0]
        homo_data = [data[i][0] for i in inds[:count]]
        list_selected.extend(homo_data)
    # stack all
    selected = torch.stack(list_selected)
    return selected
