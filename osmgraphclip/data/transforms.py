import torch


def train_transform(sample):
    sample["coords"] = coordinate_jitter(sample["coords"])
    return sample


def get_train_transform():
    return train_transform


def coordinate_jitter(
    coords,
    radius=0.01,  # approximately up to ~1 km in latitude degrees
):
    noise = (torch.rand(coords.shape, dtype=coords.dtype) * 2.0 - 1.0) * radius
    return coords + noise