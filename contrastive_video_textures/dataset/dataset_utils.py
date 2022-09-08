import numpy as np
import os
import random
import time
from collections import defaultdict
import torch

from . import transform as transform


def scale_jitter_crop_norm(
    frames,
    scale_height=240,
    scale_width=240,
    crop_size=224,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    split="train",
):
    """
    Performs scaling, random crop, color jitter and normalization on the given video 
    frames. 
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        scale_height (int): the height of scaling.
        scale_width (int): the width of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        mean (array): mean for normalization.
        std (array): std dev for normalization.
        split (str): data split. default 'train'.
    Returns:
        frames (tensor): augmented frames.
    """
    frames = frames.float()
    frames = frames / 255.0

    # Scale.
    frames = torch.nn.functional.interpolate(
        frames, size=(scale_height, scale_width), mode="bilinear", align_corners=False,
    )

    if split == "train":
        # Random crop.
        frames = transform.random_crop(frames, crop_size)

        # Do color augmentation and clamp.
        frames = transform.color_jitter(
            frames, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4,
        )
        frames = torch.clamp(frames, 0.0, 1.0)
    else:
        # Uniform crop.
        frames = transform.uniform_crop(frames, crop_size)

    # Normalize images by mean and std.
    frames = transform.color_normalization(
        frames, np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32),
    )

    return frames
