import os
import librosa
import numpy as np
from PIL import Image
from scipy import signal
import subprocess
import time
import random
import IPython.display as ipd
from multiprocessing import Pool
import argparse
import copy

import torch
import torchvision.io as io
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def q_learning(
    D2: torch.Tensor,
    sigma_factor: float,
    p: float = 0.7,
    alpha: float = 0.997,
    thresholding: float = 0.75
) -> np.array:
    D3 = D2 ** p

    eps = 10000
    D3_new = copy.deepcopy(D3)

    while eps > 10e-3:
        D3_old = copy.deepcopy(D3_new)
        # iterating from last row to first
        for i in range(D3.shape[0] - 1, 0, -1):
            mask = np.ones((D3.shape[0], D3.shape[1]), dtype=bool)
            np.fill_diagonal(mask, False)

            mins = D3_old[mask, ...].view(D3.shape[0], -1).min(axis=1)[0]

            D3_new[i] = D3[i] + alpha * mins

        eps = ((D3_new - D3_old) ** 2).mean()
        print("Eps:", eps)

    non_zero_count = torch.nonzero(D3_new).size(0)
    sigma = sigma_factor * (D3_new.sum() / non_zero_count)

    P3 = torch.exp(-D3_new / sigma)
    P3 = torch.cat((P3[1:, :], P3[-1, :].unsqueeze(0)), dim=0)

    P3 = P3 / P3.sum(1, keepdim=True)

    P3_new = copy.deepcopy(P3)

    for i in range(len(P3_new)):  # thresholding P3
        P3_new[i][P3_new[i] < (P3_new[i].max() - thresholding * P3_new[i].max())] = 0.0

    print("Non Zero in P3:", len(P3_new[0].nonzero()))

    return D3_new, P3, P3_new, sigma
