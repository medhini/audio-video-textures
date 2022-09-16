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

def compute_D2(D1: np.array, sigma_factor: float, filter_size: int=16, stride: int=1) -> np.array:
    """
    Convolves D1 with a diagonal kernel of binomial weighs. 
    
    Args:
        D1 (np.array): matrix of shape [N,N]
        
    Returns:
        D2 (np.array): matrix of shape [N,N] 
    
    """
    # binomial_filter = torch.tensor(np.diag(np.random.binomial(100, 0.5, 5)), dtype=torch.float32).cuda()
    # binomial_filter = torch.tensor(np.diag((np.poly1d([0.5, 0.5])**4).coeffs), dtype=torch.float32).cuda()
    binomial_filter = torch.tensor(np.diag((np.poly1d([0.5, 0.5])**(filter_size-1)).coeffs),\
        dtype=torch.float32).cuda()
    
    D2 = D1.view(1,1,D1.shape[0],D1.shape[0])
    binomial_filter = binomial_filter.view(1,1,filter_size,filter_size)

    # P = int((filter_size - 1)/2)
    D2 = F.conv2d(D2, binomial_filter, stride=stride)
    D2 = D2.view(D2.shape[2], D2.shape[3])
    
    non_zero_count = torch.nonzero(D2).size(0)
    sigma = sigma_factor*(D2.sum()/non_zero_count)

    P2 = torch.exp(-D2/sigma)
    P2 = torch.cat((P2[1:,:], P2[-1,:].unsqueeze(0)), dim=0)
    
    P2 = P2/P2.sum(1, keepdim=True)
    
    return D2, P2, sigma, binomial_filter