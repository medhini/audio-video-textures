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


def compute_D1(
    frames: torch.Tensor,
    sigma_factor: float,
    feats: str = "L2",
    audio: np.ndarray = None,
    sr: int = 0,
    fps: int = 30,
    slow: bool = True,
    batch_size: int = 128,
) -> np.array:
    """
    Computes the L2 distances between all pairs of frames in the video. 
    
    Args:
        frames (np.array): frames of shape [N, H, W, C]
        
    Returns:
        prob (np.array): Matrix of shape [N,N] containing L2 distances between all frames 
    
    """
    if feats == "RGB":
        # use torch pairwise distance instead; also move all of this to inside a nn.Module class
        if not slow:
            frames = frames.cuda()
            A = frames.unsqueeze(0).repeat(frames.shape[0], 1, 1, 1, 1)
            A = A.view(frames.shape[0], frames.shape[0], -1)
            B = frames.unsqueeze(1).repeat(1, frames.shape[0], 1, 1, 1)
            B = B.view(frames.shape[0], frames.shape[0], -1)

            D1 = torch.norm(A - B, dim=2)

        else:
            D1 = torch.ones((len(frames), len(frames))).cuda()
            #             D1 = torch.zeros((len(frames), len(frames))).cuda()
            for i in range(0, len(frames), batch_size):
                frames_batch_A = frames[i : min(i + batch_size, len(frames))].cuda()
                for j in range(0, len(frames), batch_size):
                    feats_A = frames_batch_A.unsqueeze(0).repeat(
                        min(batch_size, len(frames) - j), 1, 1, 1, 1
                    )
                    feats_A = feats_A.view(
                        min(batch_size, len(frames) - j),
                        min(batch_size, len(frames) - i),
                        -1,
                    )

                    frames_batch_B = frames[j : min(j + batch_size, len(frames))].cuda()
                    feats_B = frames_batch_B.unsqueeze(1).repeat(
                        1, min(batch_size, len(frames) - i), 1, 1, 1
                    )
                    feats_B = feats_B.view(
                        min(batch_size, len(frames) - j),
                        min(batch_size, len(frames) - i),
                        -1,
                    )
                    if feats_A.shape != feats_B.shape:
                        continue

                    # feats_A = nn.functional.normalize(feats_A, dim=2)
                    # feats_B = nn.functional.normalize(feats_B, dim=2)

                    D_mini = torch.norm(feats_A - feats_B, dim=2).detach().cpu()

                    # print('Feats A shape:', feats_A.shape)
                    # print('Feats B shape:', feats_B.shape)
                    # print('Dmini shape:', D_mini.shape)

                    D1[i : i + batch_size, j : j + batch_size] = D_mini.permute(1, 0)
                    del D_mini
                    torch.cuda.empty_cache()

    elif feats == "ResNet":
        resnet18 = models.resnet18(pretrained=True)
        resnet18.cuda()
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        resnet18.eval()  # ignoring dropout

        if not slow:
            frames = frames.cuda()
            with torch.no_grad():
                feats = resnet18(frames)
            feats = feats.view(feats.shape[0], feats.shape[1])
            feats_A = feats.unsqueeze(0).repeat(feats.shape[0], 1, 1)
            feats_B = feats.unsqueeze(1).repeat(1, feats.shape[0], 1)

            feats_A = nn.functional.normalize(feats_A, dim=2)
            feats_B = nn.functional.normalize(feats_B, dim=2)

            D1 = torch.norm(feats_A - feats_B, dim=2)
        else:
            # D1 = torch.zeros((len(frames), len(frames), 512)).cuda()
            D1 = torch.ones((len(frames), len(frames))).cuda()
            for i in range(0, len(frames), batch_size):
                with torch.no_grad():
                    image_feats_A = resnet18(frames[i : i + batch_size].cuda())
                image_feats_A = image_feats_A.view(
                    image_feats_A.shape[0], image_feats_A.shape[1]
                )
                image_feats_A = image_feats_A.unsqueeze(0).repeat(batch_size, 1, 1)
                for j in range(0, len(frames) - batch_size, batch_size):
                    with torch.no_grad():
                        image_feats_B = resnet18(frames[j : j + batch_size].cuda())
                    image_feats_B = image_feats_B.view(
                        image_feats_B.shape[0], image_feats_B.shape[1]
                    )
                    image_feats_B = image_feats_B.unsqueeze(1).repeat(1, batch_size, 1)
                    if image_feats_A.shape != image_feats_B.shape:
                        continue

                    image_feats_A = nn.functional.normalize(image_feats_A, dim=2)
                    image_feats_B = nn.functional.normalize(image_feats_B, dim=2)

                    # do torch.norm here as storing D and doing it later needs too much mem
                    D_mini = torch.norm(image_feats_A - image_feats_B, dim=2)

                    D1[i : i + batch_size, j : j + batch_size] = (
                        D_mini.permute(1, 0).detach().cpu()
                    )
                    del image_feats_B, D_mini
                    torch.cuda.empty_cache()

                del image_feats_A
                torch.cuda.empty_cache()

    elif feats == "ResNet_VGGish":
        # Audio Features
        vggish = torch.hub.load("harritaylor/torchvggish", "vggish")
        vggish.eval()  # ignoring dropout
        audio_feats = vggish.forward(audio, sr).cuda()
        audio_feats = audio_feats[: int(len(frames) / fps)]
        audio_feats = audio_feats.repeat(fps, 1)
        print("Shape of audio feats:", audio_feats.shape)

        # Image Features
        frames = frames[: int(len(frames) / fps) * fps]

        resnet18 = models.resnet18(pretrained=True)
        resnet18.cuda()
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        resnet18.eval()  # ignoring dropout

        if not slow:
            frames = frames.cuda()
            with torch.no_grad():
                feats = resnet18(frames)
            feats = feats.view(feats.shape[0], feats.shape[1])

            image_feats = feats.view(feats.shape[0], feats.shape[1])
            print("Shape of image feats:", image_feats.shape)

            joint_feats = torch.cat((image_feats, audio_feats), dim=1)
            print("Shape of joint feats:", joint_feats.shape)

            feats_A = joint_feats.unsqueeze(0).repeat(joint_feats.shape[0], 1, 1)
            feats_B = joint_feats.unsqueeze(1).repeat(1, joint_feats.shape[0], 1)

            feats_A = nn.functional.normalize(feats_A, dim=2)
            feats_B = nn.functional.normalize(feats_B, dim=2)

            D1 = torch.norm(feats_A - feats_B, dim=2)

        else:
            # D1 = torch.zeros((len(frames), len(frames), 512)).cuda()
            D1 = torch.zeros((len(frames), len(frames))).cuda()
            for i in range(0, len(frames), batch_size):
                frames_A = frames[i : i + batch_size].cuda()
                with torch.no_grad():
                    image_feats_A = resnet18(frames_A)
                image_feats_A = image_feats_A.view(
                    image_feats_A.shape[0], image_feats_A.shape[1]
                )
                image_feats_A = image_feats_A.unsqueeze(0).repeat(batch_size, 1, 1)
                audio_feats_A = (
                    audio_feats[i : i + batch_size]
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                )
                joint_feats_A = torch.cat((image_feats_A, audio_feats_A), dim=2)

                for j in range(0, len(frames) - batch_size, batch_size):
                    frames_B = frames[j : j + batch_size].cuda()
                    with torch.no_grad():
                        image_feats_B = resnet18(frames_B)
                    image_feats_B = image_feats_B.view(
                        image_feats_B.shape[0], image_feats_B.shape[1]
                    )
                    image_feats_B = image_feats_B.unsqueeze(1).repeat(1, batch_size, 1)
                    audio_feats_B = (
                        audio_feats[j : j + batch_size]
                        .unsqueeze(1)
                        .repeat(1, batch_size, 1)
                    )
                    joint_feats_B = torch.cat((image_feats_B, audio_feats_B), dim=2)

                    if joint_feats_A.shape != joint_feats_B.shape:
                        continue

                    # feats_A = nn.functional.normalize(feats_A, dim=2)
                    # feats_B = nn.functional.normalize(feats_B, dim=2)

                    D_mini = (
                        torch.norm(joint_feats_A - joint_feats_B, dim=2).detach().cpu()
                    )
                    D1[i : i + batch_size, j : j + batch_size] = D_mini.permute(1, 0)

                    del frames_B, image_feats_B, audio_feats_B, joint_feats_B, D_mini
                    torch.cuda.empty_cache()

                del frames_A, image_feats_A, audio_feats_A, joint_feats_A
                torch.cuda.empty_cache()

    non_zero_count = torch.nonzero(D1).size(0)
    sigma = sigma_factor * (D1.sum() / non_zero_count)
    P1 = torch.exp(-D1 /sigma)
    P1 = torch.cat((P1[1:, :], P1[-1, :].unsqueeze(0)), dim=0)

    P1 = P1 / P1.sum(1, keepdim=True)

    return D1, P1, sigma

