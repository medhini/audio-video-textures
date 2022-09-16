#!/usr/bin/env python3
import argparse
import os
import os.path
from shutil import rmtree, move
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import platform

from models import UNet, backWarp

CHEKCPOINT = "SuperSloMo.ckpt"
FFMPEG_DIR = "/home/medhini/bin"
FPS = 30
# specify the slomo factor N. This will increase the frames by Nx. Example sf=2 ==> 2x frames
# SF = 10
BATCH_SIZE = 1


def pil_augment(img, cropArea=None, resizeDim=None, frameFlip=0):
    """
    Applies data augmentation to image.

    Parameters
    ----------
        img : PIL image
           Image.
        cropArea : tuple, optional
            coordinates for cropping image. Default: None
        resizeDim : tuple, optional
            dimensions for resizing image. Default: None
        frameFlip : int, optional
            Non zero to flip image horizontally. Default: 0

    Returns
    -------
        list
            2D list described above.
    """

    # Resize image if specified.
    resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
    # Crop image if crop area specified.
    cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
    # Flip image horizontally if specified.
    flipped_img = (
        cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
    )
    return flipped_img.convert("RGB")


def modify_frames(frame0, frame1):
    # Initialize transforms
    mean = [0.429, 0.431, 0.397]
    std = [1, 1, 1]
    normalize = transforms.Normalize(mean=mean, std=std)

    negmean = [x * -1 for x in mean]
    revNormalize = transforms.Normalize(mean=negmean, std=std)

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    TP = transforms.Compose([revNormalize, transforms.ToPILImage()])

    # Get dimensions of frames
    origDim = frame0.size
    dim = int(origDim[0] / 32) * 32, int(origDim[1] / 32) * 32

    frame0 = pil_augment(frame0, resizeDim=dim)
    frame0 = transform(frame0)

    frame1 = pil_augment(frame1, resizeDim=dim)
    frame1 = transform(frame1)

    return frame0, frame1, TP


class interpolate(nn.Module):
    def __init__(self, origDim, SF):
        super(interpolate, self).__init__()
        # Initialize interpolation model
        self.flowComp = UNet(6, 4)
        for param in self.flowComp.parameters():
            param.requires_grad = False
        self.ArbTimeFlowIntrp = UNet(20, 5)
        for param in self.ArbTimeFlowIntrp.parameters():
            param.requires_grad = False

        self.SF = SF
        # Get dimensions of frames
        self.origDim = origDim
        self.dim = int(self.origDim[0] / 32) * 32, int(self.origDim[1] / 32) * 32

        self.flowBackWarp = backWarp(self.dim[0], self.dim[1], device=0)

    def forward(self, frame0, frame1, TP):
        # Interpolate frames
        frameCounter = 1
        output_frames = []
        with torch.no_grad():
            I0 = frame0.unsqueeze(0).cuda()
            I1 = frame1.unsqueeze(0).cuda()

            flowOut = self.flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            # Save reference frames in output folder
            # for batchIndex in range(BATCH_SIZE):
            #     output_frames.append(TP(frame0[batchIndex].detach()).resize(videoFrames.origDim, Image.BILINEAR))
            # frameCounter += 1

            # Generate intermediate frames
            for intermediateIndex in range(1, self.SF):
                t = float(intermediateIndex) / self.SF
                temp = -t * (1 - t)
                fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = self.flowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = self.flowBackWarp(I1, F_t_1)

                intrpOut = self.ArbTimeFlowIntrp(
                    torch.cat(
                        (I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0),
                        dim=1,
                    )
                )

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = torch.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = self.flowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = self.flowBackWarp(I1, F_t_1_f)

                wCoeff = [1 - t, t]

                Ft_p = (
                    wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f
                ) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                # Save intermediate frame
                for batchIndex in range(BATCH_SIZE):
                    output_frames.append(
                        TP(Ft_p[batchIndex].cpu().detach()).resize(
                            self.origDim, Image.BILINEAR
                        )
                    )
                frameCounter += 1

            # Set counter accounting for batching of frames
            frameCounter += self.SF * (BATCH_SIZE - 1)

        return output_frames

