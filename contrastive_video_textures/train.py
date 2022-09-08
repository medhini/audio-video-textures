import ipdb
from utils import (
    AverageMeter,
    Logger,
    overlay_cmap_image,
    waveform_to_examples,
    save_videos,
)
from interpolate import interpolate, modify_frames
import torch
import torchvision.transforms as transforms
import torchvision.io as io
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import shutil
import time
from collections import OrderedDict
from PIL import Image
import subprocess
import librosa
import matplotlib

matplotlib.use("Agg")


def to_cuda(item):
    if isinstance(item[0], list):
        return [[x.cuda() for x in y] for y in item]
    elif isinstance(item, list):
        return [x.cuda() for x in item]
    return item.cuda()


def train(
    train_loader, model, optimizer, args, epoch, tb_logger=None,
):
    """Trains the model epoch times.
    Args:
        train_loder (Dataloader): Train data loader.
        model (nn.Module): Model used for training.
        optimizer (nn.Optimizer): Optimizer. 
        args (parser.Arguments): Arguments.
        epoch (int): Number of epochs.
        tb_logger (Logger): Tensorboard logger.
        inv_t (Transforms): Inverse transformation to frames.
    Returns:
        losses_avg (int): Avg. train loss. 
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode.
    model.train()

    # Define criterion.
    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    for i, batch_data in enumerate(train_loader):

        (
            q_frames,
            q_audio_wav,
            q_audio_eg,
            t_frames,
            t_audio_wav,
            t_audio_eg,
        ) = batch_data

        # # Split input data into batches divisible by num_gpus.
        # num_gpus = torch.cuda.device_count()

        # q_f_size = [num_gpus] + [1] * (q_frames.dim() - 1)
        # q_frames = q_frames.repeat(*q_f_size)
        # q_aw_size = [num_gpus] + [1] * (q_audio_wav.dim() - 1)
        # q_audio_wav = q_audio_wav.repeat(*q_aw_size)
        # q_ae_size = [num_gpus] + [1] * (q_audio_eg.dim() - 1)
        # q_audio_eg = q_audio_eg.repeat(*q_ae_size)

        # t_frames = t_frames.view(
        #     num_gpus,
        #     -1,
        #     t_frames.shape[2],
        #     t_frames.shape[3],
        #     t_frames.shape[4],
        #     t_frames.shape[5],
        # )

        # t_audio_wav = t_audio_wav.view(num_gpus, -1, t_audio_wav.shape[2])
        # t_audio_eg = t_audio_eg.view(
        #     num_gpus, -1, t_audio_eg.shape[2], t_audio_eg.shape[3], t_audio_eg.shape[4],
        # )

        # move to gpu
        q_frames = to_cuda(q_frames)
        # q_audio_wav = q_audio_wav.cuda()
        q_audio_eg = q_audio_eg.cuda()

        t_frames = to_cuda(t_frames)
        # t_audio_wav = t_audio_wav.cuda()
        t_audio_eg = t_audio_eg.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(
            q_frames, t_frames, q_audio_eg=q_audio_eg, t_audio_eg=t_audio_eg,
        )

        if isinstance(q_frames, list):
            # set q_frames and t_frames to be slow features.

            # (B, C, T, H, W) -> (B, T, C, H, W)
            q_frames = q_frames[0].permute(0, 2, 1, 3, 4)

            # (B, #targets, C, T, H, W) -> (B, #targets, T, C, H, W)
            t_frames = t_frames[0].permute(0, 1, 3, 2, 4, 5)

        batch_size = q_frames.shape[0]
        # Labels: positive key indicators.
        labels = torch.zeros(batch_size, dtype=torch.long).cuda()

        # Compute loss.
        loss = criterion(output, labels)

        # measure accuracy and record loss
        loss = loss.mean()
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for name, param in model.state_dict().items():
        #     print(name, param.grad)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )

        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs["Train_IterLoss"] = losses.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            # Extract first query and target frames for logging.
            q_frames = q_frames[0]
            t_frames = t_frames[0]

            # Apply inverse normalization.
            if args.enc_arch == "slowfast":
                inv_t = transforms.Normalize(
                    mean=[-0.45 / 0.225, -0.45 / 0.225, -0.45 / 0.225],
                    std=[1 / 0.225, 1 / 0.225, 1 / 0.225],
                )
            else:
                inv_t = transforms.Normalize(
                    mean=[-0.4345 / 0.2768, -0.4051 / 0.2713, -0.3775 / 0.2737],
                    std=[1 / 0.2768, 1 / 0.2713, 1 / 0.2737],
                )

            q_img = torch.stack([inv_t(x.detach().cpu()) for x in q_frames[:5]])
            p_img = torch.stack([inv_t(x.detach().cpu()) for x in t_frames[0, :5]])
            # else:
            # q_img = q_frames[:10].detach().cpu()
            # p_img = t_frames[0, :10].detach().cpu()

            tb_logger.log_image(q_img, "Query", iter_count)
            tb_logger.log_image(p_img, "Pos", iter_count)

            # n_imgs = frames[2,args.window+1:].detach().cpu()
            # tb_logger.log_image(n_imgs, 'Negs', iter_count)

            output_fig = plt.figure()  # create a figure object
            ax = output_fig.add_subplot(1, 1, 1)
            i = ax.imshow(output.detach().cpu().numpy(), interpolation="nearest")
            output_fig.colorbar(i)

            tb_logger.log_figure(output_fig, "Probs", iter_count)
            tb_logger.flush()

    return losses.avg
