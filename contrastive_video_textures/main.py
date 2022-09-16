import argparse
import os
import math
import shutil
import time
from collections import OrderedDict
from PIL import Image
import subprocess
import librosa
import matplotlib
import imageio

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ipdb

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.io as io

# import torchvideo

from train import train

from validate import validate
from dataset import AudioVideoSegments

from models import (
    ContrastiveFramePrediction,
    ModelBuilder,
    ContrastivePredictionTemporal,
    ModelBuilder3D,
    AudioVisualFeatures,
    VGGish,
)
from utils import AverageMeter, Logger, overlay_cmap_image, waveform_to_examples

parser = argparse.ArgumentParser(description="PyTorch Video Textures")

# Path related arguments
parser.add_argument(
    "--enc_arch", "-ea", metavar="ARCH", default="resnet18", help="model architecture"
)
parser.add_argument(
    "--model_type",
    "-m",
    default=1,
    type=int,
    help="(1) Video Textures (2) Audio Video Textures",
)
parser.add_argument(
    "--vdata", "-vdata", default=None, type=str, help="Path to video dataset"
)
parser.add_argument("--adata", "-adata", default=None, type=str, help="Path to audio")
parser.add_argument("--pdata", "-pdata", default=None, type=str, help="Path to poses")
parser.add_argument("--fdata", "-fdata", default=None, type=str, help="Path to flow")
parser.add_argument(
    "--dadata",
    "-dadata",
    default="audio/target",
    type=str,
    help="Path to driving audio dataset",
)
parser.add_argument(
    "--video_list",
    "-vl",
    default=None,
    type=str,
    nargs="+",
    help="list of input videos",
)
parser.add_argument(
    "--fps", "-fps", default=30, type=int, help="frame rate of input video"
)
parser.add_argument(
    "--subsample_rate",
    "-subsample",
    default=1,
    type=int,
    help="rate for subsampling the video",
)
parser.add_argument(
    "--temp", "-temp", default=0.1, type=float, help="Temperature value"
)
parser.add_argument(
    "--threshold", "-th", default=0.0, type=float, help="Threshold value"
)
parser.add_argument(
    "--l2", "-l2", default=True, action="store_false", help="To use l2 norm or not"
)
parser.add_argument(
    "--interpolation",
    "-nintp",
    default=True,
    action="store_false",
    help="Interpolate frames at eval",
)
parser.add_argument(
    "--img_size",
    "-size",
    default=224,
    type=int,
    help="resize image to this size",
)
parser.add_argument(
    "--n_negs",
    "-negs",
    default=20,
    type=int,
    help="Number negative frames to use when training",
)
parser.add_argument(
    "--window", "-w", default=20, type=int, help="Size of temporal window"
)
parser.add_argument(
    "--train_stride", "-train_stride", default=4, type=int, help="Stride length"
)
parser.add_argument("--stride", "-stride", default=4, type=int, help="Stride length")
parser.add_argument(
    "--new_video_length", "-nvl", default=30, type=int, help="Length of new video"
)
parser.add_argument(
    "--alpha",
    "-alpha",
    default=0.5,
    type=float,
    help="alpha for validation to control driving audio",
)
parser.add_argument(
    "--SF",
    "-SF",
    default=5,
    type=int,
    help="slomo factor N. This will increase the frames"
    "by Nx. Example sf=2 ==> 2x frames",
)
parser.add_argument(
    "-long",
    "--long",
    dest="long",
    default=False,
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "-fb",
    "--frames_bar",
    dest="frames_bar",
    default=False,
    action="store_true",
    help="Visualize transitions.",
)

parser.add_argument(
    "--epochs", default=60, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--size", default=224, type=int, metavar="N", help="primary image input size"
)
parser.add_argument(
    "--start_epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--batch_size",
    "-bs",
    default=32,
    type=int,
    metavar="N",
    help="mini-batch size (default: 32)",
)
parser.add_argument(
    "--mini_batchsize",
    "-mbs",
    default=150,
    type=int,
    help="mini-batch size for target frames",
)
parser.add_argument(
    "--lr", "-lr", default=10e-3, type=float, metavar="LR", help="initial learning rate"
)
parser.add_argument(
    "--lr_steps",
    default=30,
    type=int,
    metavar="LRSteps",
    help="epochs to decay learning rate by 10",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight_decay",
    "--wd",
    default=0.0001,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)

parser.add_argument(
    "--workers",
    "-j",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--print_freq",
    "-p",
    default=5,
    type=int,
    metavar="N",
    help="print frequency (default: 20)",
)
parser.add_argument(
    "--log_freq",
    "-lf",
    default=10,
    type=int,
    metavar="N",
    help="frequency to write in tensorboard (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "-da",
    "--driving_audio",
    default=None,
    type=str,
    nargs="+",
    help="list of target audios",
)
parser.add_argument(
    "-daf",
    "--da_feats",
    default="VGG",
    type=str,
    help="type of feats for audio conditioning",
)
parser.add_argument(
    "-daf_resume",
    "--daf_resume",
    default="",
    type=str,
    nargs="+",
    help="List of paths to best VideoForAudio ckpt",
)

parser.add_argument(
    "-ve",
    "--visualize_evaluate",
    dest="visualize_evaluate",
    action="store_true",
    help="evaluate model on validation set and visualize logits",
)
parser.add_argument(
    "-vf",
    "--val_freq",
    default=5,
    type=int,
    metavar="VF",
    help="frequency to call validate during train)",
)

parser.add_argument(
    "--logdir", default="./logs", help="folder to output tensorboard logs"
)
parser.add_argument(
    "--logname", default="exp", help="name of the experiment for checkpoints and logs"
)
parser.add_argument(
    "-rf",
    "--results_folder",
    default="results",
    type=str,
    help="folder for result videos",
)
parser.add_argument("--ckpt", default="./ckpt", help="folder to output checkpoints")


def main(args, video_name, itr=0):
    best_loss = 1000000

    if args.visualize_evaluate:
        # create validation loader
        dataset_val = AudioVideoSegments(args, video_name, split="val")

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=3,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    elif args.evaluate == False:  # Train
        dataset_train = AudioVideoSegments(args, video_name, split="train")

        # create training loader
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        )

    # create model
    print("=> creating model '{}'".format(args.model_type))
    # Resnet 3D model
    builder = ModelBuilder3D()
    q_image_enc_model, fc_dim = builder.build_network(
        arch=args.enc_arch, img_size=args.size, window=args.window
    )
    t_image_enc_model, fc_dim = builder.build_network(
        arch=args.enc_arch, img_size=args.size, window=args.window
    )

    # VGGish Model
    audio_enc_model = VGGish()
    audio_enc_model.load_state_dict(torch.load("pytorch_vggish.pth"))

    model = ContrastivePredictionTemporal(
        q_image_enc_model,
        t_image_enc_model,
        audio_enc_model,
        args.model_type,
        fc_dim,
        args.temp,
        args.window,
        args.stride,
        args.threshold,
        mini_batchsize=args.mini_batchsize,
        enc_arch=args.enc_arch,
        img_size=args.img_size,
    )

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(
            args.resume
        )
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            args.start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )

    # create checkpoint folder
    if not os.path.exists("./ckpt"):
        os.mkdir("./ckpt")

    # update logname to include parameters
    if args.evaluate:
        logname = (
            "{}_model_{}_vd_{}_vn_{}_bs_{}_w_{}_stride_{}_temp_{}_th_{}_"
            "enca_{}_subr_{}_eval_{}".format(
                args.logname,
                args.model_type,
                os.path.split(args.vdata)[-1],
                video_name,
                args.batch_size,
                args.window,
                args.stride,
                args.temp,
                args.threshold,
                args.enc_arch,
                args.subsample_rate,
                args.evaluate or args.visualize_evaluate,
            )
        )
        if args.driving_audio is not None:
            logname = logname + "alpha_{}_daf_{}".format(args.alpha, args.da_feats)
    else:
        logname = (
            "{}_model_{}_vd_{}_vn_{}_bs_{}_negs_{}_w_{}_stride_{}_temp_{}_th_{}_"
            "enca_{}_subr_{}_eval_{}".format(
                args.logname,
                args.model_type,
                os.path.split(args.vdata)[-1],
                video_name,
                args.batch_size,
                args.n_negs,
                args.window,
                args.stride,
                args.temp,
                args.threshold,
                args.enc_arch,
                args.subsample_rate,
                args.evaluate or args.visualize_evaluate,
            )
        )

    if args.start_epoch is None:
        args.start_epoch = 0

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # start a logger
    tb_logdir = os.path.join(args.logdir, logname)
    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    if args.evaluate:
        validate(
            model,
            args,
            video_name=video_name,
            tb_logger=tb_logger,
            model_type=args.model_type,
            itr=itr,
        )
        return

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_steps)

    print("Training for {} epochs.".format(args.epochs - args.start_epoch))

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        loss = train(
            train_loader,
            model,
            optimizer,
            args,
            epoch,
            tb_logger,
        )

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.enc_arch,
                "state_dict": model.module.state_dict(),
                "best_loss": best_loss,
            },
            is_best,
            os.path.join(args.ckpt, logname),
        )
        scheduler.step()
        if loss < 0.07:
            print("Loss {}. Stopping at epoch {}.".format(loss, epoch))
            break


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + "_latest.pth.tar")
    if is_best:
        shutil.copyfile(filename + "_latest.pth.tar", filename + "_best.pth.tar")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    assert os.path.exists(args.vdata), "No videos found at {}".format(args.vdata)

    if args.adata != None and os.path.exists(args.adata):
        print("Audio found at {}".format(args.adata))

    if args.video_list is None:
        args.video_list = sorted(
            [
                f.split(".")[0]
                for f in sorted(os.listdir(args.vdata))
                if not f.startswith(".")
            ]
        )

    video = None
    for itr, video_name in enumerate(args.video_list):
        args.results_folder = "results_{}".format(video_name)

        if args.evaluate or args.visualize_evaluate:
            video_filename = os.path.join(args.vdata, "{}.mp4".format(video_name))

            reader = imageio.get_reader(video_filename)
            args.fps = reader.get_meta_data()["fps"]
            print("Frame rate: ", args.fps)

            args.window = math.ceil(args.fps / 2)
            args.stride = math.ceil(args.fps / 5)

            print("Stride {} Window {}".format(args.stride, args.window))

            if args.resume == "":
                args.resume = (
                    "ckpt/exp_model_{}_vd_{}_vn_{}_bs_{}_negs_{}_w_{}_"
                    "stride_{}_temp_0.1_th_0.0_enca_{}_subr_{}_eval_False_best.pth.tar".format(
                        args.model_type,
                        os.path.split(args.vdata)[-1],
                        video_name,
                        args.batch_size,
                        args.n_negs,
                        args.window,
                        args.stride,
                        args.enc_arch,
                        args.subsample_rate,
                    )
                )

            assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(
                args.resume
            )
            print("=> loading checkpoint '{}'".format(args.resume))

            if args.driving_audio != None:
                args.results_folder += "_target_{}_{}".format(
                    video_name,
                    os.path.split(args.driving_audio[itr])[-1].split(".")[0],
                )

        print("Starting video {}".format(video_name))
        main(args, video_name, itr)
