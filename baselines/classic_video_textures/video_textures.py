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
from interpolate import interpolate, modify_frames

import torch
import torchvision.io as io
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import Logger, read_data, save_video
from computeD1 import compute_D1
from computeD2 import compute_D2
from q_learning import q_learning


def audio_video_texture(
    args,
    P: np.array,
    frames: list,
    output_video_folder: str,
    output_video_folder_intp: str,
    audio: np.array = None,
    output_audio_file: str = "",
    intp_model=None,
):

    start = 100
    # start = np.random.randint(P.shape[0]-1)
    new_video_length = args.fps * args.new_video_length
    jump_count = 0

    if args.model_type == 1:
        this_frame = start
        new_frames_list = [start]

        new_frames_intp = []
        frame_arr = np.array(frames[start])

        frames_bar = np.zeros((15, frames.shape[-2], 3))
        frame_n = 0
        frames_bar[:, frame_n - 3 : frame_n + 3, :] = [255, 0, 0]
        frame_arr[-25:-10, :, :] = frames_bar

        frame_fig = Image.fromarray(frame_arr)
        new_frames_intp.append(frame_fig)

        for i in range(int((args.SF - 1) / 2)):
            new_frames_intp.append(frame_fig)

        if audio is not None:
            audio_per_frame = int(len(audio) / len(frames))
            new_audio = []
            new_audio.extend(
                audio[start * audio_per_frame : (start + 1) * audio_per_frame]
            )

        while len(new_frames_list) < new_video_length:
            # np.random.choice(np.arange(len(output[0])), p=output[0].detach().cpu().numpy())
            # next_frame = np.random.choice(np.arange(len(P[this_frame])), p=P[this_frame].detach().cpu().numpy())
            next_frame = np.random.choice(
                P[this_frame].nonzero().view(-1).detach().cpu().numpy()
            )
            intp_added = False
            if next_frame != this_frame + 1:
                jump_count += 1

                # remove previously added frames
                new_frames_intp = new_frames_intp[: -int((args.SF - 1) / 2)]

                intp_added = True

                # prev diff_id; last frame previously added
                frame0 = Image.fromarray(np.array(frames[this_frame]))
                frame1 = Image.fromarray(np.array(frames[next_frame]))

                frame0, frame1, inv_trans = modify_frames(frame0, frame1)
                int_frames = intp_model(frame0, frame1, inv_trans)

                print("Added {} intermediate frames.\n".format(len(int_frames)))

                for int_frame in int_frames:
                    frame_arr = np.array(int_frame)
                    frames_bar = np.zeros((15, frames.shape[-2], 3))
                    frame_arr[-25:-10, :, :] = frames_bar
                    new_frames_intp.append(Image.fromarray(frame_arr))

            new_frames_list.append(next_frame)

            if intp_added == False:
                frame_arr = np.array(frames[next_frame])

                frames_bar = np.zeros((15, frames.shape[-2], 3))
                frame_n = int(this_frame * frames.shape[-2] / len(frames))
                frames_bar[:, frame_n - 3 : frame_n + 3, :] = [255, 0, 0]
                frame_arr[-25:-10, :, :] = frames_bar

                frame_fig = Image.fromarray(frame_arr)
                new_frames_intp.append(frame_fig)

                if intp_added == False or count != 0:
                    for i in range(int((args.SF - 1) / 2)):
                        new_frames_intp.append(frame_fig)

            this_frame = copy.deepcopy(next_frame)

            if audio is not None:
                new_audio.extend(
                    audio[
                        this_frame
                        * audio_per_frame : (this_frame + 1)
                        * audio_per_frame
                    ]
                )

    elif args.model_type == 2:
        this_frame = start
        new_frames_list = list(np.arange(this_frame, this_frame + args.stride))

        if audio is not None:
            audio_per_frame = int(len(audio) / len(frames))
            new_audio = []
            new_audio.extend(
                audio[
                    this_frame
                    * audio_per_frame : (this_frame + args.stride)
                    * audio_per_frame
                ]
            )

        this_frame += args.stride

        while len(new_frames_list) < new_video_length:
            next_frame = np.random.choice(
                P[this_frame].nonzero().view(-1).detach().cpu().numpy()
            )
            if next_frame != this_frame + 1:
                jump_count += 1

            next_frames = list(
                np.arange(next_frame, min((next_frame + args.stride), P.shape[0]))
            )
            new_frames_list.extend(next_frames)

            if audio is not None:
                new_audio.extend(
                    audio[
                        (next_frame)
                        * audio_per_frame : min((next_frame + args.stride), P.shape[0])
                        * audio_per_frame
                    ]
                )

            this_frame = min((next_frame + args.stride), P.shape[0] - 1)
    else:
        this_frame = start
        new_frames_list = list(np.arange(this_frame, this_frame + args.filter_size))

        if audio is not None:
            audio_per_frame = int(len(audio) / len(frames))
            new_audio = []
            new_audio.extend(
                audio[
                    this_frame
                    * audio_per_frame : (this_frame + args.filter_size)
                    * audio_per_frame
                ]
            )

        while len(new_frames_list) < new_video_length:
            next_frame = np.random.choice(
                P[this_frame].nonzero().view(-1).detach().cpu().numpy()
            )
            if next_frame != this_frame + 1:
                jump_count += 1
            next_frames = list(
                np.arange(
                    this_frame * args.stride + (args.filter_size - args.stride),
                    this_frame * args.stride + args.filter_size,
                )
            )
            new_frames_list.extend(next_frames)

            if audio is not None:
                new_audio.extend(
                    audio[
                        (this_frame * args.stride + (args.filter_size - args.stride))
                        * audio_per_frame : (
                            this_frame * args.stride + args.filter_size
                        )
                        * audio_per_frame
                    ]
                )
        this_frame = next_frame

    print("Frames list: ", new_frames_list)
    if not os.path.exists(output_video_folder):
        os.mkdir(output_video_folder)

    for count, frame_idx in enumerate(new_frames_list):
        frames_bar = np.zeros((15, frames.shape[-2], 3))
        frame_n = int(frame_idx * frames.shape[-2] // len(frames))
        frames_bar[:, frame_n - 4 : frame_n + 4, :] = [255, 0, 0]

        frame_arr = np.array(frames[frame_idx])
        frame_arr[-25:-10, :, :] = frames_bar
        frame_fig = Image.fromarray(frame_arr)

        frame_fig.save(
            os.path.join(output_video_folder, "{:04d}.png".format(count + 1))
        )

    if not os.path.exists(output_video_folder_intp):
        os.mkdir(output_video_folder_intp)

    for count, frame_fig in enumerate(new_frames_intp):
        frame_fig.save(
            os.path.join(output_video_folder_intp, "{:04d}.png".format(count + 1))
        )

    if audio is not None:
        new_audio = np.array(new_audio)
        librosa.output.write_wav(output_audio_file, new_audio, args.sr)

    print("Written_{}".format(output_video_folder))
    return jump_count


def main(args, video_name: str):
    input_frames, video, args.fps, audio, args.sr, _ = read_data(args, video_name)
    # 1.1, 1.5, 2.0, 2.5
    # sigmas = torch.tensor([0.3, 0.6, 1.1, 2.0], dtype=torch.float32)
    # sigmas = torch.tensor([1.2, 1.5, 2.0, 2.1, 2.5, 2.6, 2.8], dtype=torch.float32)
    # sigmas = torch.tensor([2.7, 2.9, 3.0, 3.5, 4.0], dtype=torch.float32)
    sigmas = torch.tensor([4.45, 4.5, 4.52, 4.55, 4.58], dtype=torch.float32)

    jump_counts = []
    new_sigmas = []

    # Initialize interpolation model.
    print("Initializing interpolation model. ")
    if args.interpolation:
        intp_model = interpolate([video.shape[2], video.shape[1]], args.SF).cuda()
        dict1 = torch.load(
            "../contrastive_video_textures/ckpt/SuperSloMo.ckpt", map_location="cpu"
        )
        intp_model.ArbTimeFlowIntrp.load_state_dict(dict1["state_dictAT"])
        intp_model.flowComp.load_state_dict(dict1["state_dictFC"])

    for value in sigmas:
        D1, P1, sigma = compute_D1(
            input_frames,
            value,
            args.feats,
            audio=audio,
            sr=args.sr,
            fps=args.fps,
            slow=args.slow,
            batch_size=args.batch_size,
        )
        if args.model_type == 1 or args.model_type == 2:
            D2, P2, sigma, binomial_filter = compute_D2(
                D1, value, filter_size=args.filter_size
            )
        else:
            D2, P2, sigma, binomial_filter = compute_D2(
                D1, value, filter_size=args.filter_size, stride=args.stride
            )
        D3, P3, P3_new, sigma = q_learning(D2, value, thresholding=args.threshold)
        new_sigmas.append(sigma)

        logname = "{}_{}_feats_{}_vd_{}_vn_{}_w_{}_stride_{}_sigma_{}_th_{}".format(
            args.logname,
            args.model_type,
            args.feats,
            os.path.split(args.vdata)[-1],
            video_name,
            args.filter_size,
            args.stride,
            sigma.item(),
            args.threshold,
        )

        tb_logdir = os.path.join(args.logdir, logname)
        if not (os.path.exists(tb_logdir)):
            os.makedirs(tb_logdir)
        tb_logger = Logger(tb_logdir)

        bin_fig = plt.figure()  # create a figure object
        ax = bin_fig.add_subplot(1, 1, 1)
        i = ax.imshow(
            binomial_filter.view(args.filter_size, args.filter_size)
            .detach()
            .cpu()
            .numpy(),
            interpolation="nearest",
        )
        bin_fig.colorbar(i)

        tb_logger.log_figure(bin_fig, "Binomial", 1)
        tb_logger.flush()

        D1_fig = plt.figure()  # create a figure object
        ax = D1_fig.add_subplot(1, 1, 1)
        i = ax.imshow(D1.detach().cpu().numpy(), interpolation="nearest")
        D1_fig.colorbar(i)

        tb_logger.log_figure(D1_fig, "D1", 1)
        tb_logger.flush()

        P1_fig = plt.figure()  # create a figure object
        ax = P1_fig.add_subplot(1, 1, 1)
        i = ax.imshow(P1.detach().cpu().numpy(), interpolation="nearest")
        P1_fig.colorbar(i)

        tb_logger.log_figure(P1_fig, "P1", 1)
        tb_logger.flush()

        D2_fig = plt.figure()  # create a figure object
        ax = D2_fig.add_subplot(1, 1, 1)
        i = ax.imshow(D2.detach().cpu().numpy(), interpolation="nearest")
        P1_fig.colorbar(i)

        tb_logger.log_figure(D2_fig, "D2", 1)
        tb_logger.flush()

        P2_fig = plt.figure()  # create a figure object
        ax = P2_fig.add_subplot(1, 1, 1)
        i = ax.imshow(P2.detach().cpu().numpy(), interpolation="nearest")
        P2_fig.colorbar(i)

        tb_logger.log_figure(P2_fig, "P2", 1)
        tb_logger.flush()

        D3_fig = plt.figure()  # create a figure object
        ax = D3_fig.add_subplot(1, 1, 1)
        i = ax.imshow(D3.detach().cpu().numpy(), interpolation="nearest")
        D3_fig.colorbar(i)

        tb_logger.log_figure(D3_fig, "D3", 1)
        tb_logger.flush()

        P3_fig = plt.figure()  # create a figure object
        ax = P3_fig.add_subplot(1, 1, 1)
        i = ax.imshow(P3.detach().cpu().numpy(), interpolation="nearest")
        P3_fig.colorbar(i)

        tb_logger.log_figure(P3_fig, "P3", 1)
        tb_logger.flush()

        P3_fig = plt.figure()  # create a figure object
        ax = P3_fig.add_subplot(1, 1, 1)
        i = ax.imshow(P3_new.detach().cpu().numpy(), interpolation="nearest")
        P3_fig.colorbar(i)

        tb_logger.log_figure(P3_fig, "P3 New", 1)
        tb_logger.flush()

        results_folder = os.path.join(
            args.results_folder,
            "{}_{}_feats_{}_w_{}_stride_{}".format(
                args.logname, args.model_type, args.feats, args.filter_size, args.stride
            ),
        )

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        new_video_id = len(os.listdir(results_folder)) + 1
        output_video_folder = os.path.join(
            results_folder,
            "video_{}_{}_sig_{}_th_{}".format(
                video_name, new_video_id, sigma.item(), args.threshold
            ),
        )
        output_video_folder_intp = os.path.join(
            results_folder,
            "video_{}_{}_sig_{}_th_{}_intp".format(
                video_name, new_video_id, sigma.item(), args.threshold
            ),
        )

        output_audio_filename = None
        if audio is not None:
            output_audio_filename = os.path.join(
                results_folder, "audio_{}_{}.wav".format(video_name, new_video_id)
            )

        outfile = output_video_folder + ".mp4"
        outfile_intp = output_video_folder_intp + ".mp4"

        jump_count = audio_video_texture(
            args,
            P3_new,
            video,
            output_video_folder,
            output_video_folder_intp,
            audio,
            output_audio_filename,
            intp_model,
        )
        jump_counts.append(jump_count)

        # save_video(args, output_video_folder, output_video_folder_intp, output_audio_filename, outfile)
        save_video(
            output_video_folder,
            outfile,
            args.fps,
            args.interpolation,
            audio,
            args.SF,
            output_audio_filename,
            output_audio_filename,
            output_video_folder_intp,
            outfile_intp,
        )

    tb_jc_logdir = os.path.join("jump_counts")
    if not (os.path.exists(tb_jc_logdir)):
        os.makedirs(tb_jc_logdir)

    sigmas = [str(i) for i in list(np.array(new_sigmas))]
    line = plt.bar(sigmas, jump_counts)
    plt.xlabel("Sigma")
    plt.ylabel("Jump Count")

    for i in range(len(jump_counts)):
        plt.annotate(str(jump_counts[i]), xy=(sigmas[i], jump_counts[i]))

    plt.savefig(
        os.path.join(
            tb_jc_logdir,
            "jump_counts_{}_{}_{}.png".format(
                os.path.split(args.vdata)[-1], video_name, args.model_type
            ),
        )
    )

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="=Video Textures")
    parser.add_argument(
        "--model_type",
        "-m",
        default=1,
        type=int,
        help="(1) Classic (2) Classic + (3) Classic ++",
    )
    parser.add_argument(
        "--vdata", "-vdata", default=None, type=str, help="Path to video dataset"
    )
    parser.add_argument(
        "--adata", "-adata", default=None, type=str, help="Path to audio dataset"
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
        "--feats", "-f", default="RGB", type=str, help="Features to use"
    )
    parser.add_argument(
        "--slow",
        "-s",
        dest="slow",
        action="store_true",
        help="set false for large videos",
    )
    parser.add_argument(
        "--fps", "-fps", default=30, type=int, help="frame rate of input video"
    )
    parser.add_argument(
        "--sr", "-sr", default=22050, type=int, help="rate of input audio"
    )
    parser.add_argument(
        "--filter_size",
        "-fs",
        default=40,
        type=int,
        help="filter size of gaussian filter",
    )
    parser.add_argument(
        "--batch_size", "-bs", default=64, type=int, help="mini batch size"
    )
    parser.add_argument("--stride", "-stride", default=4, type=int, help="stride")
    parser.add_argument(
        "--new_video_length",
        "-nvl",
        default=30,
        type=int,
        help="frame rate of input video",
    )
    parser.add_argument(
        "--interpolation",
        "-nintp",
        default=True,
        action="store_false",
        help="Interpolate frames at eval",
    )
    parser.add_argument(
        "--SF",
        "-SF",
        default=3,
        type=int,
        help="slomo factor N. This will increase the frames"
        "by Nx. Example sf=2 ==> 2x frames",
    )
    parser.add_argument(
        "--sigma", "-sigma", default=0.5, type=float, help="Sigma value"
    )
    parser.add_argument(
        "--threshold", "-t", default=0.08, type=float, help="Threshold value for P"
    )
    parser.add_argument(
        "-rf",
        "--results_folder",
        default="results_classic",
        type=str,
        help="folder for result videos",
    )
    parser.add_argument(
        "--logdir", default="./logs", help="folder to output tensorboard logs"
    )
    parser.add_argument(
        "--logname",
        default="exp_classic",
        help="name of the experiment for checkpoints and logs",
    )

    args = parser.parse_args()
    print(args)

    if args.video_list is None:
        args.video_list = [
            f.split(".")[0]
            for f in sorted(os.listdir(args.vdata))
            if not f.startswith(".")
        ]

    for itr, video_name in enumerate(args.video_list):
        args.results_folder = "results_{}".format(os.path.split(args.vdata)[-1])

        print("Starting video {}".format(video_name))
        main(args, video_name)

