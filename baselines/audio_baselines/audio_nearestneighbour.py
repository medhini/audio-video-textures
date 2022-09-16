import os
import argparse
import math
from PIL import Image
import librosa
import subprocess
import copy

import numpy as np
import torch
import torchvision.io as io
import torch.nn as nn
import torch.nn.functional as F

from utils import *


def save_videos(
    fps, output_audio_filename, output_video_folder, outfile,
):
    try:
        subprocess.call(
            [
                "ffmpeg",
                "-r",
                str(fps),
                "-start_number",
                "1",
                "-i",
                output_video_folder + "/%04d.png",
                "-i",
                output_audio_filename,
                "-c:v",
                "libx264",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-y",
                outfile,
            ]
        )
        print("Written ", outfile)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    # remove video file
    subprocess.call(["rm", "-r", output_video_folder])

    # remove audio file
    subprocess.call(["rm", output_audio_filename])
    return


def audio_nn(args, video_name, driving_audio_name, driving_audio_path):
    # Prepare the video.
    video_path = os.path.join(args.vdata, video_name + ".mp4")
    video, _, meta = io.read_video(video_path, pts_unit="sec")
    fps = meta["video_fps"]
    print("Frame rate: ", fps)

    W = math.ceil(fps / 2)
    S = math.ceil(fps / 5)

    # Prepare source audio.
    audio_folder = os.path.join(args.adata, "{}.wav".format(video_name))
    audio_w = None
    assert os.path.exists(audio_folder), "No audio found at {}".format(audio_folder)
    audio_w, sr = librosa.load(audio_folder)
    apf = math.floor(sr / fps)
    print("Audio per frame:", apf)

    # Adjust audio length to match video length.
    audio_w = audio_w[: len(video) * apf]
    audio_eg = waveform_to_examples(audio_w, sr)
    audio_eg = torch.from_numpy(audio_eg).unsqueeze(dim=1)
    audio_eg = audio_eg.float()
    audio_w = torch.tensor(audio_w)

    # Prepare driving audio.
    print("Driving audio path:", driving_audio_path)
    assert os.path.exists(driving_audio_path), "No audio found at {}".format(
        driving_audio_path
    )
    driving_audio_w, sr_da = librosa.load(driving_audio_path)
    driving_audio_eg = waveform_to_examples(driving_audio_w, sr_da)
    driving_audio_eg = torch.from_numpy(driving_audio_eg).unsqueeze(dim=1)
    driving_audio_eg = driving_audio_eg.float()

    # Frame IDs for all, query, +ve, -ve and target(+ve + -ve) frames
    all_segment_ids = np.arange(math.floor((len(video) - W) / S))

    max_length = fps * args.new_video_length
    new_video = []
    count = 0

    while len(new_video) < max_length:
        q_id = 0
        max_sim = 0
        driving_eg = driving_audio_eg[count]
        count += 1

        for choice in all_segment_ids:
            source_eg = audio_eg[choice]

            source_eg = F.normalize(source_eg.view(-1), dim=0)
            driving_eg = F.normalize(driving_eg.view(-1), dim=0)

            # Compute similarity.
            cos = nn.CosineSimilarity(dim=0)
            sim = cos(source_eg, driving_eg)

            # pick choice that's most similar to target audio
            if sim > max_sim:
                q_id = copy.deepcopy(choice)
                max_sim = max(sim, max_sim)

        print("Max Audio Sim: ", max_sim)
        if len(new_video) == 0:
            new_video = list(video[q_id * S : q_id * S + W])
        else:
            new_video.extend(video[q_id * S + (W - S) : q_id * S + W])

    output_video_folder = os.path.join(
        args.results_folder, "{}_{}".format(video_name, driving_audio_name)
    )

    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    for count, frame in enumerate(new_video):
        frame_fig = Image.fromarray(np.asarray(frame))
        frame_fig.save(
            os.path.join(output_video_folder, "{:04d}.png".format(count + 1))
        )

    new_audio = driving_audio_w[: len(new_video) * apf]

    output_audio_file = os.path.join(
        args.results_folder, "{}.wav".format(driving_audio_name)
    )
    librosa.output.write_wav(output_audio_file, new_audio, sr_da)

    outfile = os.path.join(
        args.results_folder, "{}_{}.mp4".format(video_name, driving_audio_name)
    )
    save_videos(fps, output_audio_file, output_video_folder, outfile)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio NN Baseline")
    parser.add_argument(
        "--vdata",
        "-vdata",
        default="../contrastive_video_textures/source",
        type=str,
        help="Path to video dataset",
    )
    parser.add_argument(
        "--adata",
        "-adata",
        default="../contrastive_video_textures/audio/source",
        type=str,
        help="Path to audio dataset",
    )
    parser.add_argument(
        "--dadata",
        "-dadata",
        default="../contrastive_video_textures/audio/target",
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
        "--driving_list",
        "-dl",
        default=None,
        type=str,
        nargs="+",
        help="list of driving audios",
    )
    parser.add_argument(
        "--new_video_length", "-nvl", default=60, type=int, help="Length of new video"
    )

    parser.add_argument(
        "--results_folder",
        "-rf",
        default="audioNN_results",
        type=str,
        help="Results folder for audio NN baseline",
    )
    args = parser.parse_args()
    print(args)

    if args.video_list is None:
        args.video_list = sorted(
            [
                f.split(".")[0]
                for f in sorted(os.listdir(args.vdata))
                if not f.startswith(".")
            ]
        )

    for itr, video_name in enumerate(args.video_list):
        assert os.path.exists(args.vdata), "No videos found at {}".format(args.vdata)
        driving_audio_path = os.path.join(args.dadata, args.driving_list[itr] + ".wav")
        audio_nn(args, video_name, args.driving_list[itr], driving_audio_path)

    pass
