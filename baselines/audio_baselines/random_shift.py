import math
import os
import argparse
from PIL import Image
import librosa
import subprocess

import numpy as np
import torch
import torchvision.io as io


def shift_audio(
    video_path, audio, audio_name, sr, results_folder,
):
    audio = audio[: sr * 60]
    new_audio = np.zeros_like(audio)

    shift = np.random.randint(4, 12)
    print("Seconds to shift by: ", shift)
    new_audio[: -shift * sr] = audio[shift * sr :]
    new_audio[-shift * sr :] = audio[: shift * sr]

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    output_audio_filename = os.path.join(results_folder, "{}.wav".format(audio_name))
    librosa.output.write_wav(output_audio_filename, new_audio, sr)

    outfile = os.path.join(results_folder, "{}_{}.mp4".format(video_name, audio_name))

    subprocess.call(
        [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            output_audio_filename,
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            outfile,
        ]
    )

    # remove audio file
    subprocess.call(["rm", output_audio_filename])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Shift Baseline")
    parser.add_argument(
        "--vdata",
        "-vdata",
        default="../contrastive_video_textures/audio_conditioning_results/Contrastive",
        type=str,
        help="Path to video dataset",
    )
    parser.add_argument(
        "--adata",
        "-adata",
        default="../contrastive_video_textures/audio/target",
        type=str,
        help="Path to audio dataset",
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
        "--new_video_length", "-nvl", default=30, type=int, help="Length of new video"
    )

    parser.add_argument(
        "--results_folder",
        "-rf",
        default="shift_results",
        type=str,
        help="Results folder for random baseline",
    )
    args = parser.parse_args()
    print(args)

    if args.video_list is None:
        args.video_list = sorted(
            [f for f in sorted(os.listdir(args.vdata)) if not f.startswith(".")]
        )

    for itr, filename in enumerate(args.video_list):
        video_name = filename.split("_")[0]
        audio_name = filename.split("_")[1].split(".")[0]

        assert os.path.exists(args.vdata), "No videos found at {}".format(args.vdata)

        video_path = os.path.join(args.vdata, filename)

        target_audio_path = os.path.join(args.adata, audio_name + ".wav")
        audio, sr = librosa.load(target_audio_path)

        shift_audio(
            video_path, audio, audio_name, sr, args.results_folder,
        )
