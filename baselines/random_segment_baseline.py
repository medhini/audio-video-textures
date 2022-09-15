import math
import os
import argparse
from PIL import Image
import subprocess
import librosa

import numpy as np
import torch
import torchvision.io as io


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


def random_baseline(
    video, video_name, fps, W, S, nvl, audio_name, target_audio_filename, results_folder
):
    max_length = nvl * fps

    all_segment_ids = np.arange((len(video) - window) / stride, dtype=np.int32)
    rdm_id = np.random.choice(all_segment_ids)

    new_video = list(video[rdm_id * S : rdm_id * S + W])
    while len(new_video) < max_length:
        rdm_id = np.random.choice(all_segment_ids)
        new_video.extend(video[rdm_id * S + (W - S) : rdm_id * S + W])

    for idx, frame in enumerate(new_video):
        frame_arr = np.array(frame)

        frames_bar = np.zeros((15, video.shape[-2], 3))
        frame_n = int(idx * video.shape[-2] / len(video))
        frames_bar[:, frame_n - 3 : frame_n + 3, :] = [255, 0, 0]
        frame_arr[-25:-10, :, :] = frames_bar

        frame_fig = Image.fromarray(frame_arr)

        new_video[idx] = frame_fig

    output_video_folder = os.path.join(
        results_folder, "{}_{}".format(video_name, audio_name)
    )

    if not os.path.exists(output_video_folder):
        os.makedirs(output_video_folder)

    for count, frame_fig in enumerate(new_video):
        frame_fig.save(
            os.path.join(output_video_folder, "{:04d}.png".format(count + 1))
        )

    audio, sr = librosa.load(target_audio_filename)
    apf = math.floor(sr / fps)
    new_audio = audio[: len(new_video) * apf]

    output_audio_file = os.path.join(results_folder, "{}.wav".format(audio_name))
    librosa.output.write_wav(output_audio_file, new_audio, sr)

    outfile = os.path.join(results_folder, "{}_{}.mp4".format(video_name, audio_name))
    save_videos(fps, output_audio_file, output_video_folder, outfile)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Baseline")
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
        "--target_list",
        "-tl",
        default=None,
        type=str,
        nargs="+",
        help="list of target audios",
    )
    parser.add_argument(
        "--new_video_length", "-nvl", default=60, type=int, help="Length of new video"
    )

    parser.add_argument(
        "--results_folder",
        "-rf",
        default="random_results",
        type=str,
        help="Results folder for random baseline",
    )
    args = parser.parse_args()
    print(args)

    for itr, video_name in enumerate(args.video_list):
        assert os.path.exists(args.vdata), "No videos found at {}".format(args.vdata)

        video_path = os.path.join(args.vdata, video_name + ".mp4")
        video, _, meta = io.read_video(video_path, pts_unit="sec")
        fps = meta["video_fps"]
        print("Starting Video {}".format(video_name))
        print("Frame rate: ", fps)

        window = math.ceil(fps / 2)
        stride = math.ceil(fps / 5)

        audio_name = args.target_list[itr]

        target_audio_path = os.path.join(args.adata, args.target_list[itr] + ".wav")

        random_baseline(
            video,
            video_name,
            fps,
            window,
            stride,
            args.new_video_length,
            audio_name,
            target_audio_path,
            args.results_folder,
        )
