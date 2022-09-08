import subprocess
import numpy as np
import math
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def save_videos(
    output_video_folder,
    outfile,
    fps,
    interpolation=False,
    audio_w=None,
    SF=0,
    output_audio_filename="",
    output_audio_filename_intp="",
    output_video_folder_intp="",
    outfile_intp=None,
):
    if audio_w is None:
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
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
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
    else:
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

        # remove audio file
        subprocess.call(["rm", output_audio_filename])

    if interpolation:
        if audio_w is None:
            try:
                subprocess.call(
                    [
                        "ffmpeg",
                        "-r",
                        str(((SF + 1) / 2) * fps),
                        "-start_number",
                        "1",
                        "-i",
                        output_video_folder_intp + "/%04d.png",
                        "-c:v",
                        "libx264",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-y",
                        outfile_intp,
                    ]
                )
                print("Written ", outfile_intp)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "command '{}' return with error (code {}): {}".format(
                        e.cmd, e.returncode, e.output
                    )
                )
        else:
            try:
                subprocess.call(
                    [
                        "ffmpeg",
                        "-r",
                        str(((SF + 1) / 2) * fps),
                        "-start_number",
                        "1",
                        "-i",
                        output_video_folder_intp + "/%04d.png",
                        "-i",
                        output_audio_filename_intp,
                        "-c:v",
                        "libx264",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-y",
                        outfile_intp,
                    ]
                )
                print("Written ", outfile_intp)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "command '{}' return with error (code {}): {}".format(
                        e.cmd, e.returncode, e.output
                    )
                )

            # remove audio file
            subprocess.call(["rm", output_audio_filename_intp])

        # remove video file
        subprocess.call(["rm", "-r", output_video_folder_intp])

    # remove video file
    subprocess.call(["rm", "-r", output_video_folder])
    return


def combine_batches(tensor, num_valid):
    """Combines [num_gpus, N/num_gpus, ...] into [1, num_valid, ...].
    Args:
        tensor (torch.Tensor): Input tensor [num_gpus, N/num_gpus, ...].
        num_valid (int): Number of valid rows.
    Returns:
        output (torch.Tensor): Output tensor [1, num_valid, ...]
    """
    num_gpus, num_inputs_per_gpu = list(tensor.size())[0:2]
    assert num_valid <= num_gpus * num_inputs_per_gpu
    size = [1, num_gpus * num_inputs_per_gpu] + list(tensor.size())[2:]
    output = tensor.view(*size)
    output = output[0:1, :num_valid]
    return output


def split_into_batches(tensor, max_segments_per_gpu):
    """Splits tensor of size [1, N, ...] into [N/max_segments, max_segments, ...].
    Output will be 0-padded for even sizing.
    Args:
        tensor (torch.Tensor): Input tensor [1, N, ...].
        num_gpus (int): Number of gpus.
    Returns:
        output (torch.Tensor): Output tensor [N/max_segments, max_segments, ...].
        num_valid (int): Number of valid rows.
    """
    assert tensor.size(0) == 1
    num_inputs = tensor.size(1)
    num_batches = math.ceil(num_inputs / max_segments_per_gpu)
    size = [num_batches, max_segments_per_gpu] + list(tensor.size())[2:]
    batched = torch.zeros(*size, dtype=tensor.dtype)

    for idx in range(num_batches):
        start = idx * max_segments_per_gpu
        end = min(start + max_segments_per_gpu, num_inputs)
        num_in_batch = end - start
        batched[idx, :num_in_batch] = tensor[0, start:end]

    return batched, num_inputs


def split_into_overlapping_segments(tensor, max_segments_per_gpu, W, S):
    """Splits tensor of size [N, ...] into [batch_size, chunk_size,  H, W].
    Ouput will be 0-padded for even sizing.  
    Args:
        tensor (torch.Tensor): Input tensor [N, ...].
        num_gpus (int): Number of gpus.
        W (int): Length of the segments.
        S (int): Stride. 
    Returns:
        output (torch.Tensor): Output tensor [batch_size, chunk_size, ...].
        num_valid (int): Number of valid rows.
    """
    num_inputs = tensor.size(0)
    total_segments = math.ceil((num_inputs - W) / S)
    chunk_size = max_segments_per_gpu * S + W
    batch_size = math.ceil(total_segments / max_segments_per_gpu)

    size = [batch_size, chunk_size] + list(tensor.size()[1:])

    batched = torch.zeros(*size, dtype=tensor.dtype)

    for idx in range(batch_size):
        start = idx * S * (max_segments_per_gpu - 1)
        end = min(start + chunk_size, num_inputs)
        num_in_batch = end - start
        batched[idx, :num_in_batch] = tensor[start:end]

    return batched, num_inputs
