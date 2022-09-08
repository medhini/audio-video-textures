import argparse
import os
import librosa
import math
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torchvision.io as io
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import imageio
import copy

from utils import waveform_to_examples
from . import scale_jitter_crop_norm

from types import SimpleNamespace

from slowfast.utils.parser import load_config
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.utils import process_cv2_inputs

class AudioVideoSegments(Dataset):
    def __init__(self, args, video_name, split="train"):
        self.vdata = args.vdata
        self.adata = args.adata
        self.video_name = video_name
        self.split = split
        self.n_negs = args.n_negs
        self.crop_size = args.img_size
        self.enc_arch = args.enc_arch
        self.img_size = args.img_size

        self.video_filename = os.path.join(self.vdata, "{}.mp4".format(self.video_name))

        assert os.path.exists(self.video_filename), "No video found at '{}'".format(
            self.video_filename
        )

        self.video, _, self.meta = io.read_video(self.video_filename, pts_unit="sec")

        if self.enc_arch != "slowfast":
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((args.img_size, args.img_size)),
                    # transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737],
                    ),
                ]
            )

            self.video = torch.stack(
                [transform(i.permute(2, 0, 1)) for i in self.video]
            )
        else:
            sf_args = SimpleNamespace()
            sf_args.cfg_file = "/home/medhini/audio_video_gan/contrastive_video_textures/slowfast_configs/SLOWFAST_8X8_R50.yaml"
            sf_args.opts = None
            self.cfg = load_config(sf_args)
            self.cfg.NUM_GPUS = 1
            self.cfg.TEST.CHECKPOINT_TYPE = "caffe2"
            self.cfg.TEST.CHECKPOINT_FILE_PATH = "/home/medhini/audio_video_gan/contrastive_video_textures/pretrained/SLOWFAST_8x8_R50.pkl"

            # Scale values to 0-1.
            self.video = self.video.float() / 255

            # Convert RGB -> BGR.
            permute = [2, 1, 0]
            self.video = self.video[:, :, :, permute]

        print("Frame shape: ", self.video[0].shape)
        self.fps = self.meta["video_fps"]

        # Window of 0.5 seconds. Stride of 0.2 seconds.
        args.window = math.ceil(self.fps / 2)
        args.stride = math.ceil(self.fps / 5)

        print("Stride {} Window {}".format(args.stride, args.window))

        self.stride = args.stride
        self.window = args.window

        if self.adata is None:
            # Make dummy audio data. #TODO: Fix this later. 
            self.audio_w = torch.rand(len(self.video) * 10)
            self.apf = 10
            self.audio_eg = torch.rand(
                (math.floor((len(self.video) - self.window) / self.stride)), 10
            )
        else:
            audio_folder = os.path.join(self.adata, "{}.wav".format(self.video_name))
            assert os.path.exists(audio_folder), "No audio found at {}".format(
                audio_folder
            )

            print("Audio folder: ", audio_folder)
            self.audio_w, self.sr = librosa.load(audio_folder)
            self.apf = math.floor(self.sr / self.fps)  # audio-per-frame
            self.audio_w = self.audio_w[: len(self.video) * self.apf]

            print(
                "Audio waveform length: {}, SR: {}".format(len(self.audio_w), self.sr)
            )

            self.audio_eg = waveform_to_examples(self.audio_w, self.sr)
            self.audio_eg = torch.from_numpy(self.audio_eg).unsqueeze(dim=1)
            self.audio_eg = self.audio_eg.float()
            self.audio_w = torch.tensor(self.audio_w)

    def __len__(self):
        # last frame is never the query unless its validation
        if self.split == "train":
            return math.floor((len(self.video) - self.window) / self.stride) - 1
        else:
            return math.floor((len(self.video) - self.window) / self.stride)

    def __getitem__(self, idx):
        S = self.stride
        W = self.window

        # Get the query, positive, and negative IDs.
        query_id = idx * S

        if self.split == "train":
            pos_id = (idx + 1) * S

            neg_ids = np.arange(self.__len__() + 1)  # +1 to include last frame
            mask = np.ones(self.__len__() + 1, dtype=bool)
            mask[[idx, idx + 1]] = False
        else:
            pos_id = ((idx + 1) % (self.__len__())) * S

            neg_ids = np.arange(self.__len__())
            mask = np.ones(self.__len__(), dtype=bool)
            mask[[idx, (idx + 1) % (self.__len__())]] = False

        # Get query frames.
        if self.enc_arch != "slowfast":
            q_v_segs = self.video[query_id : query_id + W]
        else:
            q_v_segs = process_cv2_inputs(self.video[query_id : query_id + W], self.cfg)

            q_v_segs = [
                F.interpolate(
                    item.squeeze(0),
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                )
                for item in q_v_segs
            ]

        q_aw_segs = self.audio_w[query_id * self.apf : (query_id + W) * self.apf]

        # Choose audio corresponding to frame in center of window.
        q_ae_segs = self.audio_eg[idx]

        # Get pos frames.
        if self.enc_arch != "slowfast":
            t_v_segs = [self.video[pos_id : pos_id + W]]
        else:
            pos_segs = process_cv2_inputs(self.video[pos_id : pos_id + W], self.cfg)

            t_v_segs = [
                [
                    F.interpolate(
                        item.squeeze(0),
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                    )
                    for item in pos_segs
                ]
            ]

        t_aw_segs = [self.audio_w[pos_id * self.apf : (pos_id + W) * self.apf]]
        t_ae_segs = [self.audio_eg[idx + 1]]

        neg_idxs = neg_ids[mask, ...]

        if self.split == "train":
            neg_idxs = np.random.choice(neg_idxs, self.n_negs, replace=False)
            hard_negs = np.array(
                [idx - 4, idx - 3, idx - 2, idx - 1, idx + 2, idx + 3, idx + 4, idx + 5]
            )
            hard_negs = hard_negs[hard_negs >= 0]
            hard_negs = hard_negs[hard_negs <= self.__len__()]
            neg_idxs[: len(hard_negs)] = hard_negs

        neg_v_segments = []
        neg_aw_segments = []
        neg_ae_segments = self.audio_eg[neg_idxs]

        for i in neg_idxs:
            # Get negative frames.
            if self.enc_arch == "slowfast":
                neg_segs = process_cv2_inputs(self.video[i * S : i * S + W], self.cfg)
                t_v_segs.append(
                    [
                        F.interpolate(
                            item.squeeze(0),
                            size=(self.img_size, self.img_size),
                            mode="bilinear",
                        )
                        for item in neg_segs
                    ]
                )

            else:
                neg_v_segments.append(self.video[i * S : i * S + W])

            neg_aw_segments.append(
                self.audio_w[i * S * self.apf : (i * S + W) * self.apf]
            )

        if self.enc_arch == "slowfast":
            # Convert targets from list of lists to list of len(2).\
            n_t_v_segs = []
            for itr in range(len(t_v_segs[0])):
                items = [x[itr] for x in t_v_segs]
                items = torch.stack(items)
                n_t_v_segs.append(items)
            t_v_segs = copy.deepcopy(n_t_v_segs)
        else:
            t_v_segs.extend(neg_v_segments)

        t_aw_segs.extend(neg_aw_segments)
        t_ae_segs.extend(neg_ae_segments)

        if self.enc_arch != "slowfast":
            t_v_segs = torch.stack(t_v_segs)

        t_aw_segs = torch.stack(t_aw_segs)
        t_ae_segs = torch.stack(t_ae_segs)

        if self.split == "train":
            return (q_v_segs, q_aw_segs, q_ae_segs, t_v_segs, t_aw_segs, t_ae_segs)
        else:
            ordering = torch.cat(
                (torch.tensor([(idx + 1) % (self.__len__())]), torch.tensor(neg_idxs))
            )
            return (
                q_v_segs,
                q_aw_segs,
                q_ae_segs,
                t_v_segs,
                t_aw_segs,
                t_ae_segs,
                idx,
                ordering,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="=Video Textures")
    parser.add_argument(
        "--root",
        "-r",
        default="/home/medhini/audio_video_gan/video_textures",
        type=str,
        help="data root directory",
    )
    parser.add_argument(
        "--video_name", "-vn", default="clock", type=str, help="video name"
    )
    args = parser.parse_args()
    print(args)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    trainset = VideoFrames(args, transform=transform)
    trainloader = DataLoader(trainset, batch_size=2, num_workers=0)

    for i, (frames, labels) in enumerate(trainloader):
        print(frames.shape, frames[0].shape, labels)
        break
