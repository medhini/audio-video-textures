import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_models import resnet3d, resnext3d, densenet3d


class VideoForAudio(nn.Module):
    def __init__(
        self, base_video_model, base_audio_model, af_dim, vf_dim, emb_dim, temp
    ):
        super(VideoForAudio, self).__init__()

        self.af_dim = af_dim
        self.vf_dim = vf_dim
        self.emb_dim = emb_dim

        self.temp = temp

        self.audio_enc = base_audio_model

        self.audio_mlp = nn.Sequential(
            nn.Linear(512 * 12, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.af_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.af_dim, self.emb_dim),
            nn.ReLU(inplace=True),
        )

        self.video_enc = nn.Sequential(
            base_video_model, nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.video_mlp = nn.Sequential(
            nn.Linear(self.vf_dim, self.emb_dim), nn.ReLU(inplace=True),
        )

    def forward(self, q_a, t_v):

        batch_size = q_a.shape[0]
        n_targets = t_v.shape[1]
        t_len = t_v.shape[2]
        C = t_v.shape[3]
        H, W = t_v.shape[4], t_v.shape[5]

        t_v = t_v.permute(0, 1, 3, 2, 4, 5)
        t_v = t_v.contiguous().view(-1, C, t_len, H, W)
        t_v = self.video_enc(t_v).view(-1, self.vf_dim)
        t_v = self.video_mlp(t_v)
        t_v = t_v.view(-1, n_targets, self.emb_dim)
        t_v = nn.functional.normalize(t_v, dim=2)

        q_a = self.audio_enc(q_a)
        q_a = self.audio_mlp(q_a)
        q_a = nn.functional.normalize(q_a, dim=1)
        q_a = q_a.unsqueeze(1)

        t_v = t_v.permute(0, 2, 1)

        # compute logits
        output = torch.bmm(q_a, t_v).squeeze(1)
        output /= self.temp

        return output


class ModelBuilder3D(object):
    def __init__(self):
        pass

    @staticmethod
    def build_network(arch="resnet18", img_size=224, window=20, pretrained=True):
        assert arch in [
            "resnet10",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnext50",
            "resnext101",
            "resnext152" "densenet121",
        ]
        if "resnet" in arch:
            model = resnet3d.__dict__[arch](
                sample_size=img_size, sample_duration=window, pretrained=pretrained
            )

        elif "resnext" in arch:
            model = resnext3d.__dict__[arch](
                sample_size=img_size, sample_duration=window, pretrained=pretrained
            )

        elif "densenet" in arch:
            model = densenet3d.__dict__[arch](
                sample_size=img_size, sample_duration=window, pretrained=pretrained
            )
        else:
            raise Exception("Architecture undefined!")

        return model, model.fc_dim
