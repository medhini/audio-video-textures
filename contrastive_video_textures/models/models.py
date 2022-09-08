import sys
import math
import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import resnet
from .video_models import resnet3d, resnext3d, densenet3d
from .audio_visual_features import AudioVisualFeatures
from utils import log_mel_spectrogram

from types import SimpleNamespace

from slowfast.utils.parser import load_config
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.utils import process_cv2_inputs


class RandomTexture(nn.Module):
    def __init__(self):
        super(RandomTexture, self).__init__()

    def forward(self, current_emb, frames):
        new_frame = np.random.choice(len(self.frames))
        return


class ContrastivePredictionTemporal(nn.Module):
    def __init__(
        self,
        q_image_enc_model,
        t_image_enc_model,
        audio_enc_model,
        model_type,
        fc_dim,
        temp=0.1,
        window=20,
        stride=2,
        threshold=0.20,
        mini_batchsize=20,
        dropout=0.5,
        enc_arch="resnet",
        img_size=224,
    ):
        super(ContrastivePredictionTemporal, self).__init__()

        if enc_arch != "slowfast":
            self.q_encoder = nn.Sequential(
                q_image_enc_model,
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.t_encoder = nn.Sequential(
                t_image_enc_model,
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
        else:
            self.q_encoder = q_image_enc_model
            self.t_encoder = t_image_enc_model

        if model_type == 3:
            self.q_a_encoder = audio_enc_model
            self.q_a_mlp = nn.Sequential(
                nn.Linear(512 * 48, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 128),
                nn.ReLU(inplace=True),
            )

            self.t_a_encoder = audio_enc_model
            self.t_a_mlp = nn.Sequential(
                nn.Linear(512 * 48, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 128),
                nn.ReLU(inplace=True),
            )

        self.temp = temp
        self.fc_dim = fc_dim
        self.window = window
        self.stride = stride
        self.threshold = threshold
        self.mini_batchsize = mini_batchsize
        self.model_type = model_type
        self.enc_arch = enc_arch
        self.img_size = img_size

        if self.enc_arch == "slowfast":
            sf_args = SimpleNamespace()
            sf_args.cfg_file = "/home/medhini/audio_video_gan/contrastive_video_textures/slowfast_configs/SLOWFAST_8X8_R50.yaml"
            sf_args.opts = None
            self.cfg = load_config(sf_args)
            self.cfg.NUM_GPUS = 1
            self.cfg.TEST.CHECKPOINT_TYPE = "caffe2"
            self.cfg.TEST.CHECKPOINT_FILE_PATH = "/home/medhini/audio_video_gan/contrastive_video_textures/pretrained/SLOWFAST_8x8_R50.pkl"

        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        q_f,
        t_f,
        q_audio_eg=None,
        t_audio_eg=None,
        is_inference=False,
        driving_audio=None,
        da_model=None,
        da_feats=None,
        cam_viz=False,
    ):

        if self.enc_arch == "slowfast":
            C = q_f[0].shape[1]
            H, W = q_f[0].shape[3], q_f[0].shape[4]
            batch_size = q_f[0].shape[0]
        else:
            C = q_f.shape[2]
            H, W = q_f.shape[3], q_f.shape[4]
            batch_size = q_f.shape[0]

        # Compute query features.
        if self.enc_arch != "slowfast":
            # (B,window,C,H,W) -> (B,C,window,H,W)
            q_f = q_f.permute(0, 2, 1, 3, 4)
            q_f = q_f.contiguous().view(-1, C, self.window, H, W)

        q_f = self.q_encoder(q_f)
        q_f = q_f.view(batch_size, -1)

        if self.model_type == 3:
            A_c = t_audio_eg.shape[2]
            A_w, A_h = t_audio_eg.shape[3], t_audio_eg.shape[4]

            q_a = q_audio_eg.contiguous().view(-1, A_c, A_w, A_h)
            q_a = self.q_a_encoder.forward(q_a)
            q_a = q_a.view(batch_size, -1)

            q = torch.cat((q_f, q_a), dim=1)
        else:
            q = q_f

        q = nn.functional.normalize(q, dim=1)
        q = q.unsqueeze(1)

        # If validation, prepare the segments.
        if self.training == False:
            t_emb = []
            if self.enc_arch != "slowfast":
                for i in range(self.mini_batchsize):
                    t_emb.append(
                        t_f[0, i * self.stride : i * self.stride + self.window]
                    )
                t_f = torch.stack(t_emb).unsqueeze(0).cuda()  # unsqueeze for B=1
            else:
                for i in range(self.mini_batchsize):
                    frames = process_cv2_inputs(
                        t_f[0, i * self.stride : i * self.stride + self.window].cuda(),
                        self.cfg,
                    )
                    frames = [
                        F.interpolate(
                            item.squeeze(0),
                            size=(self.img_size, self.img_size),
                            mode="bilinear",
                        )
                        for item in frames
                    ]
                    t_emb.append(frames)
                # Convert targets from list of lists to list of len(2).
                t_f = []
                for itr in range(len(t_emb[0])):
                    items = [x[itr] for x in t_emb]
                    items = torch.stack(items)
                    t_f.append(items.unsqueeze(0))  # unsqueeze for B=1

        # Compute key features.
        if self.enc_arch != "slowfast":
            t_len = t_f.shape[1]
            # (B,num_target,window,C,H,W) -> #(B,num_target,C,window,H,W)
            t_f = t_f.permute(0, 1, 3, 2, 4, 5)
            t_f = t_f.contiguous().view(-1, C, self.window, H, W)
        else:
            # Input to slowfast is a list where list[0] is B,num_targets,C,8,H,W and list[1] is B,num_targets,C,32,H,W
            t_len = t_f[0].shape[1]
            # list[0]; (B,num_target,C,8,H,W) -> (B*num_target,C,8,H,W)
            t_f[0] = t_f[0].view(-1, C, 8, H, W)
            # list[1]; (B,num_target,C,32,H,W) -> (B*num_target,C,32,H,W)
            t_f[1] = t_f[1].view(-1, C, 32, H, W)

        t_f_emb = self.t_encoder(t_f)
        t_f_emb = t_f_emb.view(batch_size, t_len, -1)

        if self.model_type == 3:
            t_a = t_audio_eg.contiguous().view(-1, A_c, A_w, A_h)
            t_a = self.t_a_encoder.forward(t_a)
            t_a = t_a.view(batch_size, t_len, -1)
            t = torch.cat((t_f_emb, t_a), dim=2)
        else:
            t = t_f_emb

        t = nn.functional.normalize(t, dim=2)
        t = t.permute(0, 2, 1)

        # Compute logits.
        output = torch.bmm(q, t).squeeze(1)
        output /= self.temp

        # If driving audio is not None.
        if driving_audio is not None:
            A_c = t_audio_eg.shape[2]
            A_w, A_h = t_audio_eg.shape[3], t_audio_eg.shape[4]

            if da_feats == "VGG":
                s_a = t_audio_eg.contiguous().view(-1, A_c, A_w, A_h)
                s_a = da_model.forward(s_a)
                s_a = s_a.view(batch_size, t_len, -1)

                d_a = driving_audio.contiguous().view(-1, A_c, A_w, A_h)
                d_a = da_model.forward(d_a)
                d_a = d_a.view(batch_size, -1)

                s_a = F.normalize(s_a, dim=2)
                s_a = s_a.permute(0, 2, 1)

                d_a = F.normalize(d_a, dim=1)
                d_a = d_a.unsqueeze(1)

                output_a = torch.bmm(d_a, s_a).to(output.device)

            elif da_feats == "Contrastive":
                d_a = driving_audio.contiguous().view(-1, A_c, A_w, A_h)
                output_a = da_model(driving_audio, t_f).to(output.device)

            else:
                s_a = t_audio_eg.contiguous().view(batch_size, t_audio_eg.shape[1], -1)
                d_a = driving_audio.contiguous().view(batch_size, -1)

                s_a = F.normalize(s_a, dim=2)
                s_a = s_a.permute(0, 2, 1)

                d_a = F.normalize(d_a, dim=1)
                d_a = d_a.unsqueeze(1)

                output_a = torch.bmm(d_a, s_a).to(output.device)

            output_a /= self.temp

            if cam_viz:
                return output, output_a, q, t
            else:
                return output, output_a

        if cam_viz:
            return output, q, t
        else:
            return output


class Resnet(nn.Module):
    def __init__(self, orig_resnet, fc_dim):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4
        self.avgpool = orig_resnet.avgpool
        self.fc_dim = fc_dim

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


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
            "resnext152",
            "densenet121",
            "slowfast",
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
        elif "slowfast" in arch:
            # Load args and do surgery on args.
            args = SimpleNamespace()
            args.cfg_file = "/home/medhini/audio_video_gan/contrastive_video_textures/slowfast_configs/SLOWFAST_8X8_R50.yaml"
            args.opts = None
            cfg = load_config(args)
            cfg.NUM_GPUS = 1
            cfg.TEST.CHECKPOINT_TYPE = "caffe2"
            cfg.TEST.CHECKPOINT_FILE_PATH = "/home/medhini/audio_video_gan/contrastive_video_textures/pretrained/SLOWFAST_8x8_R50.pkl"

            # Load the network and remove the head.
            net = ActionPredictor(cfg=cfg)
            model = net.predictor.model
            model.head.dropout = nn.Identity()
            model.head.projection = nn.Identity()
            model.head.act = nn.Identity()  # Make head pass-through.

        else:
            raise Exception("Architecture undefined!")
        return model, 128
