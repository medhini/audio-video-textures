import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class AudioBlock(nn.Module):
    # Resnet block for audio
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample=None):
        super(AudioBlock, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class VideoBlock(nn.Module):
    # 3D Resnet block for video
    expansion = 1
    def __init__(self, in_channels, out_channels, kernel_size=(1,1,1), stride=1, downsample=None, padding=0):
        super(VideoBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(1,1,1), stride=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

def Linear(in_features, out_features, dropout=0.):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)

class AudioVisualFeatures(nn.Module):
    def __init__(self, W=15):
        super(AudioVisualFeatures, self).__init__()
        """Sound Features"""
        self.conv1_1 = nn.Conv1d(1, 64, 9, stride=4, padding=0, dilation=1, groups=1, bias=True)
        self.pool1_1 = nn.MaxPool1d(4, stride=4)

        self.s_net_1 = self._make_layer(AudioBlock, 64, 128, 3, 2, 1)
        self.s_net_2 = self._make_layer(AudioBlock, 128, 128, 3, 2, 1)
        self.s_net_3 = self._make_layer(AudioBlock, 128, 256, 3, 2, 1)
        
        self.pool1_2 = nn.MaxPool1d(3, stride=3)
        self.conv1_2 = nn.Conv1d(256, 128, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        
        """Image Features"""
        self.conv3_1 = nn.Conv3d(3, 64, (5,7,7), (2,2,2), padding=(2,3,3), dilation=1, groups=1, bias=True)
        self.pool3_1 = nn.MaxPool3d((1,3,3), (1,2,2), padding=(0,1,1))
        self.im_net_1 = self._make_layer(VideoBlock, 64, 64, (3,3,3), (2,2,2), 2)

        """Fuse Features"""
        if W==15:
            self.fractional_maxpool = nn.FractionalMaxPool2d((3,1), output_size=(4, 1))
        else:
            self.fractional_maxpool = nn.FractionalMaxPool2d((3,1), output_size=(8, 1))
        self.conv3_2 = nn.Conv3d(192, 512, (1, 1, 1))
        self.conv3_3 = nn.Conv3d(512, 128, (1, 1, 1))
        self.joint_net_1 = self._make_layer(VideoBlock, 128, 128, (3,3,3), (2,2,2), 2)
        self.joint_net_2 = self._make_layer(VideoBlock, 128, 256, (3,3,3), (1,2,2), 2)
        self.joint_net_3 = self._make_layer(VideoBlock, 256, 512, (3,3,3), (1,2,2), 2)

    def _make_layer(self, block, in_channels, out_channels, kernel_size, stride, blocks):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            if isinstance(kernel_size, int):
                downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels * block.expansion, kernel_size, stride),
                    nn.BatchNorm1d(out_channels * block.expansion),
                )
                layers = []
                layers.append(block(in_channels, out_channels, kernel_size, stride, downsample))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels * block.expansion, kernel_size, stride, padding=1),
                    nn.BatchNorm3d(out_channels * block.expansion),
                )
                layers = []
                layers.append(block(in_channels, out_channels, kernel_size, stride, downsample, padding=1))

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, images, sounds, is_inference=False):
        batchsize = sounds.shape[0]

        sounds = sounds.view(batchsize, 1, -1)

        batchsize, channels, window, h, w = images.shape
        
        out_s = self.conv1_1(sounds)
        out_s = self.pool1_1(out_s)
        out_s = self.s_net_1(out_s)
        out_s = self.s_net_2(out_s)
        out_s = self.s_net_3(out_s)

        out_s = self.pool1_2(out_s)
        out_s = self.conv1_2(out_s)
        
        out_im = self.conv3_1(images)
        out_im = self.pool3_1(out_im)
        out_im = self.im_net_1(out_im)

        #tile audio, concatenate channel wise
        out_s = self.fractional_maxpool(out_s.unsqueeze(3)) # Reduce dimension from 25 to 8

        # print("Out_image shape:", out_im.shape)
        # print("Out s shape:", out_s.shape)
        out_s = out_s.unsqueeze(4)
        out_s = out_s.repeat(1, 1, 1, 28, 28) # Tile
        
        out_joint = torch.cat((out_s, out_im),1)
        out_joint = self.conv3_2(out_joint)
        out_joint = self.conv3_3(out_joint)
        out_joint = self.joint_net_1(out_joint)
        out_joint = self.joint_net_2(out_joint)
        out_joint = self.joint_net_3(out_joint)
    
        return out_joint