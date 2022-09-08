"""
Create folder ckpts and download checkpoints from
https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md

Example:
>> mkdir ckpts; cd ckpts
>> wget https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl

Run this script as follows:
python load_net.py \
    --cfg configs/Kinetics/I3D_8x8_R50.yaml \
    --checkpoint ckpts/I3D_8x8_R50.pkl
"""
import argparse
import sys

import torch
import torch.nn as nn

from slowfast.utils.parser import load_config, parse_args
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.utils import process_cv2_inputs

from types import SimpleNamespace


def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint."
    )
    args, _ = parser.parse_known_args()
    return args


def main():
    """
    Main function to spawn the train and test process.
    """
    my_args = parse_my_args()
    # Remove from arg list before passing to builder.
    idx = sys.argv.index("--checkpoint")
    del sys.argv[idx : idx + 2]

    # Load args and do surgery on args.
    args = SimpleNamespace()
    args.cfg_file = "/home/medhini/audio_video_gan/contrastive_video_textures/slowfast_configs/SLOWFAST_8X8_R50.yaml"
    args.opts = None
    cfg = load_config(args)
    cfg.NUM_GPUS = 1
    cfg.TEST.CHECKPOINT_TYPE = "caffe2"
    cfg.TEST.CHECKPOINT_FILE_PATH = my_args.checkpoint

    # Load the network and remove the head.
    model = ActionPredictor(cfg=cfg)
    net = model.predictor.model

    net.head.dropout = nn.Identity()
    net.head.projection = nn.Identity()
    net.head.act = nn.Identity()  # Make head pass-through.

    # Prepare fake input data.
    print("Input images should be in this format:", cfg.DEMO.INPUT_FORMAT)
    num_frames = 16
    images = [torch.rand(224, 224, 3).numpy() for _ in range(num_frames)]

    # Sample input frames and prepare network input.
    # This function below samples frames as specified by cfg.DATA.NUM_FRAMES.
    # Question to Medhini: What is cfg.DATA.SAMPLING_RATE?

    # Expected input to `process_cv2_inputs`:
    # frames (list of HxWxC np.uint8 array): list of input images
    # (corresponding to one clip)
    # in range [0, 255] in cfg.DEMO.INPUT_FORMAT (RGB/BGR) order.
    input_frames = process_cv2_inputs(images, cfg)
    input_frames = [item.cuda() for item in input_frames]
    print("Inputs:")
    for item in input_frames:
        print(item.size())

    # Do forward.
    output = net(input_frames)
    print("Outputs:")
    for item in output:
        print(item.size())


if __name__ == "__main__":
    main()
