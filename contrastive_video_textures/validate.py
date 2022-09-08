import ipdb
from utils import (
    AverageMeter,
    Logger,
    overlay_cmap_image,
    waveform_to_examples,
    save_videos,
    split_into_batches,
    combine_batches,
    split_into_overlapping_segments,
    log_mel_spectrogram,
)
from models import VGGish, VideoForAudio, ModelBuilder3D
from interpolate import interpolate, modify_frames
import torch
import torchvision.transforms as transforms
import torchvision.io as io
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt
import argparse
import os
import math
import shutil
import time
from collections import OrderedDict
from PIL import Image
import subprocess
import librosa
import matplotlib
from dataset import scale_jitter_crop_norm

from types import SimpleNamespace

from slowfast.utils.parser import load_config
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.utils import process_cv2_inputs

matplotlib.use("Agg")


def to_cuda(item):
    if isinstance(item[0], list):
        return [[x.cuda() for x in y] for y in item]
    elif isinstance(item, list):
        return [x.cuda() for x in item]
    return item.cuda()


def construct_cam(weights, act):
    cam = torch.zeros((act.shape[1], act.shape[2]), dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(1), size=(224, 224), mode="bilinear")
    cam = cam.view(cam.shape[2], cam.shape[3])

    return cam

def validate(
    model, args, video_name="", epoch=None, tb_logger=None, model_type=2, itr=0,
):

    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # Switch to evaluate mode.
    model.eval()

    # Define criterion.
    criterion = nn.CrossEntropyLoss()

    # Prepare video.
    video_filename = os.path.join(args.vdata, "{}.mp4".format(video_name))
    video, _, meta = io.read_video(video_filename, pts_unit="sec")

    S = args.stride
    W = args.window

    if args.enc_arch != "slowfast":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737],
                ),
            ]
        )
        inv_t = transforms.Normalize(
            mean=[-0.4345 / 0.2768, -0.4051 / 0.2713, -0.3775 / 0.2737],
            std=[1 / 0.2768, 1 / 0.2713, 1 / 0.2737],
        )
    else:
        sf_args = SimpleNamespace()
        sf_args.cfg_file = "/home/medhini/audio_video_gan/contrastive_video_textures/slowfast_configs/SLOWFAST_8X8_R50.yaml"
        sf_args.opts = None
        cfg = load_config(sf_args)
        cfg.NUM_GPUS = 1
        cfg.TEST.CHECKPOINT_TYPE = "caffe2"
        cfg.TEST.CHECKPOINT_FILE_PATH = "/home/medhini/audio_video_gan/contrastive_video_textures/pretrained/SLOWFAST_8x8_R50.pkl"

        inv_t = None

    if args.model_type in (2, 4):
        idxs = np.arange(len(video) / args.subsample_rate)
        idxs = [x * args.subsample_rate for x in idxs]
        input_frames = video[idxs]

        if args.enc_arch != "slowfast":
            input_frames = torch.stack(
                [transform(i.permute(2, 0, 1)) for i in input_frames]
            )
        else:
            # Scale values to 0-1.
            input_frames = input_frames.float() / 255

            # Convert RGB -> BGR.
            permute = [2, 1, 0]
            input_frames = input_frames[:, :, :, permute]

        # else:
        #     qf_t = [
        #             F.interpolate(
        #                 item.squeeze(0),
        #                 size=(args.img_size, args.img_size),
        #                 mode="bilinear",
        #             )
        #             for item in qf_t
        #         ]

    # if args.model_type == 5:
    #     # Load poses.
    #     pose_filename = os.path.join(args.pdata, "{}.mp4".format(video_name))
    #     input_frames, _, _ = io.read_video(pose_filename, pts_unit="sec")
    #     input_frames = torch.stack(
    #         [transform(i.permute(2, 0, 1)) for i in input_frames]
    #     )

    # Prepare source audio.
    audio_w = None
    apf = 10
    audio_eg = torch.rand((math.floor((len(video) - W) / S)), 10)

    if args.adata != None:
        print("Preparing source audio. ")
        audio_folder = os.path.join(args.adata, "{}.wav".format(video_name))
        if os.path.exists(audio_folder):
            audio_w, sr = librosa.load(audio_folder)
            apf = math.floor((sr * args.subsample_rate) / args.fps)

            # Adjust audio length to match video length.
            audio_w = audio_w[: len(input_frames) * apf]
            audio_eg = waveform_to_examples(audio_w, sr * args.subsample_rate)
            audio_eg = torch.from_numpy(audio_eg).unsqueeze(dim=1)
            audio_eg = audio_eg.float()
            audio_w = torch.tensor(audio_w)

    # Prepare driving audio.
    print("Preparing driving audio. ")
    driving_audio_name = None
    if args.driving_audio is not None:
        driving_audio_name = args.driving_audio[itr]
        da_path = os.path.join(args.dadata, driving_audio_name + ".wav")
        assert os.path.exists(da_path), "No driving audio found at {}".format(da_path)

        driving_audio_w, sr_da = librosa.load(da_path)
        driving_audio_eg = waveform_to_examples(
            driving_audio_w, sr_da * args.subsample_rate
        )
        driving_audio_eg = torch.from_numpy(driving_audio_eg).unsqueeze(dim=1)
        driving_audio_eg = driving_audio_eg.float()

    # Initialize interpolation model.
    print("Initializing interpolation model. ")
    if args.interpolation:
        intp_model = interpolate([video.shape[2], video.shape[1]], args.SF).cuda()
        dict1 = torch.load("ckpt/SuperSloMo.ckpt", map_location="cpu")
        intp_model.ArbTimeFlowIntrp.load_state_dict(dict1["state_dictAT"])
        intp_model.flowComp.load_state_dict(dict1["state_dictFC"])

    # Frame IDs for all, query, +ve, -ve and target(+ve + -ve) frames.
    all_frame_ids = np.arange(len(input_frames))
    all_segment_ids = np.arange(math.floor((len(input_frames) - W) / S))

    # Ensure no. of audio segments equals no. of video segments.
    audio_eg = audio_eg[: len(all_segment_ids)]
    max_audio_segment_id = audio_eg.shape[0] - 1

    L = len(all_segment_ids)

    # if driving_audio_name is None:
    #     q_id = 100
    #     # q_id = np.random.choice(all_segment_ids, 1, replace=False)[0]
    #     print("Start:", q_id)
    # else:
    #     print("Computing driving audio similarity.")
    #     driving_eg = driving_audio_eg[0]
    #     q_id = 0
    #     cos = nn.CosineSimilarity(dim=1)
    #     sim = cos(
    #         F.normalize(
    #             driving_eg.repeat([audio_eg.shape[0], 1, 1, 1]).view(
    #                 audio_eg.shape[0], -1
    #             ),
    #             dim=0,
    #         ),
    #         F.normalize(audio_eg.view(audio_eg.shape[0], -1), dim=0),
    #     )
    #     q_id = int(sim.argmax())
    #     print("Max Audio Sim: ", sim.max())

    if driving_audio_name is None:
        q_id = 10
        # q_id = np.random.choice(all_segment_ids, 1, replace=False)[0]
        print("Start:", q_id)
    else:
        q_id = 0
        max_sim = 0
        driving_eg = driving_audio_eg[0]

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

    new_frames = []
    new_frames_intp = []
    new_frame_ids = []
    new_audio = []
    new_audio_intp = []
    non_zero_counts = []
    entropies = []
    jump_count = 0

    iter_count = 1

    p_q_id = -1  # prev q id

    max_length = math.ceil(args.fps) * args.new_video_length

    # Prepare driving audio model.
    if driving_audio_name is not None:
        max_length = min(
            max_length, np.ceil(args.fps) * np.floor(len(driving_audio_eg) * S + W)
        )
        if args.da_feats == "VGG":
            da_model = VGGish()
            da_model.load_state_dict(torch.load("pytorch_vggish.pth"))

        elif args.da_feats == "Contrastive":
            # Resnet 3D model
            builder = ModelBuilder3D()
            image_enc_model, fc_dim = builder.build_network(
                arch=args.enc_arch, img_size=args.size, window=args.window
            )

            # VGGish Model
            audio_enc_model = VGGish()
            audio_enc_model.load_state_dict(torch.load("pytorch_vggish.pth"))

            da_model = VideoForAudio(
                image_enc_model,
                audio_enc_model,
                af_dim=128,
                vf_dim=fc_dim,
                emb_dim=128,
                temp=args.temp,
            )
            checkpoint = torch.load(args.daf_resume[itr])

            da_model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded VideoForAudio checkpoint '{}' (epoch {})".format(
                    args.daf_resume[itr], checkpoint["epoch"]
                )
            )

        da_model = torch.nn.DataParallel(da_model).cuda()

    # Get weights and activation hook for visualizations.
    if args.vcam:
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        model.module.q_encoder[0].layer4[1].conv2.register_forward_hook(
            get_activation("q_act")
        )
        model.module.t_encoder[0].layer4[1].conv2.register_forward_hook(
            get_activation("t_act")
        )
        q_video = []
        pos_video = []

    end = time.time()

    # Split input data into batches divisible by num_gpus.
    num_gpus = torch.cuda.device_count()

    print("New video length: {}".format(max_length))

    while len(new_frames) < max_length:
        with torch.no_grad():
            print("Query frame: ", q_id)

            # Prepare the query.
            if args.enc_arch != "slowfast":
                qf_t = input_frames[q_id * S : q_id * S + W].unsqueeze(0).cuda()
            else:
                # frames = list(np.asarray(input_frames[q_id * S : q_id * S + W]))
                frames = process_cv2_inputs(
                    input_frames[q_id * S : q_id * S + W].cuda(), cfg
                )

                qf_t = [
                    F.interpolate(
                        item.squeeze(0),
                        size=(args.img_size, args.img_size),
                        mode="bilinear",
                    )
                    for item in frames
                ]

            q_ae_t = audio_eg[min(q_id, max_audio_segment_id)].unsqueeze(0).cuda()
            # q_aw_t = audio_w[q_id * S * apf : (q_id * S + W) * apf].unsqueeze(0).cuda()

            # Duplicate the query across GPUs.
            if args.enc_arch != "slowfast":
                qf_size = [num_gpus] + [1] * (qf_t.dim() - 1)
                b_qf_t = qf_t.repeat(*qf_size)
            else:
                qf_t = [qf_t] * num_gpus

                b_qf_t = []
                for itr in range(len(qf_t[0])):
                    items = [x[itr] for x in qf_t]
                    items = torch.stack(items)
                    b_qf_t.append(items)

            q_ae_size = [num_gpus] + [1] * (q_ae_t.dim() - 1)
            b_q_ae_t = q_ae_t.repeat(*q_ae_size)

            # q_aw_size = [num_gpus] + [1] * (q_aw_t.dim() - 1)
            # b_q_aw_t = q_aw_t.repeat(*q_aw_size)

            # Prepare the positive.
            pos_id = min((q_id + 1), L - 1)

            # Prepare the negatives.
            mask = np.ones(L, dtype=bool)
            mask[[q_id, pos_id]] = False
            n_ids = all_segment_ids[mask, ...]

            # Combine positives and negatives as targets.
            target_segment_ids = np.concatenate((np.array([pos_id]), n_ids), axis=0)
            os_ids_t = torch.tensor(target_segment_ids)

            # Find target frame ids.
            target_frame_ids = []
            for i in target_segment_ids:
                target_frame_ids.extend(list(np.arange(i * S, i * S + W)))

            # Remove non unique but maintain order.
            target_frame_ids = np.array(target_frame_ids)
            _, idxs = np.unique(target_frame_ids, return_index=True)
            target_frame_ids = target_frame_ids[np.sort(idxs)]

            # Divide target video into chunks.
            t_video = input_frames[target_frame_ids]

            t_video_chunks, _ = split_into_overlapping_segments(
                t_video, args.mini_batchsize, W, S
            )

            # Divide target audio into chunks.
            target_audio_segment_ids = [
                min(x, max_audio_segment_id) for x in target_segment_ids
            ]
            t_ae_t = audio_eg[target_audio_segment_ids]

            # t_aw_t = []
            # for i in os_ids_t:
            #     t_aw_t.append(audio_w[i * S * apf : (i * S + W) * apf])
            # t_aw_t = torch.stack(t_aw_t)

            # Prepare target audio batches.
            new_t_ae_t, num_valid = split_into_batches(
                t_ae_t.unsqueeze(0), args.mini_batchsize
            )

            # new_t_aw_t, _ = split_into_batches(t_aw_t.unsqueeze(0), args.mini_batchsize)

            # Prepare the driving audio.
            if driving_audio_name is not None:
                da_t = driving_audio_eg[iter_count]
                # Duplicate the driving audio.
                da_size = [num_gpus] + [1] * (da_t.dim())
                b_da_t = da_t.repeat(*da_size)

            output = torch.zeros(len(target_segment_ids), dtype=torch.float32)

            if args.vcam:
                t_weight = torch.zeros(
                    (len(target_segment_ids), 512), dtype=torch.float32
                )
                t_acts = torch.zeros(
                    (
                        math.ceil(len(t_video_chunks) / num_gpus) * args.mini_batchsize,
                        512,
                        7,
                        7,
                    ),
                    dtype=torch.float32,
                )

            if driving_audio_name is not None:
                output_a = torch.zeros(len(target_segment_ids), dtype=torch.float32)

            # Iterate over target batches.
            for itr in range(math.ceil(len(t_video_chunks) / num_gpus)):
                # Prepare the target.
                b_tf_t = t_video_chunks[itr * num_gpus : itr * num_gpus + num_gpus]
                b_t_ae_t = new_t_ae_t[itr * num_gpus : itr * num_gpus + num_gpus]

                if driving_audio_name is not None:
                    b_output, b_output_a, q_weight, b_t_weight = model(
                        b_qf_t,
                        b_tf_t,
                        q_audio_eg=b_q_ae_t,
                        t_audio_eg=b_t_ae_t,
                        is_inference=True,
                        driving_audio=b_da_t,
                        da_model=da_model,
                        da_feats=args.da_feats,
                        cam_viz=True,
                    )

                    output_a[
                        itr
                        * num_gpus
                        * args.mini_batchsize : itr
                        * num_gpus
                        * args.mini_batchsize
                        + min(num_valid, num_gpus * args.mini_batchsize)
                    ] = b_output_a.view(-1)[
                        : min(num_valid, args.mini_batchsize * num_gpus)
                    ]

                else:
                    b_output, q_weight, b_t_weight = model(
                        b_qf_t,
                        b_tf_t,
                        q_audio_eg=b_q_ae_t,
                        t_audio_eg=b_t_ae_t,
                        is_inference=True,
                        cam_viz=True,
                    )

                output[
                    itr
                    * num_gpus
                    * args.mini_batchsize : itr
                    * num_gpus
                    * args.mini_batchsize
                    + min(num_valid, num_gpus * args.mini_batchsize)
                ] = (
                    b_output.contiguous()
                    .view(-1)[: min(num_valid, args.mini_batchsize * num_gpus)]
                    .detach()
                    .cpu()
                )

                if args.vcam:
                    b_t_weight = b_t_weight.permute(0, 2, 1).detach().cpu()
                    t_weight[
                        itr
                        * num_gpus
                        * args.mini_batchsize : itr
                        * num_gpus
                        * args.mini_batchsize
                        + min(num_valid, num_gpus * args.mini_batchsize),
                        :,
                    ] = (
                        b_t_weight.contiguous()
                        .view(-1, 512)[: min(num_valid, args.mini_batchsize * num_gpus)]
                        .detach()
                        .cpu()
                    )
                    t_acts[
                        itr
                        * num_gpus
                        * args.mini_batchsize : itr
                        * num_gpus
                        * args.mini_batchsize
                        + num_gpus * args.mini_batchsize,
                        :,
                    ] = (activation["t_act"].squeeze(2).detach().cpu())

                # Reduce number of valid segments remaining.
                num_valid -= args.mini_batchsize * num_gpus

            output /= output.sum()
            if driving_audio_name is not None:
                output_a /= output_a.sum()
                output = args.alpha * output + (1 - args.alpha) * output_a

            # Compute loss.
            labels = torch.zeros(1, dtype=torch.long)
            loss = criterion(output.unsqueeze(0), labels)

            # Compute accuracy.
            acc = torch.zeros(1)
            acc[output.argmax() == labels] = 1.0
            acc = acc.mean()

            print("Predicted output: {}".format(output.argmax()))
            print("Predicted output value: {:.6f}".format(output.max()))
            print("Predicted output at 0: {:.6f}".format(output[0]))
            print(
                "Non Random Predicted Next Frame: {}".format(os_ids_t[output.argmax()])
            )
            print("Original Next Frame: {}".format(os_ids_t[0]))

            output_fig = plt.figure()  # create a figure object
            ax = output_fig.add_subplot(1, 1, 1)
            i = ax.imshow(output.repeat(100, 1).numpy(), interpolation="nearest")
            output_fig.colorbar(i)

            tb_logger.log_figure(output_fig, "Probs_Queryframe", iter_count)

            # set values below threshold to 0
            output[output < (output.max() - args.threshold * output.max())] = 0.0

            # renormalize so row sums to 1
            print(torch.nonzero(output).view(-1))
            output[torch.nonzero(output).view(-1)] /= output.sum()

            entropy = abs(output[output.nonzero().view(-1)].log().mean(-1))
            print("Entropy: ", entropy)
            entropies.append(entropy)

            non_zero_count = len(output.nonzero().view(-1))
            print("Non zero: ", non_zero_count)
            non_zero_counts.append(non_zero_count)

            choices = output.nonzero().view(-1)

            rdm_id = np.random.choice(choices.numpy())
            # rdm_id = np.random.choice(np.arange(len(output)), p=output.numpy())
            q_id = os_ids_t[rdm_id].item()

            print("Chosen next frame:", q_id)

            # measure accuracy and record loss
            losses.update(loss.item(), 1)
            accs.update(acc.item(), 1)

            # retrieve current query frames
            intp_added = False
            if p_q_id == -1:
                diff_ids = all_frame_ids[q_id * S : q_id * S + W]
            elif q_id == p_q_id + 1:  # next frame was chosen. no interpolation
                diff_ids = all_frame_ids[q_id * S + (W - S) : q_id * S + W]
            else:
                jump_count += 1
                # interpolate frames
                if args.interpolation:
                    # remove previously added frames
                    new_frames_intp = new_frames_intp[: -int((args.SF - 1) / 2)]

                    intp_added = True

                    # prev diff_id; last frame previously added
                    frame0 = Image.fromarray(np.array(video[diff_ids[-1]]))
                    frame1 = Image.fromarray(
                        np.array(video[(q_id * S + (W - S)) * args.subsample_rate])
                    )

                    frame0, frame1, inv_trans = modify_frames(frame0, frame1)
                    int_frames = intp_model(frame0, frame1, inv_trans)

                    print("Added {} intermediate frames.\n".format(len(int_frames)))

                    for int_frame in int_frames:
                        frame_arr = np.array(int_frame)
                        if args.frames_bar:
                            frames_bar = np.zeros((15, video.shape[-2], 3))
                            frame_arr[-25:-10, :, :] = frames_bar
                        new_frames_intp.append(Image.fromarray(frame_arr))

                diff_ids = all_frame_ids[q_id * S + (W - S) : q_id * S + W]
            # update new frame ids
            new_frame_ids.extend(diff_ids)

            # add new audio
            if driving_audio_name is None and audio_w is not None:
                # add source audio
                new_audio.extend(audio_w[diff_ids[0] * apf : (diff_ids[-1] + 1) * apf])
                new_audio_intp.extend(
                    audio_w[diff_ids[0] * apf : (diff_ids[-1] + 1) * apf]
                )

            # append current query frames to new frames list
            diff_ids = [
                np.arange(i * args.subsample_rate, (i + 1) * args.subsample_rate)
                for i in diff_ids
            ]
            diff_ids = [y for x in diff_ids for y in x]
            for count, idx in enumerate(diff_ids):
                frame_arr = np.array(video[idx])

                if args.frames_bar:
                    frames_bar = np.zeros((15, video.shape[-2], 3))
                    frame_n = int(idx * video.shape[-2] / len(input_frames))
                    frames_bar[:, frame_n - 3 : frame_n + 3, :] = [255, 0, 0]
                    frame_arr[-25:-10, :, :] = frames_bar

                frame_fig = Image.fromarray(frame_arr)

                new_frames.append(frame_fig)
                new_frames_intp.append(frame_fig)

                if intp_added == False or count != 0:
                    for i in range(int((args.SF - 1) / 2)):
                        new_frames_intp.append(frame_fig)

            if q_id != p_q_id + 1 and p_q_id != -1 and non_zero_count > 1:
                q_img = input_frames[p_q_id * S : p_q_id * S + W, :3, :, :]
                # Apply inverse normalization.
                if inv_t is not None:
                    q_img = torch.stack([inv_t(x.detach().cpu()) for x in q_img])
                tb_logger.log_image(q_img, "Query", iter_count)

                p_img = input_frames[pos_id * S : pos_id * S + W, :3, :, :]
                if inv_t is not None:
                    p_img = torch.stack([inv_t(x.detach().cpu()) for x in p_img])
                tb_logger.log_image(p_img, "Positive", iter_count)

                for idx, choice in enumerate(choices):
                    ch_id = os_ids_t[choice].item()
                    ch_img = input_frames[ch_id * S : ch_id * S + W, :3, :, :]
                    if inv_t is not None:
                        ch_img = torch.stack([inv_t(x.detach().cpu()) for x in ch_img])
                    tb_logger.log_image(ch_img, "Choice_{}".format(idx), iter_count)

                ch_img = input_frames[q_id * S : q_id * S + W, :3, :, :]
                if inv_t is not None:
                    ch_img = torch.stack([inv_t(x.detach().cpu()) for x in ch_img])
                tb_logger.log_image(ch_img, "Chosen", iter_count)

            iter_count += 1

            output_fig = plt.figure()  # create a figure object
            ax = output_fig.add_subplot(1, 1, 1)
            i = ax.imshow(output.repeat(100, 1).numpy(), interpolation="nearest")
            output_fig.colorbar(i)

            tb_logger.log_figure(output_fig, "Probs_Queryframe", iter_count)

            tb_logger.flush()

            # update prev q id
            p_q_id = copy.deepcopy(q_id)

    # measure elapsed time
    batch_time.update(time.time() - end)

    print(
        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
        "Acc {acc.val:.4f} ({acc.avg:.4f})".format(
            batch_time=batch_time, loss=losses, acc=accs
        )
    )

    # Log loss and number of jumps.
    if tb_logger is not None:
        logs = OrderedDict()
        logs["Val_EpochLoss"] = losses.avg
        logs["Jump Count"] = jump_count
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, 1)

        tb_logger.flush()
        print("Done logging Loss and Entropies.")

    # create results folder
    results_folder = os.path.join(
        args.results_folder,
        "{}_model_{}_bs_{}_w_{}_stride_{}_temp_{}_th_{}_enca_{}_alpha_{}_intp_{}".format(
            args.logname,
            args.model_type,
            args.batch_size,
            args.window,
            args.stride,
            args.temp,
            args.threshold,
            args.enc_arch,
            args.alpha,
            False,
        ),
    )

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    new_video_id = len(os.listdir(results_folder)) + 1

    # log bar plots for entropy and non zero count
    plt.bar(list(np.arange(len(entropies))), entropies)
    plt.xlabel("Frame Number")
    plt.ylabel("Entropy")
    plt.savefig(os.path.join(results_folder, "entropies_{}.png".format(new_video_id)))
    plt.close()

    plt.bar(list(np.arange(len(non_zero_counts))), non_zero_counts)
    plt.xlabel("Frame Number")
    plt.ylabel("Count of positives")
    plt.savefig(os.path.join(results_folder, "non_zero_{}.png".format(new_video_id)))
    plt.close()

    if args.vcam:
        # Log query cam videos.
        output_video_folder = os.path.join(
            results_folder, "cam_q_video_{}_{}".format(video_name, new_video_id)
        )
        os.makedirs(output_video_folder)

        for count, frame in enumerate(q_video):
            frame_fig = Image.fromarray(
                np.array(frame.permute(1, 2, 0) * 255.0).astype(np.uint8)
            )
            frame_fig.save(
                os.path.join(output_video_folder, "{:04d}.png".format(count + 1))
            )

        outfile = output_video_folder + ".mp4"
        save_videos(output_video_folder, outfile, args.fps)

        # Log pos cam videos.
        output_video_folder = os.path.join(
            results_folder, "cam_p_video_{}_{}".format(video_name, new_video_id)
        )
        os.makedirs(output_video_folder)

        for count, frame in enumerate(pos_video):
            frame_fig = Image.fromarray(
                np.array(frame.permute(1, 2, 0) * 255.0).astype(np.uint8)
            )
            frame_fig.save(
                os.path.join(output_video_folder, "{:04d}.png".format(count + 1))
            )

        outfile = output_video_folder + ".mp4"
        save_videos(output_video_folder, outfile, args.fps)

    # Log video texture result.

    output_video_folder = os.path.join(
        results_folder, "video_{}_{}".format(video_name, new_video_id)
    )
    os.makedirs(output_video_folder)

    print("Frames list: ", new_frame_ids)

    for count, frame_fig in enumerate(new_frames):
        frame_fig.save(
            os.path.join(output_video_folder, "{:04d}.png".format(count + 1))
        )

    outfile = output_video_folder + ".mp4"

    if driving_audio_name is not None:
        new_audio = driving_audio_w[: len(new_frames) * apf]

    # if driving_audio_name is not None or audio_w is not None:
    output_audio_filename = ""
    if driving_audio_name is not None:
        output_audio_filename = os.path.join(
            results_folder, "audio_{}_{}.wav".format(video_name, new_video_id)
        )
        new_audio = np.array(new_audio)
        print("New Audio shape: ", new_audio.shape)
        librosa.output.write_wav(output_audio_filename, new_audio, sr)

    if args.interpolation:
        print("Saving Interpolated Video.\n")

        assert len(new_frames_intp) == int((args.SF + 1) / 2) * len(new_frames)

        results_folder_intp = os.path.join(
            args.results_folder,
            "{}_model_{}_bs_{}_w_{}_stride_{}_temp_{}_th_{}_enca_{}_intp_{}_alpha_{}_SF_{}".format(
                args.logname,
                args.model_type,
                args.batch_size,
                args.window,
                args.stride,
                args.temp,
                args.threshold,
                args.enc_arch,
                args.interpolation,
                args.alpha,
                args.SF,
            ),
        )

        if not os.path.exists(results_folder_intp):
            os.makedirs(results_folder_intp)

        new_video_id = len(os.listdir(results_folder_intp)) + 1
        output_video_folder_intp = os.path.join(
            results_folder_intp, "video_{}_{}".format(video_name, new_video_id)
        )
        os.makedirs(output_video_folder_intp)

        print("Saving frames.")
        for count, frame_fig in enumerate(new_frames_intp):
            frame_fig.save(
                os.path.join(output_video_folder_intp, "{:04d}.png".format(count + 1))
            )

        outfile_intp = output_video_folder_intp + ".mp4"

        if driving_audio_name is not None:
            new_audio_intp = driving_audio_w[: len(new_frames) * apf]

        output_audio_filename_intp = ""
        if audio_w is not None:
            output_audio_filename_intp = os.path.join(
                results_folder_intp, "audio_{}_{}.wav".format(video_name, new_video_id)
            )
            new_audio_intp = np.array(new_audio_intp)
            print("New Intp Audio shape: ", new_audio_intp.shape)
            librosa.output.write_wav(output_audio_filename_intp, new_audio_intp, sr)

    print("Saving frames.")
    save_videos(
        output_video_folder,
        outfile,
        args.fps,
        args.interpolation,
        audio_w,
        args.SF,
        output_audio_filename,
        output_audio_filename_intp,
        output_video_folder_intp,
        outfile_intp,
    )

    return

