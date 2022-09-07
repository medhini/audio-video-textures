# Audio-Conditioned Video Texture Generation

This is the official Pytorch implementation for the paper, "Strumming to the Beat: Audio-Coniditoned Contrastive Video Texture Synthesis", WACV 2022. We provide the datasets and code for training and testing the contrastive video texture synthesis model and the baselines as described in the paper.  

If you find our repo useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@InProceedings{Narasimhan_2022_WACV,
    author    = {Narasimhan, Medhini and Ginosar, Shiry and Owens, Andrew and Efros, Alexei A. and Darrell, Trevor},
    title     = {Strumming to the Beat: Audio-Conditioned Contrastive Video Textures},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3761-3770}
}
```

## Environment Setup

Create the conda environment from the yaml file and activate the environment,

```
conda env create -f avgan.yml
conda activate avgan
```

<!-- ## Dataset

### Unconditional Video Texture Synthesis Dataset
First, extract the files in datasets/how_to_steps.tar.gz

```
tar -xzvf datasets/how_to_steps.tar.gz
```

Next, download the WikiHow Summaries dataset to the datasets folder by running

```
cd datasets
python wikihow_video_download.py
```

The WikiHow summary annotations are found in ``datasets/wikihow_summ_annt.json`` 

### Pseudo Summaries Training Dataset

We use videos from [COIN](https://coin-dataset.github.io/) and [CrossTask](https://github.com/DmZhukov/CrossTask) to create the Pseudo Summaries Training Dataset. Please follow the instructions on the original websites to download the videos (at 8 fps) and the corresponding subtitles (in .txt format). For COIN, we use yt-dlp to download subtitles. We combined the two datasets to create pseudo summaries comprising of 12,160 videos, whilst using the videos that were common to both datasets only once. We provide YouTube video IDs of the 12160 videos in ``datasets/pseudo_summary_video_ids.txt``. 

Place the videos and ASR in datasets/pseudoGT_videos and datasets/pseudoGT_asr. We also use the task annotations from both the datasets, which we  

## Pseudo Ground-Truth Summary Generation

We provide the pseudo ground-truth summary annotations computed by our algorithm in ``datasets/pseudoGT_annts.json``. 

The next steps in this section describe how we generated these summaries, and need not be run unless you want to tweak our method and generate your own pseudo summaries. To generate pseudo summaries, we first extract MIL-NCE features for all the videos in Pseudo Summaries Training Dataset and then run our pseudo summary generation algorithm. 

### MIL-NCE Feature Extraction (optional)

Our Pseudo GT Summary Generation Algorithm uses MIL-NCE features. Our scripts for extracting features are based on this [code](https://github.com/antoine77340/S3D_HowTo100M). Follow the instructions [here](https://github.com/antoine77340/S3D_HowTo100M#getting-the-data) to download the weights and word dictionary and place them in ``~/Instructional-Video-Summarization/pretrained_weights``.

Run the following to extract features. The script uses a single GPU.  

```
cd feature_extraction
python extract_text_video_feats.py
```

### Pseudo Ground-Truth Summary Generation Algorithm (optional)

These were generated by running the pseudo summary generation algorithm, 

```
cd ~/Instructional-Video-Summarization
python pseudo_gt_generation.py
```

## IV-Sum Training and Evaluation

To train and evaluate IVSum on the Pseudo GT Summary Dataset, run

```
python -m torch.distributed.launch --nproc_per_node=8 main_distributed.py
```

To test IVSum on the WikiHow Summaries Dataset, run

```
python test.py
```
## Coming soon

Baselines

Model Checkpoints -->
