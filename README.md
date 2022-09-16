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

## Dataset

Coming soon!

## Contrastive Video Textures

```cd contrastive_video_textures```

Train model for a single video:

```
python main.py -vdata <path to video folder> -m 1 -w 20 -stride 4 -temp 0.1 -th 0.0 -bs 8 -negs 14 -vl <list of video names> -ea slowfast -lr 1e-4
```

Synthesize texture for the same video using the above model:

```
python main.py -vdata <path to video folder> -m 1 -w 20 -stride 4 -temp 0.1 -th 0.3 -bs 24 -vl <list of video names> -e -mbs 100
```

## Audio-Conditioned Contrastive Video Texture Synthesis

First, train a contrastive model for the video using the command above. Ensure that the audio for the same video is in the audio folder as a wav file with the same name. Next, to synthesize a new video conditioned on an audio, 

```
python main.py -vdata <path to video folder> -adata <path to audio folder> -m 2 -w 20 -stride 4 -temp 0.1 -th 0.0 -bs 24 -negs 20 -e -vl <list of video names> -da <list of coniditioning audios> -alpha 0.5 
```

## Baselines

```cd baselines```

### Video Textures Baslines

```cd classic_video_textures```

1. Classic: 

```
python video_textures.py -m 1 -vdata <source video folder> -vl <list of video names> -s -bs 48
```

2. Classic+:

```
python video_textures.py -m 2 -vdata <source video folder> -vl <list of video names> -s -bs 48
```

3. Classic++: 

```
python video_textures.py -m 3 -vdata <source video folder> -vl <list of video names> -s -bs 48
```

### Audio-Conditioned Video Textures Baselines

```cd audio_baselines```

1. Random Clip: ```python random_segment_baseline.py -vl <original_video_list> -tl <target_audio_list>```
2. Random Baseline: ```python random_baseline.py -vl <original_video_list> -tl <target_audio_list>```
3. Random Shift: ```python random_shift.py -vl <original_video_list> -tl <target_audio_list>```
4. Audio Nearest Neighbour: ```python audio_nearestneighbour.py -vl <original_video_list> -dl <target_audio_list>```
