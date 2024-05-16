<p align="center">
  <h1 align="center">Unifying Flow, Stereo and Depth Estimation</h1>
  <h3 align="center"><a href="https://arxiv.org/abs/2211.05783">Paper</a> | <a href="https://haofeixu.github.io/slides/20221228_synced_unimatch.pdf">Slides</a> | <a href="https://haofeixu.github.io/unimatch/">Project Page</a> | <a href="https://colab.research.google.com/drive/1r5m-xVy3Kw60U-m5VB-aQ98oqqg_6cab?usp=sharing">Colab</a> | <a href="https://huggingface.co/spaces/haofeixu/unimatch">Demo</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/unimatch/resources/teaser.png" alt="Logo" width="70%">
  </a>
</p>

---

## Installation

Our code is developed based on pytorch 1.9.0, CUDA 10.2 and python 3.8. Higher version pytorch should also work well.

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda env create -f conda_environment.yml
conda activate unimatch
```

Alternatively, we also support installing with pip:

```
bash pip_install.sh
```

## Model Zoo

A large number of pretrained models with different speed-accuracy trade-offs for flow, stereo and depth are available at [MODEL_ZOO.md](MODEL_ZOO.md).

We assume the downloaded weights are located under the `pretrained` directory.

Otherwise, you may need to change the corresponding paths in the scripts.

## Demo

Given an image pair or a video sequence, our code supports generating prediction results of optical flow, disparity and depth.

Please refer to [scripts/gmflow_demo.sh](scripts/gmflow_demo.sh), [scripts/gmstereo_demo.sh](scripts/gmstereo_demo.sh) and [scripts/gmdepth_demo.sh](scripts/gmdepth_demo.sh) for example usages.

https://user-images.githubusercontent.com/19343475/199893756-998cb67e-37d7-4323-ab6e-82fd3cbcd529.mp4

## Datasets

The datasets used to train and evaluate our models for all three tasks are given in [DATASETS.md](DATASETS.md)

## Evaluation

The evaluation scripts used to reproduce the numbers in our paper are given in [scripts/gmflow_evaluate.sh](scripts/gmflow_evaluate.sh), [scripts/gmstereo_evaluate.sh](scripts/gmstereo_evaluate.sh) and [scripts/gmdepth_evaluate.sh](scripts/gmdepth_evaluate.sh).

For submission to KITTI, Sintel, Middlebury and ETH3D online test sets, you can run [scripts/gmflow_submission.sh](scripts/gmflow_submission.sh) and [scripts/gmstereo_submission.sh](scripts/gmstereo_submission.sh) to generate the prediction results. The results can be submitted directly.

## Training

All training scripts for different model variants on different datasets can be found in [scripts/\*\_train.sh](scripts).

We support using tensorboard to monitor and visualize the training process. You can first start a tensorboard session with

```
tensorboard --logdir checkpoints
```

and then access [http://localhost:6006](http://localhost:6006/) in your browser.
