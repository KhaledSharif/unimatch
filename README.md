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

### Original Requirements
Pytorch 1.9.0, CUDA 10.2, Python 3.8.0

### Fork Requirements
Pytorch 2.3.0, CUDA 12.4, Python 3.10.12

## Model Zoo

A large number of pretrained models with different speed-accuracy trade-offs for flow, stereo and depth are available at [MODEL_ZOO.md](MODEL_ZOO.md).

We assume the downloaded weights are located under the `pretrained` directory.

Otherwise, you may need to change the corresponding paths in the scripts.

## Demo

Given an image pair or a video sequence, our code supports generating prediction results of optical flow, disparity and depth.

Please refer to [scripts/gmflow_demo.sh](scripts/gmflow_demo.sh), [scripts/gmstereo_demo.sh](scripts/gmstereo_demo.sh) and [scripts/gmdepth_demo.sh](scripts/gmdepth_demo.sh) for example usages.

## Datasets

The datasets used to train and evaluate our models for all three tasks are given in [DATASETS.md](DATASETS.md)

## Evaluation

The evaluation scripts used to reproduce the numbers in our paper are given in [scripts/gmflow_evaluate.sh](scripts/gmflow_evaluate.sh), [scripts/gmstereo_evaluate.sh](scripts/gmstereo_evaluate.sh) and [scripts/gmdepth_evaluate.sh](scripts/gmdepth_evaluate.sh)

For submission to KITTI, Sintel, Middlebury and ETH3D online test sets, you can run [scripts/gmflow_submission.sh](scripts/gmflow_submission.sh) and [scripts/gmstereo_submission.sh](scripts/gmstereo_submission.sh) to generate the prediction results

The `inference_stereo` function in `evaluate_stereo.py` runs inference on a set of stereo image pairs (provided as a directory or separate left/right directories) and saves the predicted disparity maps. It supports options like bidir disparity prediction, right disparity prediction, and saving disparity maps in PFM format.

- If an `inference_size` is not provided, the images are padded to the nearest size divisible by the `padding_factor` using the InputPadder utility.
- If an `inference_size` is provided, the images are resized to that size using bilinear interpolation.
- If `pred_bidir_disp` is set, the left and right images are horizontally flipped and concatenated along the batch dimension to enable bidirectional disparity estimation.
- If `pred_right_disp` is set, the left and right images are swapped (horizontally flipped) to predict the right disparity map.

## Training

All training scripts for different model variants on different datasets can be found in [scripts/\*\_train.sh](scripts)

We support using tensorboard to monitor and visualize the training process. You can first start a tensorboard session with

```
tensorboard --logdir checkpoints
```

and then access [http://localhost:6006](http://localhost:6006/) in your browser

The code in `main_stereo.py` performs stereo matching, which is the task of estimating the per-pixel disparity (shift) between a pair of rectified stereo images, using the UniMatch model.

- Data Loading: The build_dataset function from dataloader.stereo.datasets is used to load stereo image pairs and ground truth disparity maps for training.
- Model: The UniMatch model is instantiated with the specified hyperparameters like number of scales, feature channels, transformer layers, etc. & the task argument is set to 'stereo' for stereo matching
- Training Loop: For each training batch, the left and right stereo images are passed to the model along with attention/correlation configurations
The model predicts multi-scale disparity maps pred_disps
The loss is computed as the smooth L1 loss between the predicted and ground truth disparities, weighted across multiple scales
The loss is backpropagated, and the model is optimized using an AdamW optimizer
- Evaluation/Validation: The validate_things, validate_kitti15, validate_eth3d, and validate_middlebury functions are used to evaluate the model on respective datasets. These functions pass the stereo pairs through the model to obtain predicted disparities and compute evaluation metrics like EPE (End-Point Error), D1 (percentage of pixels with disparity error > 1), and other dataset-specific metrics
- Inference/Submission: The inference_stereo function is used for making predictions on new stereo pairs. The create_kitti_submission, create_eth3d_submission, and create_middlebury_submission functions generate submission files in the required formats for respective benchmarks

---

## Appendix

```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Mar_28_02:18:24_PDT_2024
Cuda compilation tools, release 12.4, V12.4.131
Build cuda_12.4.r12.4/compiler.34097967_0
```
