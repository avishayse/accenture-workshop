# Workshop - Run Stable Diffusion in cnvrg.io: A Step by Step Guide with Gaudi

cnvrg.io is a comprehensive machine learning platform that provides tools for model development, deployment, version control, and collaboration. It's designed to empower data scientists and machine learning engineers, allowing them to focus on improving models instead of managing infrastructure.

In this guide, you'll learn how to run a Stable Diffusion 2.1 model on cnvrg.io and generate custom images.

# Stable Diffusion 2.1 for PyTorch

In this directory, you'll find scripts for running text-to-image inference using a Stable Diffusion 2.1 model. This model is tested and maintained by Habana.

To learn more about training and inference of deep learning models using Gaudi, visit [developer.habana.ai](https://developer.habana.ai/resources/).

## Table of Contents

* [Model Overview](#model-overview)
* [Setup Instructions](#setup-instructions)
* [Model Checkpoint](#model-checkpoint)
* [Inference and Examples](#inference-and-examples)
* [Supported Configurations](#supported-configurations)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview

This implementation is based on the [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper. Users must comply with any third-party licenses related to these models.

## Setup Instructions

1. Set up your environment by following the instructions in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html). Make sure to set the `$PYTHON` environment variable.

2. Clone the Habana Model-References repository and switch to the branch that matches your SynapseAI version:
    ```bash
    git clone -b master https://github.com/HabanaAI/Model-References
    ```

3. In the docker container, navigate to the model directory:
    ```bash
    cd Model-References/PyTorch/generative_models/stable-diffusion-v-2-1
    ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt --user
    ```

## Model Checkpoint

Download the pre-trained weights for 768x768 and/or 512x512 images:

* For 768x768 images:
    ```bash
    wget https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt
    ```

* For 512x512 images:
    ```bash
    wget https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt
    ```

## Inference and Examples

Generate images using the following command (default output location: `outputs/txt2img-samples`):

* For 768x768 images:
    ```bash
    python3 scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --n_samples 1 --n_iter 3 --use_hpu_graph
    ```

* For 512x512 images:
    ```bash
    python3 scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 512 --W 512 --n_samples 1 --n_iter 3 --steps 35 --k_sampler dpmpp_2m --use_hpu_graph
    ```

For a detailed description of parameters, view the help message:
    ```bash
    python3 scripts/txt2img.py -h
    ```

## Supported Configurations

| Validated on  | SynapseAI Version | PyTorch Version | Mode |
|---------|-------------------|-----------------|----------------|
| Gaudi   | 1.10.0             | 2.0.1          | Inference |
| Gaudi2   | 1.10.0             | 2.0.1          | Inference |

## Changelog

* 1.10.0: Decreased host overhead by rewriting samplers and the main sampling loop.
* 1.8.0: Initial release.

## Known Issues

* Initial random noise generation has been moved to CPU to ensure consistent output.
* The model supports batch sizes up to 16 on Gaudi and up to 8 on Gaudi2 for 512x512px images, and batch size 1 for 768x768px images on Gaudi and Gaudi2.
