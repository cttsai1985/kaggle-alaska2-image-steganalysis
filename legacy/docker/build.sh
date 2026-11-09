#!/bin/bash

IMG_REPO=cttsai1985/ml-env-torch-vision

TORCH_VER=1.5.1
CUDA_VER=10.1
BASE_IMAGE=pytorch/pytorch:${TORCH_VER}-cuda${CUDA_VER}-cudnn7-devel

echo "use base: ${BASE_IMAGE}"
docker pull ${BASE_IMAGE}

#
PRAMS=--no-cache
docker build --rm -t ${IMG_REPO} --build-arg BASE_IMAGE=${BASE_IMAGE} -f Dockerfile .

