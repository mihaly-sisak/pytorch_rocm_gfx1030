#!/bin/bash

set -euox pipefail

rm -rf venv_test || true
uv venv venv_test --python 3.12
source venv_test/bin/activate

uv pip install \
    torch@$(echo -n ./pytorch/dist/torch-*.whl) \
    pytorch_triton_rocm@$(echo -n ./pytorch/pytorch_triton_rocm-*.whl) \
    torchvision@$(echo -n ./vision/dist/torchvision-*.whl) \
    torchaudio@$(echo -n ./audio/dist/torchaudio-*.whl) \
    torchcodec@$(echo -n ./torchcodec/dist/torchcodec-*.whl) \
    triton@$(echo -n ./triton/dist/triton-*.whl)

export HSA_OVERRIDE_GFX_VERSION=10.3.0

./test_install.py
