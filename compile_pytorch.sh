#!/bin/bash

set -euox pipefail

RESET_VENV=0
BUILD_PYTORCH=0
BUILD_TORCHVISION=0
BUILD_TORCHAUDIO=0
BUILD_TORCHCODEC=1
BUILD_TRITRON=0

# create venv
if [ "$RESET_VENV" -eq "1" ]
then
    rm -rf venv || true
    uv venv venv --python 3.12
fi

source venv/bin/activate

# build versions
export PYTORCH_BUILD_VERSION=2.9.1
export PYTORCH_VISION_VERSION=0.24.1
export PYTORCH_CODEC_VERSION=0.9.1

# build env vars
export PYTORCH_BUILD_NUMBER=0
export AOTRITON_INSTALL_FROM_SOURCE=1
export USE_CUDA=0
export USE_XPU=0
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx1030
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}"
export CMAKE_BUILD_TYPE=Release
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc --all)

# pytorch
if [ "$BUILD_PYTORCH" -eq "1" ]
then
    rm -rf pytorch || true
    git clone --depth 1 --branch v$PYTORCH_BUILD_VERSION https://github.com/pytorch/pytorch.git
    cd pytorch
    git submodule update --init --recursive --depth=1
    git apply ../pytorch.patch

    uv pip install pip
    uv pip install --group dev

    python tools/amd_build/build_amd.py
    python setup.py bdist_wheel

    # sudo apt install libelf-dev libnuma-dev libncurses-dev patchelf
    uv pip install ninja cmake wheel pybind11
    python .github/scripts/build_triton_wheel.py --release --device=rocm

    uv pip install \
        torch@$(echo -n ./dist/torch-*.whl) \
        pytorch_triton_rocm@$(echo -n ./pytorch_triton_rocm-*.whl)

    cd ..
fi

# torchvision
if [ "$BUILD_TORCHVISION" -eq "1" ]
then
    rm -rf vision || true
    git clone --depth 1 --branch v$PYTORCH_VISION_VERSION https://github.com/pytorch/vision.git
    cd vision
    export BUILD_VERSION=$PYTORCH_VISION_VERSION
    python setup.py bdist_wheel
    export -n BUILD_VERSION
    uv pip install torchvision@$(echo -n ./dist/torchvision-*.whl)
    cd ..
fi

# torchaudio
if [ "$BUILD_TORCHAUDIO" -eq "1" ]
then
    rm -rf audio || true
    git clone --depth 1 --branch v$PYTORCH_BUILD_VERSION https://github.com/pytorch/audio.git
    cd audio
    export BUILD_VERSION=$PYTORCH_BUILD_VERSION
    python setup.py bdist_wheel
    export -n BUILD_VERSION
    uv pip install torchaudio@$(echo -n ./dist/torchaudio-*.whl)
    cd ..
fi

# torchcodec
if [ "$BUILD_TORCHCODEC" -eq "1" ]
then
    rm -rf torchcodec || true
    git clone --depth 1 --branch v$PYTORCH_CODEC_VERSION https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git apply ../torchcodec.patch
    uv pip install pkgconfig
    # sudo apt install libavdevice-dev libavfilter-dev
    original_CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
    export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir);$CMAKE_PREFIX_PATH"
    # torchcodec is trying to use NVDEC
    export ENABLE_CUDA=OFF
    export TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR=ON
    export I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=ON
    export BUILD_VERSION=$PYTORCH_CODEC_VERSION
    python setup.py bdist_wheel
    export -n BUILD_VERSION
    export -n I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION
    export -n TORCHCODEC_DISABLE_COMPILE_WARNING_AS_ERROR
    export -n ENABLE_CUDA
    export CMAKE_PREFIX_PATH=$original_CMAKE_PREFIX_PATH
    uv pip install torchcodec@$(echo -n ./dist/torchcodec-*.whl)
    cd ..
fi

# tritron
if [ "$BUILD_TRITRON" -eq "1" ]
then
    TRITRON_VERSION=$(cat pytorch/.ci/docker/triton_version.txt)
    rm -rf triton || true
    git clone --depth 1 --branch v$TRITRON_VERSION https://github.com/triton-lang/triton.git
    cd triton
    #git checkout $(cat pytorch/.ci/docker/ci_commit_pins/tritron.txt)
    git submodule update --init --recursive --depth=1
    uv pip install -r python/requirements.txt
    mv .git .git_release
    python setup.py bdist_wheel
    mv .git_release .git
    uv pip install triton@$(echo -n ./dist/triton-*.whl)
    cd ..
fi
