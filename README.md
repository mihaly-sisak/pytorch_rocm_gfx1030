# PyTorch for ROCm gfx1030

Pre-compiled wheels did not work for me for some reason.
This works on my machine, Ubuntu 24.04, ROCm 7.2, video card RX 6750 XT.

Packages needed for pytorch:
`sudo apt install libelf-dev libnuma-dev libncurses-dev patchelf`

Packages needed by torchcodec:
`sudo apt install libavdevice-dev libavfilter-dev`

Patch notes:
 - aotritron : To me it looks like this is using tritron to pre-compile some python code, just added gfx1030 everywhere
 - pytorch : Apply aotritron patch during install, locate some libs
 - torchcodec : Needed by torchaudio, does not recognize pybind11 in venv without the patch

## Usage

Run `compile_pytorch.sh`. This compiles pytorch, torchvision, torchaudio, tritron for your system.
