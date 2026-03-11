#!/bin/bash

set -euox pipefail

cd aotriton
git diff > ../aotriton.patch
cd ..

cd pytorch
git diff cmake/External/aotriton.cmake .github/scripts/amd/package_triton_wheel.sh > ../pytorch.patch
cd ..
