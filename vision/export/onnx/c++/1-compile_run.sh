#!/bin/bash

# https://pytorch.org/tutorials/advanced/cpp_export.html
#   wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
#   unzip libtorch-shared-with-deps-latest.zip
rm -R build
mkdir build
cd build || exit
cmake -DCMAKE_PREFIX_PATH=./libtorch ..
cmake --build . --config Release
echo
taskset -c 0 ./inference traced_model.pt
