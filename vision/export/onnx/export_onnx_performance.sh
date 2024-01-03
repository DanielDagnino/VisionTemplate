#!/bin/bash

export PYTHONPATH=.:../../common
export OMP_WAIT_POLICY=ACTIVE

N_CPU=1
echo -e "\nPerformance with ${N_CPU} CPUs"
export OMP_NUM_THREADS=${N_CPU}

taskset -c 0-$[${N_CPU}-1] ./export/export_onnx.py --n_threads ${N_CPU} --sequential \
  --input_size 640 --arch ecaresnet50t #--cuda

N_CPU=4
echo -e "\nPerformance with ${N_CPU} CPUs"
export OMP_NUM_THREADS=${N_CPU}
taskset -c 0-$[${N_CPU}-1] ./export/export_onnx.py --n_threads ${N_CPU} --sequential \
  --input_size 640 --arch ecaresnet50t #--cuda
