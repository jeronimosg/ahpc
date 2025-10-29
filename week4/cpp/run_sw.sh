#!/bin/bash

# make sure stack size is unlimited
ulimit -s unlimited

# Startup CUDA Multi-Process Service
mkdir -p /home/jovyan/mps
export CUDA_MPS_LOG_DIRECTORY=/home/jovyan/mps
nvidia-cuda-mps-control -d

# Set current device to the correct value for MPS
export CUDA_VISIBLE_DEVICES=0

if [ "$#" -gt 2 ] || [ "$#" -eq 0 ]; then
  echo "Usage: $0 <SM> <fill>"
  echo "SM: number of streaming multiprocessors to run on. Has tp be 2, 4, 6, 8, 10, 12, or 14"
  echo "fill: 0 - just launch one copy, 1 - launch as many copies of sw_parallel as can fit on the GPU (floor(14/SMs))"
  # shutdown MPS service
  echo quit | nvidia-cuda-mps-control
  exit 1
fi
export SM=$1
export FILL=0
if [ "$#" -eq 2 ]; then
  export FILL=$2
fi

case $SM in

  2)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=15  #  2 SMs
    if [ "$FILL" -eq 0 ]; then
      ./sw_parallel
    else
      ./sw_parallel &
      ./sw_parallel &
      ./sw_parallel &
      ./sw_parallel &
      ./sw_parallel &
      ./sw_parallel &
      ./sw_parallel &
      wait
    fi;;

  4)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=29  #  4 SMs
    if [ "$FILL" -eq 0 ]; then
      ./sw_parallel
    else
      ./sw_parallel &
      ./sw_parallel &
      ./sw_parallel &
      wait
    fi;;
    
  6)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=43  #  6 SMs
    if [ "$FILL" -eq 0 ]; then
      ./sw_parallel
    else
      ./sw_parallel &
      ./sw_parallel &
      wait
    fi;;
    
  8)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=58  #  8 SMs
    ./sw_parallel;;
    
  10)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=72  # 10 SMs
    ./sw_parallel;;

  12)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=86  # 12 SMs
    ./sw_parallel;;

  14)
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 # 14 SMs
    ./sw_parallel;;

  *)
    echo "Unknown value for SM = $SM"
    echo "Valid values are 2, 4, 6, 8, 10, 12 or 14";;

esac

# shutdown MPS service
echo quit | nvidia-cuda-mps-control
