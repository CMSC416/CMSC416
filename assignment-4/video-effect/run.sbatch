#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o cuda-video-%A.out
#SBATCH -J cuda-video
#SBATCH -t 00:05:00

# load necessary libraries
module load cuda opencv

# modules doesn't add lib64 path to LD_LIBRARY_PATH for some reason
# do it manually
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/software/spack-software/2020.05.14/linux-rhel8-ivybridge/gcc-8.4.0/opencv-4.2.0-f7wcimpv42jmxqe22uuf5ghmsza6hvic/lib64"

# run video-effect
./video-effect video video-edge.mp4 edge 100 frame-100.csv
