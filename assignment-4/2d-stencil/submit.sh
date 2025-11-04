#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -A cmsc416-class
#SBATCH -o cuda-2d-stencil-%A.out
#SBATCH -J cuda-2d-stencil
#SBATCH -t 00:00:15

# load cuda libraries
module load cuda

# first problem
./2d-stencil 32x32-input.csv 100 32 32 1 1 32x32-output.csv

# second problem
./2d-stencil 64x128-input.csv 100 64 128 2 4 64x128-output.csv

# third problem
./2d-stencil 128x256-input.csv 100 128 128 4 4 128x128-output.csv
