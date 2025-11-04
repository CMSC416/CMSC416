#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_1g.5gb
#SBATCH -A cmsc416-class
#SBATCH -o cuda-2d-stencil-%A.out
#SBATCH -J cuda-2d-stencil
#SBATCH -t 00:00:15

# load cuda libraries
module load cuda

# first problem
./jacobi-2d 32x32-input.csv 32x32-output.csv 100 32 32 1 1

# second problem
./jacobi-2d 64x128-input.csv 64x128-output.csv 100 64 128 2 4

# third problem
./jacobi-2d 128x128-input.csv 128x128-output.csv 100 128 128 4 4
