#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o cuda-game-of-life-%A.out
#SBATCH -J cuda-game-of-life
#SBATCH -t 00:00:30

# load cuda libraries
module load cuda

# first problem
./game-of-life life.1.256x256.data 100 256 256

# second problem
./game-of-life life.2.256x256.data 100 256 256