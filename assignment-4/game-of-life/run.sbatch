#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_1g.5gb
#SBATCH -A cmsc416-aac
#SBATCH -o cuda-game-of-life-%A.out
#SBATCH -J cuda-game-of-life
#SBATCH -t 00:00:15

# load cuda libraries
module load cuda

# first problem
./game-of-life life.1.256x256.data 100 256 256 result1.csv

# second problem
./game-of-life life.2.256x256.data 100 256 256 result2.csv