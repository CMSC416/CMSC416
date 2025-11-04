#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:a100_1g.5gb
#SBATCH -A cmsc416-class
#SBATCH -o cuda-jacobi-2d-%A.out
#SBATCH -J cuda-jacobi-2d
#SBATCH -t 00:00:15

# load cuda libraries
module load cuda

INPUTS="32x32-input.csv 64x128-input.csv 128x128-input.csv"
for input in ${INPUTS} ; do
  output=${input/input/output}
  ./jacobi-2d ${input} 100 ${output}
done

