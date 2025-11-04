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

num_sms=80

function run_jacobi_2d {
  x=$1
  y=$2
  iterations=$3
  input=${x}x${y}-input.csv
  output=${x}x${y}-output.csv
  grid_dims=$((32 * num_sms))
  echo "Running jacobi-2d with input: ${input}, output: ${output}, iterations: ${iterations}"
  ./jacobi-2d ${input} ${output} ${iterations} ${x} ${y} ${grid_dims} ${grid_dims}
}

INPUTS="32x32-input.csv 64x128-input.csv 128x128-input.csv"

run_jacobi_2d 32 32 100
run_jacobi_2d 64 128 100
run_jacobi_2d 128 256 100
