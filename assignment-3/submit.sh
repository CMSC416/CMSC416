#!/bin/bash

# Request 8 cores on a single node for 5 minutes
#SBATCH --cpus-per-task=8
#SBATCH -t 00:05:00
#SBATCH --exclusive
#SBATCH --mem-bind=local


# This is to suppress the warning about not finding a GPU resource
export OMPI_MCA_mpi_cuda_support=0

# Env variables to reduce performance variability
export OMP_PROCESSOR_BIND=true
export OMP_NUM_THREADS=8

# Run the executable
./quake-omp < quake.in > quake.out
