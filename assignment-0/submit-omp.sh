#!/bin/bash

# Allocate 16 cores on a single node for 5 minutes
#SBATCH -N 1
#SBATCH --ntasks=16
#SBATCH -t 00:05:00
#SBATCH -A cmsc416-class

# This is to suppress the warning about not finding a GPU resource
export OMPI_MCA_mpi_cuda_support=0

# Set the value of OMP_NUM_THREADS
export OMP_NUM_THREADS=16

# Run the OpenMP executable
./cpi-omp &> my-openmp.out

