CC     = gcc
MPICC  = mpicc
CFLAGS = -O2

all: cpi-omp cpi-mpi

cpi-omp: cpi-omp.c
	$(CC) $(CFLAGS) -fopenmp -o $@ $<

cpi-mpi: cpi-mpi.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f cpi-omp cpi-mpi
