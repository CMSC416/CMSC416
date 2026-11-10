# Assignment 5: MPI+OpenMP (only for 616 students)

For this assignment, you will implement a hybrid MPI+OpenMP version of [Assignment 2](../assignment-2/README.md). There are three parts/steps to this assignment.

## Part I: Implementing a 2D decomposition in MPI

First you will implement a **2D decomposition** in MPI using non-blocking
send/receive routines. You will add two arguments to the command line at the
end that enable the user to provide the X and Y dimensions of the MPI virtual
grid. As an example, if we are running with 64 MPI processes, we should be able
to use an 8x8 or 16x4 or 4x16 virtual grid of processes. The modified command
line will like this:

```bash
mpirun -np <# of processes> ./life <data-file-name> <# of generations> <X_limit> <Y_limit> <# of processes in X> <# of processes in Y>
```

You can assume that X_limit and Y_limit will be powers of 2 as will be the
number of processes you will be running on. You can also assume that you will
be running the program on a minimum of 16 processes and X_limit and Y_limit are
much larger than the number of processes in each dimension.

> [!CAUTION]
> Login nodes are for code development and compilation only. Any runs including launching your parallel program using mpirun should be done in a batch job or interactive job.

## Part II: Using OpenMP in the compute loop

Next you will create a hybrid MPI+OpenMP version of the program implemented in
Part I by adding support for OpenMP in the sequential compute region of the
program. This is the code region that is executed sequentially by each MPI
process in the Part I implementation.

## Part III: Evaluating MPI processes vs. OpenMP threads

Finally, once you have a correctly working implementation from Part II, you
will study the impact of using a varying number of processes vs. threads on
performance.

You will use 2 nodes of zaratan for these studies. On one extreme, you can use
256 MPI processes (and 1 thread/process) on 2 nodes, and on the other extreme,
you can create 16 MPI processes, 8 on each node, and create 16 OpenMP threads
per MPI process. And you can do anything in between.

## What to Submit

You must submit the following files and no other files:

- `life-nonblocking-2d.[c,cpp,f77,f90]`: parallel version using non-blocking Isend/Irecv routines, where the file extension depends on the language used for the implementation
- `life-nonblocking-hybrid.[c,cpp,f77,f90]`: parallel version using non-blocking Isend/Irecv routines and OpenMP, where the file extension depends on the language used for the implementation
- `Makefile` that will compile your programs successfully on
zaratan when using mpicc or mpicxx.
Make sure that the executable
names are `life-nonblocking-2d` and `life-nonblocking-hybrid`, and do not include the executable in the tarball.
- You must also submit a report (pdf) with performance
results comparing processes vs. threads and your reasoning for the observations. The line plot should present the execution times to run
the parallel version on the input file [life.1024x1024.data](../assignment-2/data/life.1024x1024.data)
(a 1024x1024 board) for
16, 32, 64, 128, and 256 processes and the corresponding no. of OpenMP threads, running for 500 iterations.
In the report, you should mention:
  - how was the initial data distribution done
  - any considerations for implementing the hybrid version
  - what are the performance results, and are they what you expected

You should put the code, Makefile and report in a single directory (named
`LastName-FirstName-assign5`), compress it to .tar.gz
(`LastName-FirstName-assign5.tar.gz`) and upload that to gradescope.

> [!CAUTION]
> **AI tool disclosure:** If you choose to use an AI tool to assist in coding,
> you must disclose this information in the report. This disclosure should
> include the name of the AI tool and explain how it was used.

## Tips

- [Zaratan primer](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml)
- Use the compiler flag -g while debugging but -O2 when collecting performance numbers.
- MPI_Wtime [example](https://www.open-mpi.org/doc/v3.0/man3/MPI_Wtime.3.php)
- Use the --exclusive flag with sbatch when collecting execution times.

## Grading

The project will be graded as follows:

| Component | Percentage |
| --------- | ---------- |
| 2D MPI only version runs correctly | 40 |
| Hybrid version runs correctly | 30 |
| Performance evaluation and writeup | 30 |

> [!WARNING]
> NOTE: If your program does not compile when submitted on gradescope, you get
> 0 points. If your program does not run correctly, you do NOT get any points.
