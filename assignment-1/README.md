# Assignment 1: OpenMP

The purpose of this programming assignment is to gain experience in parallel
programming on a cluster and OpenMP. You will start with working serial
versions of four different programs, and add OpenMP directives to parallelize
them. You should examine the loops in the code and figure out how to add OpenMP
directives. The programs will be run on a single compute node of zaratan.

There are four different serial programs that solve different problems:

1. [problem1.cpp](problem1.cpp): computes the distance between the closest two points given a vector of points
2. [problem2.cpp](problem2.cpp): computes the number of edges in a directed graph, stored in the adjacency matrix representation
3. [problem3.cpp](problem3.cpp): computes the product of elements in a vector with every odd-indexed element inverted
4. [problem4.cpp](problem4.cpp): computes the discrete fourier transform of a vector

Each problem takes two optional command line arguments (size and seed), is
deterministic, and can be run with different problem sizes, and seeds. A sample
Makefile can be found [here](Makefile).

## Using OpenMP

To compile OpenMP code, we will use gcc/g++ version 11.3.0 (the default version
on zaratan, which you can get by doing `module load gcc` on the zaratan login
node), which nicely has OpenMP support built in. In general, you can compile
problems in this assignment with:

```bash
g++ -fopenmp -O2 -o problem1 problem1.cpp
```

The `-fopenmp` tells the compiler to, you guessed it, recognize OpenMP
directives.

The environment variable **OMP_NUM_THREADS** sets the number of threads (and
presumably cores) that will run the program. Set the value of this environment
variable in the script you submit the job from. The value of this variable
defaults to using all available cores, and you might not want to do that
always.

A sample batch script can be found [here](submit.sh).

> Reminder: login nodes are for code development and compilation only. Any runs
> including executing/running your OpenMP programs should be done in a batch
> job or interactive job.

## Running and testing correctness of the problems

Each problem can run without any command line arguments to run a small test
problem. This can be used to test correctness of your modified program.  This
is how you can test correctness for the default case:

- Problems 1 through 3 print some value on standard out (redirected to a file
  if you use the provided batch script), which you can compare between the
  modified version and the original version (make two copies of the .cpp files,
  and redirect output from the two executables to different files).

- For problem 4, we want you to make a copy of the `dft` function in the file,
  call it `dft_omp`, modify `dft_omp` to use OpenMP, and then change line 61 to
  call `dft_omp` instead. Problem 4 will automatically print if your modified
  function is correct.

We also want you to ensure that your modified code runs correctly in a variety
of scenarios. So run the OpenMP versions with different OMP_NUM_THREADS=1, 2,
4, 8, 16, 32, 64 (this will require modifying the provided batch script). In
addition, try these command line arguments for the respective problems (these
should also be used for performance studies, more on that below):

- problem1: 16384
- problem2: 8192
- problem3: 67108864
- problem4: 8192

You can play with the optional second argument which changes the initial seed
used to generate random numbers for filling the vectors in each problem. This
can help with testing correctness.

## Studying performance of the problems

Finally, we want you to time the problems to study their performance when using
an increasing number of threads. To time your program, use `omp_get_wtime()` by
placing one call before the function call you want to time and another after
it. Sample timing code below:

```cpp
double totalTime = 0.0;

double start = omp_get_wtime();
... work to be timed ...
totalTime = omp_get_wtime() - start;

printf("TIME %.5f\n", totalTime);
```

Also, in order to minimize performance variability, you want to add these to
your job submission steps or script:

```bash
#SBATCH --mem-bind=local
#SBATCH --exclusive

export OMP_PROCESSOR_BIND=true
```

## What to Submit

You must submit the following files and no other files:

- A <2-page PDF report (called `report-assign1.pdf`) that describes what you did.
- In the same report, discuss the performance of running each problem with the larger sizes above (for 1, 2, 4, 8, 16, 32, and 64 threads on a single node).
- Modified problem1.cpp, problem2.cpp, problem3.cpp, and problem4.cpp files with OpenMP directives.
- A Makefile to compile all four problems.

You should put the code files, Makefile and report in a single directory (named
`LastName-FirstName-assign1`), compress it to .tar.gz
(`LastName-FirstName-assign1.tar.gz`) and upload that to
[gradescope](https://www.gradescope.com/courses/1367315).

> NOTE: Do not add any additional prints in your program. The timing prints
> should be used for studying performance but commented out before submitting
> your code.

## Tips

- [zaratan primer](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml)
- `omp_get_wtime()` [example](https://www.openmp.org/spec-html/5.0/openmpsu160.html)
- Use the compiler flag `-g` while debugging but `-O2` when collecting performance numbers for the report.
- Make sure that your batch script has the `--exclusive` flag when collecting execution times.

## Grading

The project will be graded as follows:

| Component | Percentage |
| --------- | ---------- |
| Problem 1 runs correctly on 4 and 16 threads | 10 + 10 |
| Problem 2 runs correctly on 4 and 16 threads | 10 + 10 |
| Problem 3 runs correctly on 4 and 16 threads | 10 + 10 |
| Problem 4 runs correctly on 4 and 16 threads | 15 + 15 |
| Writeup | 10 |
