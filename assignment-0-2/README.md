# Assignment 0.2: Running an MPI example

The purpose of this programming assignment is to gain experience in running an
example MPI program on zaratan.  You will find this [MIT
course](https://missing.csail.mit.edu) and [command line
reference](https://www.cs.umd.edu/~mmarsh/books/cmdline/cmdline.html) useful.

> Reminder: login nodes are for code development and compilation only. Any runs
> including launching your parallel program should be done in a batch job or
> interactive job.

## Steps to Follow

- Download [cpi-mpi.c](cpi-mpi.c), an MPI program that calculates the value of
  Pi in parallel, to zaratan. You can either clone the git repository on
  zaratan or download the file locally to your laptop first and then `scp` to
  zaratan.
- Compile cpi-mpi.c using `mpicc`:

  ```bash
  mpicc -O2 -o cpi-mpi cpi-mpi.c
  ```

  Also, get familiar with using `make`. A sample Makefile is [here](Makefile).
  If `make` throws an error, you might need to load mpi first using: `module load
  openmpi/gcc`.
- Run the code by submitting a batch job using `sbatch` and a [batch
  script](submit-mpi.sh) on 1 and 16 processes. The batch script that is
  provided is hard-coded for a 16-process run.
- Run the code as an [interactive
  job](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml) using
  `sinteractive` on 1 and 16 processes. More details on that are on the [Zaratan
  quick primer](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml)
  page.

## What to Submit

You must submit the following files and no other files in a single tarball with
extension .tar.gz (delete the executable and any other files not mentioned
below before using `tar`):

- `cpi-mpi.c`
- `Makefile` that will compile your code successfully on zaratan when using mpicc or mpicxx.
- Output files from running the batch job on 1 and 16 processes. These files should be named `myfile-1.out` and `myfile-16.out` respectively. You only need to submit the output from the batch jobs and not the interactive jobs.

You should put the code, Makefile and output files in a single directory (named
`LastName-FirstName-assign0.2`), compress it to .tar.gz
(`LastName-FirstName-assign0.2.tar.gz`) and upload that to
[gradescope](https://www.gradescope.com/courses/1367315).

> Important things to check before submitting:
>
> - Remove unnecessary files (the executable, slurm-\*.out files, etc.) before creating the tarball.
> - When issuing the `tar` command, the current working directory (cwd) should be the parent directory of `LastName-FirstName-assign0.2`.
> - Do not issue the `tar` command from your top-level home directory or from within the assignment directory or anywhere else. When we use `untar` on your tarball, it should untar cleanly to create a single directory named `LastName-FirstName-assign0.2` with all the required files in it.

## Resources

- [Zaratan quick primer](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml)
- [Zaratan usage docs](https://hpcc.umd.edu/hpcc/help/usage.html)
- [The Missing Semester of Your CS Education](https://missing.csail.mit.edu)
- [Mike Marsh's Using the Bash Command Line book](https://www.cs.umd.edu/~mmarsh/books/cmdline/cmdline.html)
- [Mike Marsh's A General Systems Handbook](https://www.cs.umd.edu/~mmarsh/books/tools/tools.html)

## Grading

This assigment is for 0 points. However, all students are required to complete
the assignment. The autograder for this assignment will check that you created
the tarball correctly.
