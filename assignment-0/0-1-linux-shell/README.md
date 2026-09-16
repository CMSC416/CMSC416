# Assignment 0.1: Using the Linux Shell on HPC Clusters

**Due: September 18, 2025 @ 11:59 PM Eastern Time**

The purpose of this programming assignment is to gain experience in using a
Linux shell and shell commands for tasks such as compiling a program,
editing source code, submitting a batch job, and submitting your assignments as
tarballs to gradescope. You will find this [MIT course](https://missing.csail.mit.edu) and [command line reference](https://www.cs.umd.edu/~mmarsh/books/cmdline/cmdline.html) useful.

> Reminder: login nodes are for code development and compilation only. Any runs including launching your parallel program should be done in a batch job or interactive job.

## Steps to Follow

- Download [cpi-openmp.c](cpi-omp.c), an OpenMP program that calculates the value of Pi in parallel, to zaratan. You can either clone the git repository on zaratan or download the file locally to your laptop first and then `scp` to zaratan.
- Compile cpi-omp.c using `gcc`:

  ```bash
  gcc -O2 -fopenmp -o cpi-omp cpi-omp.c
  ```

  Also, get familiar with using `make`. A sample Makefile is [here](Makefile). If you specify the openmp target by typing: `make cpi-omp`, you should not see any errors.
- Run the code by submitting a batch job using `sbatch` and a [batch script](submit-omp.sh). The batch script that is provided is hard-coded for a 16-thread run.
- Run the code as an [interactive job](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml) using `sinteractive`. More details on that are on the [Zaratan quick primer](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml) page.

## What to Submit

You must submit the following files and no other files in a single tarball with extension .tar.gz (delete the executable and any other files not mentioned below before using `tar`):

- `cpi-omp.c`
- `Makefile` that will compile your code successfully on zaratan when using gcc.
- Output file (`my-openmp.out`) from running the batch job. You only need to submit the output from the batch job and not the interactive job.

You should put the code, Makefile and output file in a single directory (named `LastName-FirstName-assign0.1`), compress it to .tar.gz (`LastName-FirstName-assign0.1.tar.gz`) using:

```bash
tar -cvzf <tarname>.tar.gz <dirname>
```

The tar command can be executed on zaratan, you can then scp the tarball to your laptop, and then
upload the tarball to [gradescope](https://www.gradescope.com/courses/1367315).

> Important things to check before submitting:
>
> - Remove unnecessary files (the executable, slurm-\*.out files, etc.) before creating the tarball.
> - When issuing the `tar` command, the current working directory (cwd) should be the parent directory of `LastName-FirstName-assign0.1`.
> - Do not issue the `tar` command from your top-level home directory or from within the assignment directory or anywhere else. When we use `untar` on your tarball, it should untar cleanly to create a single directory named `LastName-FirstName-assign0.1` with all the required files in it.

## Resources

- [Zaratan quick primer](https://www.cs.umd.edu/class/fall2026/cmsc416/zaratan.shtml)
- [HPC Knowledge Base and Documentation](https://hpcc.umd.edu/kb/)
- [The Missing Semester of Your CS Education](https://missing.csail.mit.edu)
- [Mike Marsh's Using the Bash Command Line book](https://www.cs.umd.edu/~mmarsh/books/cmdline/cmdline.html)
- [Mike Marsh's A General Systems Handbook](https://www.cs.umd.edu/~mmarsh/books/tools/tools.html)

## Grading

This assigment is for 0 points. However, all students are required
to complete the assignment. The autograder for this assignment will check that you created the tarball correctly.
