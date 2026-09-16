# Assignment 3: Performance Tools

**Due: October 28, 2025 @ 11:59 PM Eastern Time**

The purpose of this programming assignment is to gain experience in using
performance analysis tools for parallel programs. There are two parts to this assignment. In Part I, you will
run an existing parallel code, [LULESH](https://computing.llnl.gov/projects/co-design/lulesh) and
collect performance data using HPCToolkit. In Part II, you will use performance data (gathered using another tool called Caliper) provided to you, and analyze this data using Hatchet.

## Part I: Recording performance data

### Downloading and building LULESH

You can get LULESH by cloning its [git repository](https://github.com/LLNL/LULESH) as follows:

```bash
git clone https://github.com/LLNL/LULESH.git
```

For this assignment, we will use an older version of gcc (9.4.0) and openmpi. You can get this by doing: `module load openmpi/gcc/9.4.0`. If you get an error, unload `openmpi/gcc/11.3.0` that you might have loaded for Assignment 2.

You can use CMake to build LULESH on zaratan by following these steps:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_CXX_FLAGS="-g -O3" -DMPI_CXX_COMPILER=`which mpicxx` -DWITH_OPENMP=Off -DWITH_SILO=Off ..
make
```

This should produce an executable `lulesh2.0` in the build directory.

### Running LULESH

Lets say you want to run LULESH on 8 processes for 10 iterations/timesteps. This would be the mpirun line:

```bash
mpirun -np 8 ./lulesh2.0 -i 10 -p
```

> Reminder: login nodes are for code development and compilation only. Any runs including launching your parallel program using mpirun should be done in a batch job or interactive job.

### Using HPCToolkit

[HPCToolkit](http://www.hpctoolkit.org/manual/HPCToolkit-users-manual.pdf) can be loaded on zaratan via the `hpctoolkit/gcc/9.4.0` module. You can use HPCtoolkit to collect profiling data for a parallel program in three steps.

1. Step I: Running the code (LULESH) with hpcrun (**in a batch job**):

   `mpirun -np <num_ranks> hpcrun ./exe <args>`

   This will generate a measurements directory.
2. Step II: First post-processing step on the measurements directory (this can be run on the login node)

   `hpcstruct <measurements_directory>`
3. Step III: Second post-processing step on the measurements directory (this can be run on the login node)

   `hpcprof <measurements-directory>`

   This will generate a database directory.

You can use hpcviewer or [Hatchet](https://hatchet.readthedocs.io/en/latest/) with `from_hpctoolkit` to analyze the database directory generated in Step III.

## Part II: Analyzing performance data using Hatchet

### Installing Hatchet

You can install hatchet on zaratan or your local computer using pip (after loading the module for Python 3.10):

```bash
module load python/3.10.10/gcc
pip install hatchet
```

The above command should successfully install hatchet, provided that it runs without any errors. On zaratan, you may have to add `--user` to the pip command.

In case this does not work OR you want to install from source, then you can do so by following the steps below. First clone the hatchet git repository and add the path of the directory where you cloned hatchet to your `PYTHONPATH`:

```bash
git clone https://github.com/hatchet/hatchet
git checkout v1.4.1
export PYTHONPATH=<path-to-hatchet>:$PYTHONPATH
```

Next you can install the requirements using pip and then run install.sh inside the hatchet directory:

```bash
cd hatchet/
module load python/3.10.10/gcc
pip install -r requirements.txt
source install.sh
```

### Datasets

You can find four datasets from four different runs of LULESH at: [lulesh-1core.json](https://www.cs.umd.edu/class/fall2025/cmsc416/assignments/assign3/lulesh-1core.json) [lulesh-8cores.json](https://www.cs.umd.edu/class/fall2025/cmsc416/assignments/assign3/lulesh-8cores.json) [lulesh-27cores.json](https://www.cs.umd.edu/class/fall2025/cmsc416/assignments/assign3/lulesh-27cores.json) [lulesh-64cores.json](https://www.cs.umd.edu/class/fall2025/cmsc416/assignments/assign3/lulesh-64cores.json). These were gathered by running LULESH with 1, 8, 27, and 64 processes respectively. These profiles were gathered using Caliper, hence you can use `from_caliper` in the hatchet API to read them. You will use these profiles/datasets for all the tasks below.

### Analysis Tasks

> - For each of these tasks below, you should work with the assumption that the data files above are in the same current working directory as your Python scripts.
> - The datasets have multiples nodes in the tree that represent calls to the same MPI routines such as MPI_Recv, MPI_Wait etc. The answers will be different if you first merge such nodes (using a groupby_aggregate operation) versus if you do not. We do NOT want you to use groupby in any of the problems.
> - When printing the function name and time, only print the name and time as shown below with a space in between. Do NOT print the index column in the dataframe. The output below is just an example, not the actual output.

1. Problem 1: Analyze lulesh-1core.json and identify the top N
functions where the code spends the largest amounts of (exclusive) time. You
should set the value of N in your Python program from the first command line
argument to your script. The only output from the Python script should be N
lines, where each line prints the function name, followed by a space, and then
time (exclusive) spent in that function. What we will run to check
correctness (an example):

   `./problem1.py 3`

   Sample output below (this is what the output should look like, not the actual output/correct answer):

   ```
   func1_name 0.574
   func2_name 0.522
   func3_name 0.374
   ```

2. Problem 2: Use the `load_imbalance()` function on
the lulesh-64cores.json dataset, and given a command line parameter, X (value of X≥1),
identify the function that is X from the top (top starts at 1 and not 0) if the functions are sorted in
decreasing order of the imbalance. You should set the value of X in your Python
program from the first command line argument to your script. The only output
from the script should be the processes list generated by hatchet for this
function that have the most imbalance. What we will run to check
correctness (an example):

   `./problem2.py 6`

   Sample output below (this is what the output should look like, not the actual output/correct answer):

   ```
   [60 15  0 51  3]
   ```

3. Problem 3: Create two graphframes for the 8 and 64 process
case, use `drop_index_levels()` on both, and then subtract the
8-process graphframe from the 64-process one. Identify N functions with the
largest time differences (**NOT** the [absolute magnitude](https://en.wikipedia.org/wiki/Absolute_value)) between the two runs where N is a command line
argument. You should set the value of N in your Python program from the first
command line argument to your script. The only output from the Python script
should be N lines, where each line prints the function name, followed by a
space, and then different in time (exclusive) between the two runs for that
function. What we will run to check
correctness (an example):

   `./problem3.py 4`

   Sample output below (this is what the output should look like, not the actual output/correct answer):

   ```
   func1_name 0.574
   func2_name 0.552
   func3_name 0.522
   func4_name 0.374
   ```

> - Don't follow the build and running instructions in this assignment blindly. The goal is for you to learn to compile and run parallel code, and learn how to use HPCToolkit and Hatchet.
> - If you use the filter function in hatchet, add this argument to it: num_procs=1.

## What to Submit

You must submit the following files and no other files:

- Database directory generated on 8 processes for Part 1 renamed to lulesh-8processes.
- Python scripts that use hatchet for the analyses: problem1.py, problem2.py, and problem3.py.
- A report (called `report-assign3.pdf`) that describes what you did, and which hatchet functions you found to be the most useful?

You should put the dataset, 3 Python scripts, and report in a single directory (named `LastName-FirstName-assign3`), compress it to .tar.gz (`LastName-FirstName-assign3.tar.gz`) and upload that to [gradescope](https://www.gradescope.com/courses/1367315).
Replace `LastName` and `FirstName` with your last and first name, respectively.

> - We will not accept .ipynb files -- if you use a Python notebook, you should export to .py files, make sure they work correctly in a terminal and then submit the .py files
> - Do not submit the dataset files in the tarball

## Tips

- Helpful resources: [HPCToolkit user manual](http://www.hpctoolkit.org/manual/HPCToolkit-users-manual.pdf) and [Hatchet User Guide](https://hatchet.readthedocs.io/en/latest/user_guide.html)
- If you have questions about using these tools or Python and pandas, try using [Google](http://google.com) first.

## Grading

The project will be graded as follows:

| Component | Percentage |
| --------- | ---------- |
| Successful data collection | 30 |
| Problem 1 correctness | 20 |
| Problem 2 correctness | 20 |
| Problem 3 correctness | 20 |
| Writeup | 10 |
