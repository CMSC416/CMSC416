#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <assert.h>
#include "cuda.h"
using namespace std;

/*  use this to set the block size of the kernel launches.
    CUDA kernels will be launched with block size blockDimSize by blockDimSize. */
constexpr int blockDimSize = 8;

/* allocates a new grid on the gpu. exits on error. */
void **allocate_grid_on_device(int width, int length) {
    /* your code here */
}

/* frees grid from gpu memory. exits on error */
void deallocate_grid_on_device(double **matrix) {
    /* your code here */
}

/*  Your job is to write compute_on_gpu. It computes a single iteration of the 2D stencil.
    matrix is the grid to write data into. previous_matrix is the grid from the previous iteration.
    Previous matrix has already been copied into matrix from the last iteration. You do not need to copy it again.
    X_limit and Y_limit are the problem size.
    This kernel is called with block size blockDimSize x blockDimSize
    and grid size gridSizeX x gridSizeY.
*/
__global__ void compute_on_gpu(double **matrix, double **previous_matrix, int X_limit, int Y_limit) {
    /* your code here */
}

/*
 *  Copies from src into padded dst. src is width X height and dst is (width
 *  + 2 * padding) X (height + 2 * padding). You want to copy src into the
 *  middle of dst, leaving the padding untouched.
 */
__global__ void padded_matrix_copy(double **dst, double **src, int width, int height, int padding) {
    /* your code here */
}

/* copy grid from cpu to gpu. exits on error */
void copy_grid_to_device(double **host_matrix, double **device_matrix, int width, int length) {
    /* your code here */
}

/* copy grid from gpu to cpu. exits on error */
void copy_grid_to_host(double **host_matrix, double **device_matrix, int width, int length) {
    /* your code here */
}

/*
 * Reads the input file line by line and stores it in a 2D matrix.
 */
void read_input_file(double **matrix, string const &input_file_name,
    int X_limit, int Y_limit) {
    
    // Open the input file for reading.
    ifstream input_file;
    input_file.open(input_file_name);
    if (!input_file.is_open())
        perror("Input file cannot be opened");

    int r = 0;
    for (string line; getline(input_file, line) && r < length; ++r) {
        stringstream ss(line);
        int c = 0;
        for (string val; getline(ss, val, ',') && c < width; ++c) {
            //cout << "'" << val << "' ";
            matrix[r][c] = stod(val);
        }
        //cout << endl;
    }

    input_file.close();
}

/* 
 * Writes out the final state of the 2D matrix to a csv file. 
 */
void write_output(double **result_matrix, int X_limit, int Y_limit,
                  string const &output_name, int num_iterations) {
    
    // Open the output file for writing.
    ofstream output_file(output_name);
    if (!output_file.is_open())
        perror("Output file cannot be opened");

    output_file << fixed << setprecision(1);
    
    // Output each live cell on a new line. 
    for (int r = 0; r < length; r++) {
        for (int c = 0; c < width; c++) {
            output_file << result_matrix[r][c];
            if (c < width - 1) {
                output_file << ",";
            } else {
                output_file << "\n";
            }
        }
    }
    output_file.close();
}


/**
  * The main function to execute "Game of Life" simulations on a 2D board.
  */
int main(int argc, char *argv[]) {

    string input_file_name;
    string output_file_name;
    int num_iterations;
    int X_limit;
    int Y_limit;
    int gridSizeX;
    int gridSizeY;

    if (argc == 8) {
        input_file_name = argv[1];
        num_iterations = stoi(argv[2]);
        X_limit = stoi(argv[3]);
        Y_limit = stoi(argv[4]);
        gridSizeX = stoi(argv[5]);
        gridSizeY = stoi(argv[6]);
        output_file_name = argv[7];

    } else {
        perror("Expected arguments: ./matrix <input_file> <output_file> <num_iterations> [<X_limit> <Y_limit> <gridSizeX> <gridSizeY>]");
        return 2;
    }

    double *matrix = new double [X_limit * Y_limit];
    read_input_file(matrix, input_file_name, X_limit, Y_limit);

    // Use previous_matrix to track the pervious state of the board.
    // Pad the previous_matrix matrix with 0s on all four sides by setting all
    // cells in the following rows and columns to 0:
    //  1. Row 0
    //  2. Column 0
    //  3. Row X_limit+1
    //  4. Column Y_limit+1
    double **previous_matrix = new double[(X_limit+2) * (Y_limit+2)];

    // allocate GPU data
    double **d_matrix = allocate_grid_on_device(X_limit, Y_limit);
    double **d_previous_matrix = allocate_grid_on_device(X_limit+2, Y_limit+2);

    // copy the grid and tmp grid onto GPU
    copy_grid_to_device(matrix, d_matrix, X_limit, Y_limit);
    copy_grid_to_device(previous_matrix, d_previous_matrix, X_limit+2, Y_limit+2);

    dim3 blockSize (blockDimSize, blockDimSize);
    dim3 gridSize (gridSizeX, gridSizeY);

    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaEventRecord (start, 0);
    for (int numg = 0; numg < num_iterations; numg++) {
        /* copy matrix into previous_matrix */
        padded_matrix_copy<<<gridSize, blockSize>>>(d_previous_matrix, d_matrix, X_limit, Y_limit, 1);

        compute_on_gpu<<<gridSize, blockSize>>>(d_matrix, d_previous_matrix, X_limit, Y_limit);
    }
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    
    float elapsed;
    cudaEventElapsedTime (&elapsed, start, stop);
    cout << "Runtime: " << elapsed/1000.0 << " s\n";

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    // copy the results back onto the CPU
    cudaDeviceSynchronize();
    copy_grid_to_host(matrix, d_matrix, X_limit, Y_limit);

    // Write out the final state to the output file.
    write_output(matrix, X_limit, Y_limit, output_file_name, num_iterations);

    deallocate_grid_on_host(matrix);
    deallocate_grid_on_host(previous_matrix);
    deallocate_grid_on_device(d_matrix);
    deallocate_grid_on_device(d_previous_matrix);
    return 0;
}

#undef gpuErrchk
