#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include "cuda.h"
using namespace std;

/*  use this to set the block size of the kernel launches.
    CUDA kernels will be launched with block size blockDimSize by blockDimSize. */
constexpr int blockDimSize = 8;

/*  Your job is to write compute_on_gpu. It computes a single iteration of Game of Life.
    mesh is the grid to write data into. previous_mesh is the grid from the previous generation.
    Previous mesh has already been copied into mesh from last generation. You do not need to copy it again.
    X_limit and Y_limit are the problem size.
    This kernel is called with block size blockDimSize x blockDimSize
    and grid size gridSizeX x gridSizeY.
*/
__global__ void compute_on_gpu(int *mesh, int *previous_mesh, int X_limit, int Y_limit) {
    /* your code here */
}

/*
 *  Copies from src into padded dst. src is width X height and dst is (width
 *  + 2 * padding) X (height + 2 * padding). You want to copy src into the
 *  middle of dst, leaving the padding untouched.
 */
__global__ void padded_matrix_copy(int *dst, int *src, int width, int height, int padding) {
    /* your code here */
}

/* allocates a new grid on the gpu. exits on error. */
int *allocate_grid_on_device(int length) {
    /* your code here */

    return 0;
}

/* frees grid from gpu memory. exits on error */
void deallocate_grid_on_device(int *array) {
    /* your code here */
}

/* copy grid from cpu to gpu. exits on error */
void copy_grid_to_device(int *host_array, int *device_array, int length) {
    /* your code here */
}

/* copy grid from gpu to cpu. exits on error */
void copy_grid_to_host(int *host_array, int *device_array, int length) {
    /* your code here */
}

/*
 * Reads the input file line by line and stores it in a 2D matrix.
 */
void read_input_file(int *mesh, string const &input_file_name, int Y_limit) {
    
    // Open the input file for reading.
    ifstream input_file;
    input_file.open(input_file_name);
    if (!input_file.is_open())
        perror("Input file cannot be opened");

    string line, val;
    int x, y;
    while (getline(input_file, line)) {
        stringstream ss(line);
        
        // Read x coordinate.
        getline(ss, val, ',');
        x = stoi(val);
        
        // Read y coordinate.
        getline(ss, val);
        y = stoi(val);

        // Populate the mesh matrix in column-major order.
        mesh[x*Y_limit + y] = 1;
    }
    input_file.close();
}

/* 
 * Writes out the final state of the 2D matrix to a csv file. 
 */
void write_output(int *result_matrix, int X_limit, int Y_limit,
                  string const &output_name, int num_of_generations) {
    
    // Open the output file for writing.
    ofstream output_file(output_name);
    if (!output_file.is_open())
        perror("Output file cannot be opened");
    
    // Output each live cell on a new line. 
    for (int i = 0; i < X_limit; i++) {
        for (int j = 0; j < Y_limit; j++) {
            if (result_matrix[i*Y_limit + j] == 1) {
                output_file << i << "," << j << "\n";
            }
        }
    }
    output_file.close();
}


/**
  * The main function to execute "Game of Life" simulations on a 2D board.
  */
int main(int argc, char *argv[]) {

    if (argc < 8)
        perror("Expected arguments: ./mesh <input_file> <num_of_generations> <X_limit> <Y_limit> <gridSizeX> <gridSizeY> <output_file>");

    string input_file_name = argv[1];
    int num_of_generations = stoi(argv[2]);
    int X_limit = stoi(argv[3]);
    int Y_limit = stoi(argv[4]);
    int gridSizeX = stoi(argv[5]);
    int gridSizeY = stoi(argv[6]);
    string output_file_name = argv[7];
    
    int *mesh = new int [X_limit*Y_limit];
    fill_n(mesh, X_limit*Y_limit, 0);

    // Use previous_mesh to track the pervious state of the board.
    // Pad the previous_mesh matrix with 0s on all four sides by setting all
    // cells in the following rows and columns to 0:
    //  1. Row 0
    //  2. Column 0
    //  3. Row X_limit+1
    //  4. Column Y_limit+1
    int *previous_mesh = new int[(X_limit+2)*(Y_limit+2)];
    fill_n(previous_mesh, (X_limit+2)*(Y_limit+2), 0);
    
    read_input_file(mesh, input_file_name, Y_limit);

    // allocate GPU data
    int *d_mesh = allocate_grid_on_device(X_limit*Y_limit);
    int *d_previous_mesh = allocate_grid_on_device((X_limit+2)*(Y_limit+2));

    // copy the grid and tmp grid onto GPU
    copy_grid_to_device(mesh, d_mesh, X_limit*Y_limit);
    copy_grid_to_device(previous_mesh, d_previous_mesh, (X_limit+2)*(Y_limit+2));

    dim3 blockSize (blockDimSize, blockDimSize);
    dim3 gridSize (gridSizeX, gridSizeY);

    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaEventRecord (start, 0);
    for (int numg = 0; numg < num_of_generations; numg++) {
        /* copy mesh into previous_mesh */
        padded_matrix_copy<<<gridSize, blockSize>>>(d_previous_mesh, d_mesh, X_limit, Y_limit, 1);

        compute_on_gpu<<<gridSize, blockSize>>>(d_mesh, d_previous_mesh, X_limit, Y_limit);
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
    copy_grid_to_host(mesh, d_mesh, X_limit*Y_limit);

    // Write out the final state to the output file.
    write_output(mesh, X_limit, Y_limit, output_file_name, num_of_generations);

    delete[] mesh;
    delete[] previous_mesh;
    deallocate_grid_on_device(d_mesh);
    deallocate_grid_on_device(d_previous_mesh);
    return 0;
}

#undef gpuErrchk
