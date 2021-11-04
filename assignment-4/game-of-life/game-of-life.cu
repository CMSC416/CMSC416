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
    life is the grid to write data into. previous_life is the grid from the previous generation.
    Previous life has already been copied into life from last generation. You do not need to copy it again.
    X_limit and Y_limit are the problem size.
    This kernel is called with block size blockDimSize x blockDimSize
    and grid size gridSizeX x gridSizeY.
*/
__global__ void compute_on_gpu(int *life, int *previous_life, int X_limit, int Y_limit) {
    /* your code here */
}



/* allocates a new grid on the gpu. exits on error. */
int *allocate_grid_on_device(int length) {
    int *grid;
    cudaError_t status = cudaMalloc((void **)&grid, length*sizeof(int));
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not allocate memory on GPU.  Error code: %d\n", status);
        exit(status);
    }
    return grid;
}

/* frees grid from gpu memory. exits on error */
void deallocate_grid_on_device(int *array) {
    cudaError_t status = cudaFree(array);
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not deallocate memory on GPU.  Error code: %d\n", status);
    }
}

/* copy grid from cpu to gpu. exits on error */
void copy_grid_to_device(int *host_array, int *device_array, int length) {
    cudaError_t status = cudaMemcpy(device_array, host_array, length*sizeof(int), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not copy array from CPU to GPU.  Error code: %d\n", status);
        exit(status);
    }
}

/* copy grid from gpu to cpu. exits on error */
void copy_grid_to_host(int *host_array, int *device_array, int length) {
    cudaError_t status = cudaMemcpy(host_array, device_array, length*sizeof(int), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "Could not copy array from GPU to CPU.  Error code: %d\n", status);
        exit(status);
    }
}

/*
 * Reads the input file line by line and stores it in a 2D matrix.
 */
void read_input_file(int *life, string const &input_file_name, int Y_limit) {
    
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

        // Populate the life matrix in column-major order.
        life[x*Y_limit + y] = 1;
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

/*
    Copies from src into padded dst. src is widthxheight and dst is (width+padding)x(height+padding)
*/
__global__ void padded_matrix_copy(int *dst, int *src, int width, int height, int padding) {
    /* your code here */
}

/*
 * Processes the life array for the specified number of iterations.
 */
void compute(int *life, int *previous_life, int X_limit, int Y_limit) {
    int neighbors = 0;

    // Update the previous_life matrix with the current life matrix state.
    for (int i = 0; i < X_limit; i++) {
        for (int j = 0; j < Y_limit; j++) {
            previous_life[(i+1)*(Y_limit+2) + (j+1)] = life[i*Y_limit + j];
        }
    }

    // For simulating each generation, calculate the number of live
    // neighbors for each cell and then determine the state of the cell in
    // the next iteration.
    for (int i = 1; i < X_limit + 1; i++) {
        for (int j = 1; j < Y_limit + 1; j++) {
            neighbors = previous_life[(i-1)*(Y_limit+2) + (j-1)] + previous_life[(i-1)*(Y_limit+2) + (j)] +
                        previous_life[(i-1)*(Y_limit+2) + (j+1)] + previous_life[(i)*(Y_limit+2) + (j-1)] +
                        previous_life[(i)*(Y_limit+2) + (j+1)] + previous_life[(i+1)*(Y_limit+2) + (j-1)] +
                        previous_life[(i+1)*(Y_limit+2) + (j)] + previous_life[(i+1)*(Y_limit+2) + (j+1)];

            if (previous_life[i*(Y_limit+2) + j] == 0) {
                // A cell is born only when an unoccupied cell has 3 neighbors.
                if (neighbors == 3)
                    life[(i - 1)*Y_limit + (j - 1)] = 1;
            } else {
                // An occupied cell survives only if it has either 2 or 3 neighbors.
                // The cell dies out of loneliness if its neighbor count is 0 or 1.
                // The cell also dies of overpopulation if its neighbor count is 4-8.
                if (neighbors != 2 && neighbors != 3) {
                    life[(i - 1)*Y_limit + (j - 1)] = 0;
                }
            }
        }
    }
}

/**
  * The main function to execute "Game of Life" simulations on a 2D board.
  */
int main(int argc, char *argv[]) {

    if (argc < 8)
        perror("Expected arguments: ./life <input_file> <num_of_generations> <X_limit> <Y_limit> <gridSizeX> <gridSizeY> <output_file>");

    string input_file_name = argv[1];
    int num_of_generations = stoi(argv[2]);
    int X_limit = stoi(argv[3]);
    int Y_limit = stoi(argv[4]);
    int gridSizeX = stoi(argv[5]);
    int gridSizeY = stoi(argv[6]);
    string output_file_name = argv[7];
    
    int *life = new int [X_limit*Y_limit];
    fill_n(life, X_limit*Y_limit, 0);

    // Use previous_life to track the pervious state of the board.
    // Pad the previous_life matrix with 0s on all four sides by setting all
    // cells in the following rows and columns to 0:
    //  1. Row 0
    //  2. Column 0
    //  3. Row X_limit+1
    //  4. Column Y_limit+1
    int *previous_life = new int[(X_limit+2)*(Y_limit+2)];
    fill_n(previous_life, (X_limit+2)*(Y_limit+2), 0);
    
    read_input_file(life, input_file_name, Y_limit);

    // allocate GPU data
    int *d_life = allocate_grid_on_device(X_limit*Y_limit);
    int *d_previous_life = allocate_grid_on_device((X_limit+2)*(Y_limit+2));

    // copy the grid and tmp grid onto GPU
    copy_grid_to_device(life, d_life, X_limit*Y_limit);
    copy_grid_to_device(previous_life, d_previous_life, (X_limit+2)*(Y_limit+2));

    dim3 blockSize (blockDimSize, blockDimSize);
    dim3 gridSize (gridSizeX, gridSizeY);

    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaEventRecord (start, 0);
    for (int numg = 0; numg < num_of_generations; numg++) {
        /* copy life into previous_life */
        padded_matrix_copy<<<gridSize, blockSize>>>(d_previous_life, d_life, X_limit, Y_limit, 1);

        compute_on_gpu<<<gridSize, blockSize>>>(d_life, d_previous_life, X_limit, Y_limit);
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
    copy_grid_to_host(life, d_life, X_limit*Y_limit);

    // Write out the final state to the output file.
    write_output(life, X_limit, Y_limit, output_file_name, num_of_generations);

    delete[] life;
    delete[] previous_life;
    deallocate_grid_on_device(d_life);
    deallocate_grid_on_device(d_previous_life);
    return 0;
}

#undef gpuErrchk
