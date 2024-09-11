#include <stdlib.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>

/* Count the number of edges in the directed graph defined by the adjacency
   matrix A. A is an NxN adjacency matrix stored in row-major. A represents a
   directed graph.

   Example:
      input: [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]]
      output: 3
*/
int edgeCount(std::vector<int> const& A, size_t N) {
    int count = 0;

    for (size_t i = 0; i < N; i += 1) {
        for (size_t j = 0; j < N; j += 1) {
            if (A[i * N + j] == 1) {
                count++;
            }
        }
    }

    return count;
}

int main(int argc, char **argv) {
    int N = 128;
    int seed = 17;

    if (argc == 2) {
        N = std::stoi(argv[1]);
    }
    if (argc == 3) {
	    N = std::stoi(argv[1]);
	    seed = std::stoi(argv[2]);
    }

    std::vector<int> A(N * N);
    srand(seed);

    std::fill(A.begin(), A.end(), 0);
    for (int i = 0; i < N; i += 1) {
        for (int j = 0; j < N; j += 1) {
            if (rand() % 2 == 0) {
                A[i * N + j] = 1;
            }
        }
    }

    // double totalTime = 0.0;
    // double start = omp_get_wtime();

    int count = edgeCount(A, N);
    printf("Count : %d\n", count);

    // totalTime = omp_get_wtime() - start;
    // printf("Time: %.5f\n", totalTime);
}
