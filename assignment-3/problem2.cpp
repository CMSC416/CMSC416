#include <stdlib.h>
#include <cmath>
#include <vector>
#include <cstdio>

const size_t TEST_SIZE = 128;

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
    std::vector<int> A(TEST_SIZE * TEST_SIZE);

    srand(17);

    std::fill(A.begin(), A.end(), 0);
    for (int i = 0; i < TEST_SIZE; i += 1) {
        for (int j = 0; j < TEST_SIZE; j += 1) {
            if (rand() % 2 == 0) {
                A[i * TEST_SIZE + j] = 1;
            }
        }
    }

    // double totalTime = 0.0;
    // double start = omp_get_wtime();

    int count = edgeCount(A, TEST_SIZE);
    printf("Count : %d\n", count);

    // totalTime += omp_get_wtime() - start;
    // printf("Time: %.2f\n", totalTime);
}
