#include <stdlib.h>
#include <cmath>
#include <vector>

#define POINTS_MIN  1.0
#define POINTS_MAX  1000.0

const size_t TEST_SIZE = 1024;

/* Return the product of the vector x with every odd indexed element inverted.
   i.e. x_0 * 1/x_1 * x_2 * 1/x_3 * x_4 ...
   Example:

   input: [4, 2, 10, 4, 5]
   output: 25
*/
double productWithInverses(std::vector<double> const& x) {
    double result = 1.0;

    for (int i = 0; i < x.size(); i++) {
        if (i % 2 == 1) {
            result *= 1 / x[i];
        } else {
            result *= x[i];
        }
    }

    return result;
}

int main(int argc, char **argv) {
    std::vector<double> x(TEST_SIZE);

    srand(273);

    for (int i = 0; i < TEST_SIZE; i += 1) {
        x[i] = (rand() / (double) RAND_MAX) * (POINTS_MAX - POINTS_MIN) + POINTS_MIN;
    }

    // double totalTime = 0.0;
    // double start = omp_get_wtime();

    double val = productWithInverses(x);
    printf("Product: %.5f\n", val);

    // totalTime += omp_get_wtime() - start;
    // printf("Time: %.2f\n", totalTime);
}
