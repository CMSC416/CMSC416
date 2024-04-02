#include <stdlib.h>
#include <cmath>
#include <vector>
#include <limits>
#include <cstdio>

#define POINTS_MIN  1.0
#define POINTS_MAX  1000.0

const size_t TEST_SIZE = 1024;

struct Point {
    double x, y;
};

double getDistance(Point const& p1, Point const& p2) {
    return std::sqrt(std::pow(p2.x-p1.x, 2) + std::pow(p2.y-p1.y, 2));
}

/* Return the distance between the closest two points in the vector points.
   Example:

   input: [{2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}]
   output: 1.41421
*/
double closestPair(std::vector<Point> const& points) {
    // The polygon needs to have at least two points
    if (points.size() < 2)   {
        return 0;
    }

    double min_dist = std::numeric_limits<double>::max();

    for (int i = 0; i < points.size(); i++) {
        for (int j = i+1; j < points.size(); j++) {
            double dist = getDistance(points[i], points[j]);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
    }

    return min_dist;
}

int main(int argc, char **argv) {
    std::vector<Point> points(TEST_SIZE);
    srand(17);

    for (size_t i = 0; i < points.size(); i++) {
        points[i].x = (rand() / (double) RAND_MAX) * (POINTS_MAX - POINTS_MIN) + POINTS_MIN;
        points[i].y = (rand() / (double) RAND_MAX) * (POINTS_MAX - POINTS_MIN) + POINTS_MIN;
    }

    // double totalTime = 0.0;
    // double start = omp_get_wtime();

    double dist = closestPair(points);
    printf("Distance: %.5f\n", dist);

    // totalTime += omp_get_wtime() - start;
    // printf("Time: %.2f\n", totalTime);
}
