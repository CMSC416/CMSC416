/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "omp.h"
#include <stdio.h>
#include <math.h>

double f(double);

double f(double a)
{
    return (4.0 / (1.0 + a*a));
}

int main(int argc,char *argv[])
{
    int    n, i;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;
    double startwtime = 0.0, endwtime;

    n = 10000;          /* default # of rectangles */
    startwtime = omp_get_wtime(); 

    h   = 1.0 / (double) n;
    sum = 0.0;

    /* A slightly better approach starts from large i and works back */
    #pragma omp parallel for private(x) reduction(+:sum)
    for (i = 1; i <= n; i += 1)
    {
    x = h * ((double)i - 0.5);
    sum += f(x);
    }
    pi = h * sum;

    endwtime = omp_get_wtime(); 
    printf("pi is approximately %.16f, Error is %.16f\n",
           pi, fabs(pi - PI25DT));
    printf("wall clock time = %f\n", endwtime-startwtime);         
    fflush(stdout);

    return 0;
}
