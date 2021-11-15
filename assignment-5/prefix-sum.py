#!/usr/bin/env python

import numpy as np

if __name__ == "__main__":
    input_arr = np.random.randint(low=1, high=100, size = 131072)
    print(input_arr)

    with open("sample-input.data", "w") as infile:
        for num in np.nditer(input_arr):
            infile.write(str(num) + "\n")

    for i in range(1, len(input_arr)):
        input_arr[i] = input_arr[i] + input_arr[i-1]
    print(input_arr)

    with open("sample-output.data", "w") as outfile:
        for num in np.nditer(input_arr):
            outfile.write(str(num) + "\n")
