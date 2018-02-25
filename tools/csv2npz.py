#!/usr/bin/python3

"""
Create a numpy zip (npz) with variables corresponding to the csv file names.
Each variable contains the data from the file as a list. Additionally, each
variable is a key in the main dictionary.
"""

import sys
import re
import numpy as np


def main():
    if len(sys.argv) < 3:
        print('{0} requires at least 2 arguments: {0} file.npz file1.csv file2.csv [...]'.format(sys.argv[0].split('/')
                                                                                                 [-1]))
        return 1

    params = dict()
    for fname in sys.argv[2:]:
        # create variable name from file name
        new_fname = re.sub(r'(.csv)', '', fname).split('/')[-1]
        # load the data into a dictionary with the correct key name
        params[new_fname] = np.loadtxt(fname, comments='%', delimiter=',')

    # store as compressed binary
    np.savez(sys.argv[1], **params)

    return 0


if __name__ == '__main__':
    sys.exit(main())
