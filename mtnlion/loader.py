import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def save_npz_file(filename, data_dict, **kwargs):
    """
    Save data to an npz file. See numpy.savez for additional argument options.

    :param data_dict: data to be saved to an npz file
    :param filename: name of the npz file
    :param kwargs: additional numpy.savez arguments
    """
    logger.info('Saving data to npz: {}'.format(filename))
    np.savez(filename, **data_dict, **kwargs)


def load_numpy_file(filename, **kwargs):
    """
    Load data from an npz file. See numpy.load for additional argument options.

    :param filename: name of the npz file
    :param kwargs: additional numpy.load arguments
    """
    logger.info('Loading data from npz: {}'.format(filename))
    with np.load(filename, **kwargs) as data:
        return {k: v for k, v in data.items()}


def load_csv_file(filename, comments: str = '%', delimiter: str = ',', d_type=np.float64, **kwargs):
    """
    Load data from a csv file. See numpy.load for additional argument options.

    :param filename: name of the csv file
    :param comments: lines starting with a comment will be ignored
    :param delimiter: delimiting character(s)
    :param d_type: data type
    :param kwargs: additional numpy.loadtxt arguments
    :return:
    """
    return np.loadtxt(filename, comments=comments, delimiter=delimiter, dtype=d_type, **kwargs)


def format_name(name):
    return os.path.splitext(os.path.basename(name))[0]


def collect_files(file_list: List[str], format_key=None, loader=load_numpy_file):
    """
    Collect CSV data from list of filenames and create a dictionary of the data where the key is the basename of the
    file, and the data is a 2D ndarray, where the first column is the mesh, and the second is the data. Both are
    repeated for each new time step. Cannot read entire file names if they contain extra periods that do not proceed
    an extension. I.e. j.csv.bz2 or j.csv are okay, but my.file.csv is not.

    :param csv_file_list: list of files to read

    TODO: abstract out dimensionality requirement
    """
    logger.info('Collecting files: {}'.format(file_list))
    if not format_key:
        format_key = format_name

    data_dict = dict()
    for f in file_list:
        logger.debug('Reading "{variable}" from: {file}'.format(variable=format_key(f), file=f))
        data_dict[format_key(f)] = loader(f)

    # return {format_key(k): loader(k) for k in file_list}
    return data_dict
