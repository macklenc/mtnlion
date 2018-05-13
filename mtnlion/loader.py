"""
Utilities for loading/saving files in various formats.
"""
import logging
import os
from typing import List, Dict, Callable

import numpy as np

logger = logging.getLogger(__name__)


def save_npz_file(filename: str, data_dict: Dict[str, np.ndarray], **kwargs) -> None:
    """
    Save data to an npz file. See numpy.savez for additional argument options.

    :param data_dict: data to be saved to an npz file
    :param filename: name of the npz file
    :param kwargs: additional numpy.savez arguments
    """
    logger.info('Saving data to npz: {}'.format(filename))
    np.savez(filename, **data_dict, **kwargs)


def load_numpy_file(filename: str, **kwargs) -> Dict[str, np.ndarray]:
    """
    Load data from an npz file. See numpy.load for additional argument options.

    :param filename: name of the npz file
    :param kwargs: additional numpy.load arguments
    :return: data dictionary
    """
    logger.info('Loading data from npz: {}'.format(filename))
    with np.load(filename, **kwargs) as data:
        return {k: v for k, v in data.items()}


def load_csv_file(filename: str, comments: str = '%', delimiter: str = ',', d_type: type = np.float64, **kwargs) \
    -> np.ndarray:
    """
    Load data from a csv file. See numpy.load for additional argument options.

    :param filename: name of the csv file
    :param comments: lines starting with a comment will be ignored
    :param delimiter: delimiting character(s)
    :param d_type: data type
    :param kwargs: additional numpy.loadtxt arguments
    :return: file data
    """
    logger.info('Loading CSV file: {}'.format(filename))
    return np.loadtxt(filename, comments=comments, delimiter=delimiter, dtype=d_type, **kwargs)


def format_name(name: str) -> str:
    """
    Default function for formatting variable names from filenames
    :param name: filename
    :return: variable name
    """
    key = os.path.splitext(os.path.basename(name))[0]
    logger.info('Using key name: {}'.format(key))
    return key


def collect_files(file_list: List[str], format_key: Callable = format_name, loader: Callable = load_numpy_file,
                  **kwargs) \
    -> Dict[str, np.ndarray]:
    """
    Collect files given as a list of filenames using the function loader to load the file and the function format_key
    to format the variable name.
    :param file_list: list of filenames
    :param format_key: function to format variable names
    :param loader: function to load files
    :param kwargs: extra arguments to the loader
    :return: data dictionary
    """
    logger.info('Collecting files: {}'.format(file_list))

    data_dict = dict()
    for f in file_list:
        data_dict[format_key(f)] = loader(f, **kwargs)

    # return {format_key(k): loader(k) for k in file_list}
    return data_dict
