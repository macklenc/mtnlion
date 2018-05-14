#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `loader` module."""

import os
from typing import List, Union, Callable, Tuple

import _pytest
import numpy as np
import pytest

from mtnlion import loader


@pytest.fixture(scope='session')
def save_npz(tmpdir_factory: _pytest.tmpdir.TempdirFactory):
    filename = 'test.npz'
    fn1 = tmpdir_factory.mktemp('data').join(filename)
    data = {'test1': np.arange(0, 50), 'test2': np.arange(50, 100)}

    loader.save_npz_file(fn1, data)
    return str(fn1), data


@pytest.mark.order1
def test_save_npz_file(save_npz) -> None:
    """Test the CLI. Ensure return codes are as expected."""
    save_npz


@pytest.mark.order2
def test_load_numpy_file(save_npz) -> None:
    filename, data = save_npz
    result = loader.load_numpy_file(filename)

    for k, v in data.items():
        assert np.array_equal(v, result[k])

def test_load_csv_file():
    data = loader.load_csv_file('tests/reference/comsol_solution/v.csv.bz2')

    assert data.any()


def test_format_name():
    testname = 'tests/reference/comsol_solution/v.csv.bz2'
    result = loader.format_name(testname)

    assert 'v.csv' == result


def test_collect_files():
    filelist = ['test/a.ext', 'test/b.asd']

    file_loader = lambda x: np.arange(0,50)
    format_key = lambda x: x

    result = loader.collect_files(filelist, format_key, file_loader)

    for k in filelist:
        assert np.array_equal(np.arange(0, 50), result[k])


if __name__ == '__main__':
    pytest.main(args=['-s', os.path.abspath(__file__)])
