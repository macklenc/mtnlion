#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `comsol` module."""

import os
from typing import Callable, List, Union

from mtnlion import comsol
import numpy as np
import pytest


def test_fix_boundaries() -> None:
    """Test the removal of duplicate boundaries and the addition of missing boundaries."""
    boundaries = [4, 5, 6]
    test1_mesh = np.arange(0, 10)
    test2_mesh = np.insert(test1_mesh, [4, 5], [4, 5])

    test1_data = test1_mesh
    test1_expected = np.insert(test1_data, boundaries, boundaries)
    test2_data = np.arange(0, len(test2_mesh))
    test2_expected = np.arange(0, len(test2_mesh))
    test2_expected = np.insert(test2_expected, 8, 8)

    test1_result = comsol.fix_boundaries(test1_mesh, test1_data, boundaries)
    test2_result = comsol.fix_boundaries(test2_mesh, test2_data, boundaries)
    test3_result = comsol.fix_boundaries(test2_mesh, test2_data, [])

    assert 1 == np.array_equal(test1_result, test1_expected)
    assert 1 == np.array_equal(test2_result, test2_expected)
    assert 1 == np.array_equal(test3_result, test2_data)


@pytest.fixture()
def make_cell() -> Union[comsol.domain.ReferenceCell, Callable]:
    """
    Create a reference cell.

    :return: reference cell
    """
    # TODO: test pseudo mesh
    def cell(
        mesh: np.ndarray = None,
        pseudo_mesh: np.ndarray = None,
        time_mesh: np.ndarray = None,
        bound: List[float] = None,
        **kwargs: np.ndarray,
    ) -> comsol.domain.ReferenceCell:
        """
        Create a reference cell with default values.

        :param pseudo_mesh:
        :param mesh: space mesh
        :param time_mesh: time mesh
        :param bound: boundaries
        :param kwargs: data
        :return: reference cell
        """
        if mesh is None:
            mesh = np.array([0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3])
        if pseudo_mesh is None:
            pseudo_mesh = np.array([0, 0.5, 1, 1, 1.5, 1.5, 2, 2.5])
        if time_mesh is None:
            time_mesh = np.array([0, 1])
        if bound is None:
            bound = [1, 2]

        return comsol.domain.ReferenceCell(mesh, pseudo_mesh, time_mesh, bound, **kwargs)

    return cell


def test_remove_dup_boundary(make_cell: Union[comsol.domain.ReferenceCell, Callable]) -> None:
    """Test the removal of duplicate boundary values."""
    data = np.array([range(0, 9), range(9, 18)])
    expected = np.array([[0, 1, 2, 4, 6, 7, 8], [9, 10, 11, 13, 15, 16, 17]])

    cell = make_cell()

    result = comsol.remove_dup_boundary(cell, data)

    assert 1 == np.array_equal(expected, result)
    pass


# TODO: remove cs patch
def test_get_standardized(make_cell: Union[comsol.domain.ReferenceCell, Callable]) -> None:
    """Test that the correct reference cells are created."""
    data = np.array([range(0, 9), range(9, 18)])
    expected_data = np.array([[0, 1, 2, 4, 6, 7, 8], [9, 10, 11, 13, 15, 16, 17]])

    expected = make_cell(test1=expected_data, test2=expected_data, cs=expected_data)
    cell = make_cell(test1=data, test2=data, cs=expected_data)
    result = comsol.get_standardized(cell)

    assert 1 == np.array_equal(expected.data.test1, result.data.test1)
    assert 1 == np.array_equal(expected.data.test2, result.data.test2)


def test_format_data() -> None:
    """Test that the supplied COMSOL data is formatted correctly."""
    data_dict = dict()

    mesh1 = np.array([0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3])
    mesh2 = np.array([0, 0.5, 1, 1.5, 2, 2, 2.5, 3])
    mesh3 = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    time_mesh = np.array([0, 1])
    bound = [1, 2]

    data_dict["d1"] = np.array([mesh1, np.arange(0, len(mesh1))]).T
    data_dict["d2"] = np.array([mesh2, np.arange(0, len(mesh2))]).T
    data_dict["d3"] = np.array([mesh3, np.arange(0, len(mesh3))]).T
    data_dict["bad"] = np.array([1])
    data_dict["bad2"] = np.vstack((np.insert(mesh1, 4, 1), np.arange(0, len(mesh1) + 1))).T
    data_dict["mesh"] = mesh1
    data_dict["time_mesh"] = time_mesh

    result = comsol.format_2d_data(data_dict, bound)

    expected = {
        "d1": np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]]),
        "d2": np.array([[0, 1, 2, 2, 3, 4, 5, 6, 7]]),
        "d3": np.array([[0, 1, 2, 2, 3, 4, 4, 5, 6]]),
    }

    assert bound == result.pop("boundaries")
    assert np.array_equal(mesh1, result.pop("mesh"))
    assert np.array_equal(time_mesh, result.pop("time_mesh"))
    assert np.array_equal(expected["d1"], result["d1"])
    assert np.array_equal(expected["d2"], result["d2"])
    assert np.array_equal(expected["d3"], result["d3"])

    data_dict.pop("mesh")
    with pytest.raises(Exception):
        comsol.format_2d_data(data_dict, bound)


# TODO: move to different test file
# @pytest.fixture(scope='session')
# def run_full(tmpdir_factory: _pytest.tmpdir.TempdirFactory) \
#     -> Union[click.testing.Result, Tuple[click.testing.Result, str]]:
#     """
#     Run full program and save result
#     :param tmpdir_factory: directory to save to
#     :return: execution result and directory name
#     """
#     fn1 = tmpdir_factory.mktemp('data').join('test_cli.npz')
#
#     solutions = 'tests/reference/comsol_solution/'
#     csvlist = ['j.csv.bz2', 'ce.csv.bz2', 'cse.csv.bz2', 'phie.csv.bz2', 'phis.csv.bz2', 'v.csv.bz2', 'mesh.csv.bz2']
#     csvlist = [solutions + i for i in csvlist]
#     options = ['-t', '0', '50', '0.1']
#
#     runner = CliRunner()
#     result = runner.invoke(comsol.main, [str(fn1)] + csvlist + options)
#     return result, str(fn1)
#
#
# @pytest.mark.order1
# def test_command_line_interface(run_full: Union[click.testing.Result, Tuple[click.testing.Result, str]]) -> None:
#     """Test the CLI. Ensure return codes are as expected."""
#     import os
#     print(os.getcwd())
#     result, _ = run_full
#     assert 0 == result.exit_code
#
#
# @pytest.mark.order2
# def test_load(run_full: Union[click.testing.Result, Tuple[click.testing.Result, str]]) -> None:
#     _, file = run_full
#     cell = comsol.load(file)
#
#     assert cell.data


if __name__ == "__main__":
    pytest.main(args=["-s", os.path.abspath(__file__)])
