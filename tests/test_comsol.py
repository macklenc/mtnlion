#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `comsol` module."""

import os

import pytest
from click.testing import CliRunner

from mtnlion import comsol


# class test_ComsolData:
#     def test_collect_csv_data(self):
#         resources = 'reference/comsol_solution/'
#
#         # test singular files
#         c = comsol.ComsolData
#         for file in os.listdir(resources):
#             c.collect_csv_data([file])


# @pytest.fixture
# def response():
#     """Sample pytest fixture.
#
#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')
#
#
# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface(tmpdir_factory):
    """Test the CLI. Ensure return codes are as expected."""

    fn1 = tmpdir_factory.mktemp('data').join('test_cli.npz')

    csvlist = ['reference/comsol_solution/j.csv', 'reference/comsol_solution/ce.csv',
               'reference/comsol_solution/cse.csv', 'reference/comsol_solution/phie.csv',
               'reference/comsol_solution/phis.csv', 'reference/comsol_solution/v.csv', str(fn1)]

    runner = CliRunner()
    result = runner.invoke(comsol.main, [str(fn1), csvlist[0]])
    assert 0 == result.exit_code
    result = runner.invoke(comsol.main, [str(fn1)] + csvlist[0:5])
    assert 0 == result.exit_code
    result = runner.invoke(comsol.main, [str(fn1)] + csvlist[0:7])
    assert 0 == result.exit_code
    help_result = runner.invoke(comsol.main, ['--help'])
    assert 0 == help_result.exit_code

    result = runner.invoke(comsol.main, [str(fn1), csvlist[6]])
    assert 2 == result.exit_code


if __name__ == '__main__':
    pytest.main(args=['-s', os.path.abspath(__file__)])
