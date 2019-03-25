#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0',
                'numpy',
                'matplotlib',
                'munch',
                'scipy',
                'xlrd',
                'sympy', ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Christopher Macklen",
    author_email='cmacklen@uccs.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Mountain Lion Continuum-Scale Lithium-Ion Cell Simulator uses FEniCS to solve partial differential "
                "equations for the internal states of Lithium-Ion cells.",
    entry_points={
        'console_scripts': [
            'mtnlion=mtnlion.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mtnlion',
    name='mtnlion',
    packages=find_packages(include=['mtnlion']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/macklenc/mtnlion',
    version='0.0.1',
    zip_safe=False,
)
