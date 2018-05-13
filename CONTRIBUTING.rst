.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/macklenc/mtnlion/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Mountian Lion CSS could always use more documentation, whether as part of the
official Mountian Lion CSS docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/macklenc/mtnlion/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `mtnlion` for local development.

For the time being, Ubuntu is the easiest distribution to setup for development. The instructions here *may* be able to
be adapted for other distributions. To start off with, install ``python3-dolfin`` to get the python3 modules. I
recommend using ``virtualenvwrapper`` to setup an isolated virtual environment to work on:

1. Install `FEniCS <https://fenicsproject.org/download/>`_
2. Install `git-lfs <https://git-lfs.github.com/>`_
3. Fork the `mtnlion` repo on GitHub.
4. Clone your fork locally::

    $ git clone git@github.com:your_name_here/mtnlion.git

5. Install your local copy into a virtualenv. Assuming you have
   `virtualenvwrapper <https://virtualenvwrapper.readthedocs.io/en/latest/>`_ installed, this is how you set up your
   fork for local development::

    $ mkvirtualenv -p python3 --system-site-packages mtnlion
    $ cd mtnlion/
    $ echo "$(pwd)" > $WORKON_HOME/mtnlion/.project
    $ python setup.py devel

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::
   
    $ flake8 mtnlion tests
    $ python setup.py test or pytest
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
   - If a new module is added, make sure to run::
        
        $ sphinx-apidoc -F -o docs mtnlion --separate

     in order to generate new documentation for the modules. 
3. The pull request should work for Python 3.6, and for PyPy. Check
   https://travis-ci.org/macklenc/mtnlion/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

- To run a subset of tests::

    $ pytest tests.test_mtnlion

- To quickly setup ``virtualenvwrapper`` add these to your shell rc file::

    export WORKON_HOME=$HOME/.virtualenvs
    export PROJECT_HOME=$HOME/devel
    export VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3'
    source /usr/local/bin/virtualenvwrapper.sh

  and make sure that your clone of mtnlion is in ``$HOME/devel``.

- Use ``deactivate`` to leave the virtualenv, and verify that you are in the virtual env with ``which python`` which
  should point to a directory in ``$WORKON_HOME``.

- Use pycharm! To setup pycharm simply import mtnlion and go to settings ``Ctrl+Alt+S`` then go to
  ``Project: mtnlion -> Project Interpreter``, click on the gear and select ``add``. Select ``existing interpreter``,
  and the virtual environment in ``~/.virtualenvs`` should be auto-discovered. Choose that and exit all menu's
  selecting "OK".


Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.
