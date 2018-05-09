=================
Mountian Lion CSS
=================


.. image:: https://img.shields.io/pypi/v/mtnlion.svg
        :target: https://pypi.python.org/pypi/mtnlion

.. image:: https://img.shields.io/travis/macklenc/mtnlion.svg
        :target: https://travis-ci.org/macklenc/mtnlion

.. image:: https://readthedocs.org/projects/mtnlion/badge/?version=latest
        :target: https://mtnlion.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/macklenc/mtnlion/shield.svg
     :target: https://pyup.io/repos/github/macklenc/mtnlion/
     :alt: Updates



                Mountain Lion Continuum-Scale Lithium-Ion Cell Simulator uses FEniCS to solve partial differential equation models for lithium-ion cells.


* Free software: MIT license
* Documentation: https://mtnlion.readthedocs.io.


Features
--------

* TODO

Usage
-----
First, install `FEniCS <https://fenicsproject.org/download/>`_.

Development Environment
^^^^^^^^^^^^^^^^^^^^^^^
If using Ubuntu, install ``python3-dolfin`` to get the python3 modules. I recommend using ```virtualenvwrapper``` to setup an isolated virtual environment to work on:

.. code-block:: bash
    sudo pip3 install virtualenvwrapper

With ``virtualenvwrapper`` installed, add the following lines the the end of your shell rc file, i.e. ``.bashrc``:

.. code-block:: bash
    export WORKON_HOME=$HOME/.virtualenvs
    export PROJECT_HOME=$HOME/devel
    export VIRTUALENVWRAPPER_PYTHON='/usr/bin/python3'
    source /usr/local/bin/virtualenvwrapper.sh

This will set the location of the virtual python interpreter and related packages such as ``pip`` in ``~/.virtualenvs``
and set the location of the source code directories in ``~/devel``. This also tells ``virtualenvwrapper`` to use python3
in the virtual environment rather than Ubuntu's default python2. The final line adds the ``virtualenvwrapper`` adds the
wrapper commands to your environment. Next, clone mtnlion:

.. code-block:: bash
    git clone https://github.com/macklenc/mtnlion.git ~/devel

and setup your virtual environment:

.. code-block:: bash
    mkvirtualenv -p python3 --system-site-packages

What this does is create a new unadulterated copy of python3 in ``~/.virtualenvs`` and allows it to inherit the system
python packages, which is required to get access to ``python3-dolfin`` modules. Next run

.. code-block:: bash
    echo "~/devel/mtnlion" > ~/.virtualenvs/mtnlion/.project

to allow ``virtualenvwrapper`` to cd into mtnlion. Finally, use the ``virtualenvwrapper`` command ``workon mtnlion`` to
enter the virtual environment. You can confirm this worked by running ``which python``, if it points to python3 in the
``.virtualenvs`` folder, then all is well. To exit the virtual environment, use ``deactivate``. Finally install any
dependencies from the ``requirements_dev.txt`` and ``requirements.txt`` using:

.. code-block:: bash
    pip install -r requirements_dev.txt
    pip install -r requirements.txt

... then the environment should be good to go! I highly recommend using pycharm for your IDE, and there's an included
project file for that. Should you decide to use pycharm, after opening the project use ``ctl+alt+s`` to open the
settings and go to  ``Project: mtnlion -> Project Interpreter``, click on the gear and select ``add``. Select existing
interpreter, and the virtual environment in ``~/.virtualenvs`` should be auto-discovered. Choose that and exit all
menu's selecting "OK".

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
