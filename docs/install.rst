************
Installation
************

Requirements
============

cygrid has the following strict requirements:

- `Python <http://www.python.org/>`__ 3.8 or later

- `setuptools <https://setuptools.readthedocs.io/en/latest/>`__: Used for the package
  installation.

- `Cython <http://cython.org/>`__ 0.29 or later

- `NumPy <http://www.numpy.org/>`__ 1.18.1 or later

- `astropy <http://www.astropy.org/>`__: 4.0

- `pytest <https://pypi.python.org/pypi/pytest>`__ 6.0 or later


Installing cygrid
==================

Using Anaconda
--------------
The easiest way to install `~cygrid` is certainly to make use of the
great `Anaconda Python distribution <https://www.anaconda.com/>`_:

.. code-block:: bash

    conda install cygrid -c conda-forge



Using pip
-------------

To install cygrid with `pip <http://www.pip-installer.org/en/latest/>`__, simply run

.. code-block:: bash

    pip install cygrid

.. note::

    You may need a C++ compiler (e.g., ``g++``) with OpenMP support to be
    installed for the installation to succeed, if no `binary wheel
    <https://pythonwheels.com/>`_ is available for your OS and Python version
    on the `Python Package Index <https://pypi.org/project/cygrid/#files>`_


.. note::

    Use the ``--no-deps`` flag if you already have dependency packages
    installed, since otherwise pip will sometimes try to "help" you
    by upgrading your installation, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <https://pip.pypa.io/en/stable/user_guide/#user-installs>`__.

    We recommend to use a Python distribution, such as `Anaconda
    <https://www.continuum.io/downloads>`_, especially, if you are on
    :ref:`windows_install`.

    Do **not** install cygrid or other third-party packages using ``sudo``
    unless you are fully aware of the risks.

.. _source_install:

Installation from source
------------------------

There are two options, if you want to build cygrid from sources. Either, you
install the tar-ball (`*.tar.gz` file) from `PyPI
<https://pypi.python.org/pypi/cygrid>`_ and extract it to the directory of
your choice, or, if you always want to stay up-to-date, clone the git
repository:

.. code-block:: bash

    git clone https://github.com/bwinkel/cygrid

Then go into the cygrid source directory and run:

.. code-block:: bash

    python -m pip install .

Again, consider the ``--user`` option or even better use a python distribution
such as `Anaconda <https://www.continuum.io/downloads>`_ to avoid messing up
the system-wide Python installation.


.. _windows_install:

Installation on Windows
-----------------------

Note that for Windows machines we provide binary wheels via `PyPI`_ and installation is as easy as with Linux:

.. code-block:: bash

    pip install cygrid

.. note::

    If you are desperate, you can install cygrid from source even on Windows.
    You'll need to install a suitable C-compiler; `see here
    <https://wiki.python.org/moin/WindowsCompilers>`__. The cygrid
    package needs Python 3.8 or later, which means VC++ Version 14 is
    mandatory.


.. _macos_install:

Installation on MacOS
---------------------

Installation on MacOS can be a bit tricky, because the standard C compiler
does not support OpenMP. We provide wheels on PyPI, such that you can

.. code-block:: bash

    pip install cygrid

however, depending on the C++ compiler used on your system you may still
get into trouble. We can't provide support for this.

.. _testing_installed_cygrid:

Testing an installed cygrid
----------------------------

The easiest way to test if your installed version of cygrid is running
correctly, is to use the `~cygrid.test()` function::

    import cygrid
    cygrid.test()

The tests should run and print out any failures, which you can report at
the `cygrid issue tracker <http://github.com/bwinkel/cygrid/issues>`__.

.. note::

    This way of running the tests may not work if you do it in the
    cygrid source distribution directory.


If you prefer testing on the command line and usually work with the source
code, you can also do

.. code-block:: bash

    python setup.py test
