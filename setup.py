#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# adapted from https://github.com/astropy/astropy


import sys


TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:
    tox -e test
If you don't already have tox installed, you can install it with:
    pip install tox
If you only want to run part of the test suite, you can also use pytest
directly with::
    python -m pip install --no-build-isolation --no-deps -v -v -v -e .
    python -m pytest -rsx --ignore-glob="*/setup_package.py" cygrid
    # or individual tests:
    python -m pytest -rsx --ignore-glob="*/setup_package.py" cygrid -k <test_func_name/module_name/etc.>
    # doc tests
    python -m pytest -rsx --doctest-plus --doctest-glob="*.rst" --doctest-ignore-import-errors docs

For more information, see:
  https://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if 'test' in sys.argv:
    print(TEST_HELP)
    sys.exit(1)


DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:
    tox -e build_docs
If you don't already have tox installed, you can install it with:
    pip install tox
You can also build the documentation with Sphinx directly using::
    python -m pip install --no-build-isolation --no-deps -v -v -v -e .
    cd docs
    # make clean  # to rebuild everything
    make html
    # alternatively (in project dir):
    sphinx-build docs docs/_build/html -b html
    sphinx-build docs docs/_build/html -b html -W  # fail on warnings

For more information, see:
  https://docs.astropy.org/en/latest/install.html#builddocs
"""

if 'build_docs' in sys.argv or 'build_sphinx' in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)


# Only import these if the above checks are okay
# to avoid masking the real problem with import error.
from setuptools import setup  # noqa
from extension_helpers import get_extensions  # noqa

# import numpy as np
# np.import_array()

setup(ext_modules=get_extensions())
