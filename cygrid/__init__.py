# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

__minimum_python_version__ = "3.5"

class UnsupportedPythonError(Exception):
    pass

if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    raise UnsupportedPythonError("cygrid does not support Python < {}".format(__minimum_python_version__))

from .cygrid import *
from .healpix import *
from .hphashtab import *
from .helpers import *
from .mock import *
from .init_testrunner import *
from .version import version


__version__ = version
