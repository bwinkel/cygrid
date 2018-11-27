#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Allow cythonizing of our pyx files and provide custom compiler options.
'''

import os
from setuptools.extension import Extension
import platform
# Note: importing numpy from here won't work, see:
# http://docs.astropy.org/en/stable/development/ccython.html#using-numpy-c-headers
# import numpy as np
# 'include_dirs': [np.get_include()], --> 'include_dirs': 'numpy'

PYXDIR = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    comp_args = {
        'extra_compile_args': ['-fopenmp', '-O3', '-std=c++11'],
        'extra_link_args': ['-fopenmp'],
        'language': 'c++',
        'libraries': ['m'],
        'include_dirs': ['numpy'],
        }

    if platform.system().lower() == 'windows':
        comp_args['extra_compile_args'] = ['/openmp']
        del comp_args['extra_link_args']
    elif 'darwin' in platform.system().lower():
        # os.environ["CC"] = "g++-6"
        # os.environ["CPP"] = "cpp-6"
        # os.environ["CXX"] = "g++-6"
        # os.environ["LD"] = "gcc-6"
        comp_args['extra_compile_args'].append('-mmacosx-version-min=10.7')
        comp_args['extra_link_args'].append('-lgomp')

    ext_module_cygrid_cygrid = Extension(
        name='cygrid.cygrid',
        sources=[os.path.join(PYXDIR, 'cygrid.pyx')],
        **comp_args
        )

    ext_module_cygrid_helpers = Extension(
        name='cygrid.helpers',
        sources=[os.path.join(PYXDIR, 'helpers.pyx')],
        **comp_args
        )

    ext_module_cygrid_healpix = Extension(
        name='cygrid.healpix',
        sources=[os.path.join(PYXDIR, 'healpix.pyx')],
        **comp_args
        )

    ext_module_cygrid_hphashtab = Extension(
        name='cygrid.hphashtab',
        sources=[os.path.join(PYXDIR, 'hphashtab.pyx')],
        **comp_args
        )

    ext_module_cygrid_kernels = Extension(
        name='cygrid.kernels',
        sources=[os.path.join(PYXDIR, 'kernels.pyx')],
        **comp_args
        )

    # print('get_extensions', ext_module_cygrid_cygrid)
    return [
        ext_module_cygrid_cygrid,
        ext_module_cygrid_helpers,
        ext_module_cygrid_healpix,
        ext_module_cygrid_hphashtab,
        ext_module_cygrid_kernels
        ]
