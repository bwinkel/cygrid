#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Allow cythonizing of our pyx files and provide custom compiler options.
'''

import os
from setuptools.extension import Extension
from extension_helpers import get_compiler, add_openmp_flags_if_available
import platform
import numpy as np

PYXDIR = os.path.relpath(os.path.dirname(__file__))
PYXFILES = [
    'cygrid.pyx', 'helpers.pyx', 'healpix.pyx', 'hphashtab.pyx', 'kernels.pyx'
    ]


def get_extensions():

    comp_args = {
        'extra_compile_args': ['-O3', '-std=c++11'],
        'language': 'c++',
        'libraries': ['m'],
        # 'include_dirs': ['numpy'],
        'include_dirs': [np.get_include()],
        }

    if platform.system().lower() == 'windows':

        comp_args = {
            'language': 'c++',
            # 'include_dirs': ['numpy'],
            'include_dirs': [np.get_include()],
            }

    elif 'darwin' in platform.system().lower():

        extra_compile_args = ['-stdlib=libc++', '-mmacosx-version-min=10.7']

        if 'clang' in get_compiler():
            extra_compile_args += ['-stdlib=libc++', ]

        comp_args['extra_compile_args'] = extra_compile_args

    ext_list = []
    for pyx in PYXFILES:
        ext = Extension(
            name='cygrid.{}'.format(pyx.replace('.pyx', '')),
            sources=[os.path.join(PYXDIR, pyx)],
            **comp_args
            )
        add_openmp_flags_if_available(ext)
        ext_list.append(ext)

    return ext_list
