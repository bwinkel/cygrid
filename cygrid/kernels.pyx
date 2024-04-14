#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :kernels.pyx
# description            :Grid-kernel definitions.
# author                 :Benjamin Winkel, Lars Flöer & Daniel Lenz
#
# ####################################################################
#  Copyright (C) 2010+ by Benjamin Winkel, Lars Flöer & Daniel Lenz
#  bwinkel@mpifr.de, mail@lfloeer.de, dlenz.bonn@gmail.com
#  This file is part of cygrid.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ####################################################################

from .constants cimport PI


__all__ = []


cdef double sinc(double x) noexcept nogil:
    '''
    Sinc function with simple singularity check.
    '''

    if fabs(x) < 1.e-10:
        return 1.
    else:
        return sin(x) / x


cdef double gaussian_1d_kernel(
        double distance, double bearing, double[::1] kernel_params
        ) noexcept nogil:
    '''
    Gaussian-1D kernel function.

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        unused - this is only in the call signature to allow function pointers
    kernel_params : double[::1]
        mem-view on kernel-parameters array
        must contain:
        kernel_params[0] : 1. / 2. / sigma_kernel ** 2

    Returns
    -------
    Kernel weight : double
    '''

    return exp(-distance * distance * kernel_params[0])


cdef double gaussian_2d_kernel(
        double distance, double bearing, double[::1] kernel_params
        ) noexcept nogil:
    '''
    Gaussian-2D kernel function.

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        unused - this is only in the call signature to allow function pointers
    kernel_params : double[::1]
        mem-view on kernel-parameters array
        must contain:
        kernel_params[0] : w_a
        kernel_params[1] : w_b
        kernel_params[2] : alpha


    Returns
    -------
    Kernel weight : double
    '''

    cdef:
        double ellarg, Earg

    ellarg = (
        kernel_params[0] ** 2 * sin(bearing - kernel_params[2]) ** 2 +
        kernel_params[1] ** 2 * cos(bearing - kernel_params[2]) ** 2
        )
    Earg = (distance / kernel_params[0] / kernel_params[1]) ** 2 / 2. * ellarg

    return exp(-Earg)


cdef double tapered_sinc_1d_kernel(
        double distance, double bearing, double[::1] kernel_params
        ) noexcept nogil:
    '''
    Kaiser-Bessel-1D kernel function (Gaussian-tapered sinc).

    Parameters
    ----------
    distance : double
        Radial distance/separation
    bearing : double
        unused - this is only in the call signature to allow function pointers
    kernel_params : double[::1]
        mem-view on kernel-parameters array
        must contain:
        kernel_params[0] : sigma_kernel
        kernel_params[1] : a (should be set to 2.52)
        kernel_params[2] : b (should be set to 1.55)

    Returns
    -------
    Kernel weight : double

    Notes: the sigma_kernel widths is defined in a manner compatible with
        gaussian_1d, i.e., the kernel-sphere radius should also be three times
        as large. See
            http://casa.nrao.edu/aips2_docs/glossary/g.html
        for details.
    '''

    cdef:
        double arg

    arg = PI * distance / kernel_params[0]

    return sinc(arg / kernel_params[2]) * exp(-(arg / kernel_params[1]) ** 2)
