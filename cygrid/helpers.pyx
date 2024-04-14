#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :helpers.pyx
# description            :Helper functions for cygrid.
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
# Note: Some HEALPix related helper functions are adapted from the
#   official Healpix_cxx (HEALPix C++) library.
#   (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke)
#   This was denoted in the docstrings accordingly.
#   For more information about HEALPix, see http://healpix.sourceforge.net
#   Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
#   and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
#   (DLR).
# ####################################################################

from cpython.version cimport PY_MAJOR_VERSION
from libc.time cimport time
from libcpp.vector cimport vector

__all__ = ['nside_to_order', 'npix_to_nside']


cdef unicode ustring(s):
    '''
    From cython docs: http://docs.cython.org/src/tutorial/strings.html
    '''

    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        # only accept byte strings in Python 2.x, not in Py3
        return (<bytes>s).decode('ascii')
    elif isinstance(s, unicode):
        # an evil cast to <unicode> might work here in some(!) cases,
        # depending on what the further processing does.  to be safe,
        # we can always create a copy instead
        return unicode(s)
    else:
        raise TypeError(
            'Could not convert object {} to unicode'.format(str(s))
            )


cdef uint64_t ilog2(uint64_t arg) noexcept nogil:
    '''
    Compute integer base-2 logarithm.

    Parameters
    ----------
    arg : 64-bit unsigned int
        The argument to take the log2 of.

    Notes
    -----
    1. This function was adapted from the HEALPix C++ library.
       (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
       see http://healpix.sourceforge.net)

    2. Cython-only to allow GIL-releasing.
    '''

    cdef uint64_t r = 0

    while (arg > 0x0000FFFF):
        r += 16
        arg >>= 16
    if arg > 0x000000FF:
        r |= 8
        arg >>= 8
    if arg > 0x0000000F:
        r |= 4
        arg >>= 4
    if arg > 0x00000003:
        r |= 2
        arg >>= 2
    if arg > 0x00000001:
        r |= 1
        arg >>= 1
    return r


cdef uint64_t isqrt(uint64_t arg) noexcept nogil:
    '''
    Compute integer square root.

    Parameters
    ----------
    arg : 64-bit unsigned int
        The argument to take the sqrt of.

    Notes
    -----
    1. This function was adapted from the HEALPix C++ library.
       (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
       see http://healpix.sourceforge.net)

    2. Cython-only to allow GIL-releasing.
    '''

    return (<uint64_t> sqrt(arg + 0.5))


cdef int64_t imod(int64_t a, int64_t b) noexcept nogil:
    '''
    Integer modulo for C.

    The %-operator in C can yield negative values, which is unpleasant.
    '''

    cdef int64_t r = a % b
    if r < 0:
        return r + b
    else:
        return r


cdef double fmodulo(double v1, double v2) noexcept nogil:
    '''
    Compute remainder of division v1 / v2.

    Parameters
    ----------
    v1, v2 : double
        v1 can be positive or negative; v2 must be positive.

    Notes
    -----
    1. This function was adapted from the HEALPix C++ library.
     (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
     see http://healpix.sourceforge.net)

    2. Cython-only to allow GIL-releasing.
    '''

    cdef double tmp

    if v1 >= 0:
        if v1 < v2:
            return v1
        else:
            return fmod(v1, v2)

    tmp = fmod(v1, v2) + v2
    if tmp == v2:
        return 0.
    else:
        return tmp


cpdef uint64_t nside_to_order(uint64_t nside):
    '''
    Compute HEALPix `order` from `nside`.

    Parameters
    ----------
    nside : uint64_t
        HEALPix `nside` parameter.

    Raises
    ------
    ValueError
        Invalid value for `nside` (must be positive).

    Notes
    -----
    This function was adapted from the HEALPix C++ library.
    (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
    see http://healpix.sourceforge.net)
    '''

    if nside <= 0:
        raise ValueError('Invalid value for nside (must be positive).')
    if (nside & (nside-1)):
        return -1
    else:
        return ilog2(nside)


cpdef uint64_t npix_to_nside(uint64_t npix):
    '''
    Compute HEALPix `nside` from `npix`.

    Parameters
    ----------
    npix : uint64_t
        HEALPix `npix` parameter.

    Raises
    ------
    ValueError
        Invalid value for `npix`.

    Notes
    -----
    This function was adapted from the HEALPix C++ library.
    (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
    see http://healpix.sourceforge.net)
    '''

    cdef uint64_t res = isqrt(npix // 12)
    if npix != res * res * 12:
        raise ValueError('Invalid value for npix.')
    return (<uint64_t> res)


cdef double true_angular_distance(
        double l1, double b1, double l2, double b2
        ) noexcept nogil:
    '''
    Calculate true angular distance between two points on the sphere.

    Parameters
    ----------
    l1, b1; l2, b2 : double
        Longitude and latitude (in rad) of two points on the sphere.

    Notes
    -----
    1. Based on Haversine formula. Good accuracy for distances < 2*pi
       See http://en.wikipedia.org/wiki/Haversine_formula.

    2. Cython-only to allow GIL-releasing.
    '''

    return 2 * asin(sqrt(
        sin((b1 - b2) / 2.) ** 2 +
        cos(b1) * cos(b2) * sin((l1 - l2) / 2.) ** 2
        ))


cdef double great_circle_bearing(
        double l1, double b1, double l2, double b2
        ) noexcept nogil:
    '''
    Calculate great circle bearing of (l2, b2) w.r.t. (l1, b1).

    Parameters
    ----------
    l1, b1; l2, b2 : double
        Longitude and latitude (in rad) of two points on the sphere.

    Notes
    -----
    1. Based on Haversine formula. Good accuracy for distances < 2*pi
       See http://en.wikipedia.org/wiki/Haversine_formula.

    2. Cython-only to allow GIL-releasing.
    '''

    cdef:
        double l_diff_rad, cos_b2

    l_diff_rad = l2 - l1
    cos_b2 = cos(b2)

    return atan2(
        cos_b2 * sin(l_diff_rad),
        cos(b1) * sin(b2) - sin(b1) * cos_b2 * cos(l_diff_rad)
        )


cdef void fill_time_info(
        vector[TimingInfo] &timing_info,
        char* info, ParallelState parstate,
        ) noexcept nogil:

    cdef TimingInfo this_info

    this_info.info = info
    this_info.parstate = parstate
    this_info.curtime = <uint64_t> time(NULL)

    timing_info.push_back(this_info)


cdef void print_time_info(vector[TimingInfo] timing_info):

    cdef:
        uint64_t last_time, first_time = timing_info.at(0).curtime
        uint64_t i

    last_time = first_time
    # print('Timing info size', timing_info.size())
    for i in range(1, timing_info.size()):
        print('{:60s} {:6d} s ({:s})'.format(
            timing_info.at(i).info.decode('ascii'),
            timing_info.at(i).curtime - first_time
            if timing_info.at(i).parstate == P_TOTAL else
            timing_info.at(i).curtime - last_time,
            'Single'
            if timing_info.at(i).parstate == P_SINGLE else
            'Multi'
            if timing_info.at(i).parstate == P_MULTI else
            'Total',
            ))
        last_time = timing_info.at(i).curtime



