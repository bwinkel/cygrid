#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :helpers.pxd
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

from libc.math cimport cos, sin, sqrt, asin, fmod, atan2
from libcpp.vector cimport vector
from numpy cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
    uint32_t, uint64_t, float32_t, float64_t)
from libcpp cimport bool

cdef:
    unicode ustring(s)
    uint64_t ilog2(uint64_t arg) noexcept nogil
    uint64_t isqrt(uint64_t arg) noexcept nogil
    int64_t imod(int64_t a, int64_t b) noexcept nogil
    double fmodulo (double v1, double v2) noexcept nogil
    double true_angular_distance(
        double l1, double b1, double l2, double b2
        ) noexcept nogil
    double great_circle_bearing(
        double l1, double b1, double l2, double b2
        ) noexcept nogil

cpdef uint64_t nside_to_order(uint64_t nside)
cpdef uint64_t npix_to_nside(uint64_t npix)

cdef enum ParallelState:
    P_MULTI, P_SINGLE, P_TOTAL

cdef struct TimingInfo:
    char* info
    ParallelState parstate
    uint64_t curtime


# cdef void fill_time_info(
#     uint64_t code, uint64_t is_prange,
#     vector[vector[uint64_t]] &timing_info
#     ) noexcept nogil
# cdef void print_time_info(vector[vector[uint64_t]] timing_info)

cdef void fill_time_info(
    vector[TimingInfo] &timing_info,
    char* info, ParallelState parstate,
    ) noexcept nogil
cdef void print_time_info(vector[TimingInfo] timing_info)
