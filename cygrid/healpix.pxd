#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :healpix.pxd
# description            :Healpix helper class.
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

from cpython cimport bool as python_bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set as cpp_set
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

from numpy cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
    uint32_t, uint64_t, float32_t, float64_t
    )

cdef class Healpix(object):

    cdef:
        uint32_t _scheme
        uint64_t _nside, _order, _nrings, _max_npix_per_ring
        uint64_t _npface, _ncap, _npix
        double _fact1, _fact2, _resolution, _omega
        python_bool _params_dirty

    cpdef _update_params(self)
    cpdef _on_nside_changed(self)
    cdef inline void _get_ring_info_small(
        self,
        uint64_t ring,
        uint64_t & startpix,
        uint64_t & num_pix_in_ring,
        bool & shifted,
        ) noexcept nogil
    cdef uint64_t _pix2ring(self, uint64_t pix) noexcept nogil
    cdef uint64_t _loc2pix(
        self,
        double z,
        double phi,
        double sin_theta,
        bool have_sin_theta
        ) noexcept nogil
    cdef uint64_t _ang2pix(self, double theta, double phi) noexcept nogil
    cdef void _pix2loc (
        self,
        uint64_t pix,
        double &z,
        double &phi,
        double &sin_theta,
        bool &have_sin_theta
        ) noexcept nogil
    cdef void _pix2ang(
        self, uint64_t pix, double & theta, double & phi
        ) noexcept nogil
    cdef vector[uint64_t] _query_disc_phi180(
        self, double disc_size_rad, uint64_t disc_ring
        ) noexcept nogil
    cdef vector[uint64_t] _query_disc(
        self, double theta, double phi, double disc_size_rad
        ) noexcept nogil
