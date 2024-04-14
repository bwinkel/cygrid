#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :hphashtab.pxd
# description            :Hash tables based on Healpix helper class.
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

from .healpix cimport Healpix
from .helpers cimport TimingInfo, P_SINGLE, P_MULTI, P_TOTAL


cdef class HpxHashTable(Healpix):

    cdef:
        # key: HPX index, values: lists of target pixel pairs (as integer hash)
        unordered_map[uint64_t, vector[uint64_t]] target_dict

        # key: HPX ring index
        unordered_map[uint64_t, uint64_t] ring_info_hash_startpix
        unordered_map[uint64_t, uint64_t] ring_info_hash_num_pix_in_ring
        unordered_map[uint64_t, bool] ring_info_hash_shifted

        # key: HPX ring index of disc center
        unordered_map[uint64_t, vector[uint64_t]] disc_hash_ring
        unordered_map[uint64_t, vector[uint64_t]] disc_hash_indices
        bint dbg_messages

    cpdef set_optimal_nside(self, double target_res)
    cpdef clear_hashes(self)
    cpdef _on_nside_changed(self)
    cdef prepare_helpers(
        self,
        double ksq,
        uint64_t[::1] xpix_fs,
        uint64_t[::1] ypix_fs,
        double[::1] xwcs_fs,
        double[::1] ywcs_fs,
        )
    cdef void calculate_output_pixels(
        self,
        double[::1] lons,
        double[::1] lats,
        double kernel_radius,
        unordered_map[uint64_t, vector[uint64_t]] & output_input_mapping,
        vector[uint64_t] & output_pixels,
        )
    cdef void _fill_ring_info_hashes(self) noexcept nogil
    cdef void _compute_target_hpx_pixels(
        self,
        uint64_t[::1] xpix_fs,
        uint64_t[::1] ypix_fs,
        double[::1] xwcs_fs,
        double[::1] ywcs_fs,
        ) noexcept nogil
    cdef void _fill_disc_hash(
        self,
        vector[uint64_t] urings,
        double disc_size_rad,
        ) noexcept nogil
    #cdef unordered_map[uint64_t, vector[uint64_t]] _compute_input_output_mapping(
        #self,
        #double[::1] lons,
        #double[::1] lats,
        #double disc_size_rad,
        #) noexcept nogil
    cdef void _compute_input_output_mapping(
        self,
        double[::1] lons,
        double[::1] lats,
        double disc_size_rad,
        vector[uint64_t] &intermediary_hpidxs_vec,
        unordered_map[uint64_t, vector[uint64_t]] &hpxidx_output_map,
        vector[TimingInfo] &timing_info,
        ) noexcept nogil
    cdef void _compute_output_input_mapping(
        self,
        vector[uint64_t] &intermediary_hpidxs_vec,
        unordered_map[uint64_t, vector[uint64_t]] & hpxidx_output_map,
        unordered_map[uint64_t, vector[uint64_t]] & output_input_map,
        vector[TimingInfo] &timing_info,
        ) noexcept nogil
    cdef void _get_output_pixels(
        self,
        unordered_map[uint64_t, vector[uint64_t]] & output_input_mapping,
        vector[uint64_t] & output_pixels,
        ) noexcept nogil

# cpdef dict test_varreduction(dict idict)
