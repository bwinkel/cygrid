#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :hphashtab.pyx
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

# import python3 compat modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import std lib
import sys
import traceback

# import cython specifics
cimport cython
from cython.parallel import parallel, prange, threadid  # , threadsavailable
cimport openmp
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool as python_bool

# import C/C++ modules
from libc.math cimport exp, cos, sin, sqrt, asin, acos, atan2, fabs, fmod
from libc.stdio cimport sprintf
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set as cpp_set
from libcpp.unordered_set cimport unordered_set as unordered_set
from libcpp.list cimport list as cpp_list
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map

# import numpy/data types
import numpy as np
from numpy cimport (
    int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
    uint32_t, uint64_t, float32_t, float64_t
    )
cimport numpy as np

np.import_array()

INT8 = np.dtype(np.int8)
INT16 = np.dtype(np.int16)
INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
UINT8 = np.dtype(np.uint8)
UINT16 = np.dtype(np.uint16)
UINT32 = np.dtype(np.uint32)
UINT64 = np.dtype(np.uint64)
FLOAT32 = np.dtype(np.float32)
FLOAT64 = np.dtype(np.float64)

__all__ = ['HpxHashTable']

from .helpers cimport (
    ilog2, isqrt, imod, fmodulo, nside_to_order,
    npix_to_nside, true_angular_distance,
    fill_time_info, print_time_info,
    )
from .constants cimport NESTED, RING, MAX_Y
from .constants cimport DEG2RAD, RAD2DEG
from .constants cimport PI, TWOTHIRD, TWOPI, HALFPI, INV_TWOPI, INV_HALFPI


cdef class HpxHashTable(Healpix):
    '''
    Helper class for creating cygrid's lookup tables (HEALPix).

    This is used to speed-up cygrid, there is not really any other
    meaningful application of this class. `HpxHashTable` derives from the
    `Healpix` class and adds-in a few helper functions and lookup-table
    members.

    Usually, only two methods have to be called: `prepare_helpers` (once)
    and `calculate_output_pixels` (for each `~cygrid.Cygrid.grid` call)
    See the documentation of the methods for more information.

    Parameters
    ----------
    nside : unsigned 64-bit int
        The HEALPix `nside` parameter.
    scheme : enum
        The HEALPix `scheme`. Currently, only 'RING' is supported.

    Notes
    -----
    1. The class holds a dictionary (`target_dict`) that contains for each
       HEALPix index the list of pixel (centers) of the target (WCS) map that
       fall into this HEALPix index. For sanity a vector for the pixel centers
       is used, in case multiple pixels from the WCS map have the same HPX
       index. Since we need to invert the dictionary later, the coordinate
       pixel pair is mapped to an integer using the very simple
       transformation::

        xy_hash = xpix * MAX_Y + ypix  # with MAX_Y == 2**30

    2. All operations make use of the 'RING' scheme indexing. Therefore, we
       store the HEALPix index of the first pixel in each ring, the number of
       HEALPix pixels per ring and the index of the
       :math:`\\varphi=180^\\circ` pixel (or rather the closest pixel to the
       left of :math:`\\varphi=180^\\circ`, if ring is shifted).
    3. Furthermore, we store for each ring the HEALPix indices of a disc (with
       specified radius) as valid for :math:`\\varphi=180^\\circ`. These
       indices can easily be shifted along the :math:`\\varphi` coordinate to
       quickly construct discs at any position.
    4. For all of the above, only those indices/rings are stored that are
       going to be used. Of course, if disc radius (aka kernel support
       radius) is changed, the caches need to be recomputed.

    '''

    def __init__(
            self,
            uint64_t nside=1,
            uint32_t scheme=RING,
            bint dbg_messages=False
            ):
        super(HpxHashTable, self).__init__(nside, scheme)
        self.dbg_messages = dbg_messages

    cpdef set_optimal_nside(self, double target_res):
        '''
        Set the HEALPix `nside` such that HEALPix resolution is less than `target_res`.

        Parameters
        ----------
        target_res : double
            Requested target resolution of the HEALPix representation. `nside`
            will be set such that the HEALPix resolution is better than
            `target_res`.
        '''

        cdef uint64_t target_nside = <uint64_t> (
            sqrt(PI / 3.) / target_res + 0.5
            )
        target_nside = 1 << (ilog2(target_nside) + 1)
        self.nside = target_nside

    cpdef clear_hashes(self):
        '''
        Clear all internal caches/dictionaries.
        '''

        if self.dbg_messages:
            print('Clearing hashes, because nside did change')

        self.target_dict.clear()

        self.ring_info_hash_startpix.clear()
        self.ring_info_hash_num_pix_in_ring.clear()
        self.ring_info_hash_shifted.clear()

        self.disc_hash_ring.clear()
        self.disc_hash_indices.clear()

    cpdef _on_nside_changed(self):
        '''
        Protected method that can be used to react to changes of `nside`.
        '''

        # clear hashes on nside change
        self.clear_hashes()

    cdef prepare_helpers(
            self,
            double hpx_max_resolution,
            uint64_t[::1] xpix_fs,
            uint64_t[::1] ypix_fs,
            double[::1] xwcs_fs,
            double[::1] ywcs_fs,
            ):
        '''
        Prepare lookup-tables (caches).

        Parameters
        ----------
        hpx_max_resolution : double
            Maximum acceptable HEALPix resolution (`kernel_sigma / 2` is a
            reasonable value).
        xpix_fs/ypix_fs : `~numpy.array` [1D] of unsigned 64-bit ints
            Flat lists of the target-map pixel coordinates :math:`(x, y)`.
        xwcs_fs/ywcs_fs : `~numpy.array` [1D] of unsigned 64-bit ints
            Flat lists of the target-map world coordinates :math:`(l, b)`.

        Notes
        -----
        1. This function needs to be called once per gridding job (not
           necessarily during each invocation of the `~cygrid.Cygrid.grid`
           method), or if the kernel parameters are changed. It (re-)sets the
           internal HEALPix resolution to the largest possible value smaller
           than `hpx_max_resolution`. It is advised to use half of the
           sigma-width of the kernel. Then  the resolution will be such that
           each (Gaussian) kernel disc will be sampled with about 100 to 150
           HEALPix pixels, assuming `sphere_radius = 3 * kernel_sigma`.
        2. It also calls the internal method to fill the `target_dict` member.
           This contains for each HEALPix index the list of pixel (centers) of
           the target (WCS) map that fall into this HEALPix index. For sanity
           a vector for the pixel centers is used, in case multiple pixels
           from the WCS map have the same HEALPix index. Since we need to
           invert the dictionary later, the coordinate pixel pair is mapped
           to an integer using the very simple transformation::

                xy_hash = xpix * MAX_Y + ypix  # with MAX_Y == 2**30

        3. Then it fills the ring-information dictionaries that contain the
           HEALPix index of the first pixel in each ring, the number of
           HEALPix pixels per ring, and the index of the
           :math:`\\varphi=180^\\circ` pixel (or rather the closest
           pixel to the left of :math:`\\varphi=180^\\circ`, if ring is
           shifted).
        4. This is a Cython-domain only method.

        '''

        if self.dbg_messages:
            print('Recomputing hpx lookup tables')

        # find "optimal" HPX nside value for given kernel size
        self.set_optimal_nside(DEG2RAD * hpx_max_resolution)

        if self.dbg_messages:
            print(
                'Will use nside of {}, according to a resolution of '
                '{:.4f} arcmin'.format(
                    self.nside, RAD2DEG * self.resolution * 60.
                ))

        self._compute_target_hpx_pixels(
            xpix_fs,
            ypix_fs,
            xwcs_fs,
            ywcs_fs,
            )
        if self.dbg_messages:
            print('Size of target_dict {}'.format(self.target_dict.size()))
            print('Filling ring info hashes')

        self._fill_ring_info_hashes()
        if self.dbg_messages:
            print('Size of ring info hashes {}'.format(
                self.ring_info_hash_startpix.size()
                ))

    cdef void calculate_output_pixels(
            self,
            double[::1] lons,
            double[::1] lats,
            double kernel_radius,
            unordered_map[uint64_t, vector[uint64_t]] & output_input_mapping,
            vector[uint64_t] & output_pixels,
            ):
        '''
        Return `output_input_mapping` and its keys vector.

        Parameters
        ----------
        lons/lats : `~numpy.array` [1D] of doubles
            Input coordinates of the points to be gridded.
        kernel_radius : double
            The kernel support radius in radians. Defines out to which
            distance the convolution is to be calculated. A good value is
            3 to 4 times the sigma-width of the Gaussian kernel.
        output_input_mapping : std::unordered_map; return value (call-by-reference)
            keys : unsigned 64-bit integer
                xy-hash of the x/y indices of the target map
            values : std::vector of unsigned 64-bit integer
                list of indices for the input coordinates to be gridded

            The `output_input_mapping` is a dictionary that contains for
            each target pixel (in the desired map) a list of indices that
            point to the input coordinates (to be gridded). This allows to
            use a simple prange (parallelization) in the gridding routine.
            Otherwise a concurrent write access (race condition) could occur.
        output_pixels : std::vector of unsigned 64-bit integer; return value (call-by-reference)
            For convenience, this contains the list of keys of the
            `output_input_mapping`.

        Notes
        -----
        1. In order to compute the `output_input_mapping`, first the straight-
           forward-to-calculate `input_output_mapping` is calculated. This is
           done by (effectively) calling the HEALPix `query_disc` function
           for each of the to-be-gridded input coordinate pairs, to produce
           for each input coordinate a list of target pixels that are
           influenced. In a second step, the `input_output_mapping` is
           inverted to give `output_input_mapping`.
        2. Internally, the HEALPix `query_disc` routine isn't actually used.
           Instead, for each ring in HEALPix the :math:`\\varphi=180^\\circ`
           disc is stored and then (horizontally) shifted to any target
           coordinate, because this is computationally less demanding.
        3. This is a Cython-domain only method.
        '''

        cdef:
            vector[uint64_t] intermediary_hpidxs_vec
            unordered_map[uint64_t, vector[uint64_t]] hpxidx_output_map
            unordered_map[uint64_t, vector[uint64_t]].iterator tmp_it_64_64
            uint64_t element_counter

            vector[TimingInfo] timing_info

        fill_time_info(timing_info, '[start]', P_SINGLE)
        self._compute_input_output_mapping(
            lons,
            lats,
            kernel_radius,
            intermediary_hpidxs_vec,
            hpxidx_output_map,
            timing_info,
            )
        fill_time_info(
            timing_info, '-> <compute_input_output_mapping>', P_TOTAL
            )
        if self.dbg_messages:
            print_time_info(timing_info)
        timing_info.clear()

        if self.dbg_messages:
            element_counter = 0
            tmp_it_64_64 = hpxidx_output_map.begin()
            while tmp_it_64_64 != hpxidx_output_map.end():
                element_counter += deref(tmp_it_64_64).second.size()
                inc(tmp_it_64_64)
            print(
                'Size of intermediary hpxidx_output_map '
                '{} ({} elements)'.format(
                    hpxidx_output_map.size(), element_counter
                ))

            element_counter = 0
            tmp_it_64_64 = self.disc_hash_ring.begin()
            while tmp_it_64_64 != self.disc_hash_ring.end():
                element_counter += deref(tmp_it_64_64).second.size()
                inc(tmp_it_64_64)
            print('Size of disc_hash_ring {} ({} elements)'.format(
                self.disc_hash_ring.size(), element_counter
                ))

            element_counter = 0
            tmp_it_64_64 = self.disc_hash_indices.begin()
            while tmp_it_64_64 != self.disc_hash_indices.end():
                element_counter += deref(tmp_it_64_64).second.size()
                inc(tmp_it_64_64)
            print('Size of disc_hash_indices {} ({} elements)'.format(
                self.disc_hash_indices.size(), element_counter
                ))

        # the tricky part is, that we want to apply prange (aka open-mp)
        # but we cant prange over the input pixels, because it could happen
        # that we try to write into the same output pixel at the same time
        # so we need to invert the input_output_mapping and iterate
        # over the output
        fill_time_info(timing_info, '[start]', P_SINGLE)
        self._compute_output_input_mapping(
            intermediary_hpidxs_vec,
            hpxidx_output_map,
            output_input_mapping,
            timing_info,
            )
        fill_time_info(
            timing_info, '-> <compute_output_input_mapping>', P_TOTAL
            )
        if self.dbg_messages:
            element_counter = 0
            tmp_it_64_64 = output_input_mapping.begin()
            while tmp_it_64_64 != output_input_mapping.end():
                element_counter += deref(tmp_it_64_64).second.size()
                inc(tmp_it_64_64)
            print('Size of output_input_mapping {} ({} elements)'.format(
                output_input_mapping.size(), element_counter
                ))
        if self.dbg_messages:
            print_time_info(timing_info)
        timing_info.clear()

        # now we need to get the keys
        fill_time_info(timing_info, '[start]', P_SINGLE)
        self._get_output_pixels(output_input_mapping, output_pixels)
        fill_time_info(timing_info, '-> <get_output_pixels>', P_TOTAL)
        if self.dbg_messages:
            print('Size of output_pixels {}'.format(
                output_pixels.size()
                ))
        if self.dbg_messages:
            print_time_info(timing_info)
        timing_info.clear()
        return

    cdef void _fill_ring_info_hashes(self) noexcept nogil:
        '''
        Fill the ring info dictionaries.

        Notes
        -----
        1. All operations in this Class utilize the 'RING' scheme indexing.
           Therefore, we store (1) the HEALPix index of the first pixel in
           each ring, (2) the number of HEALPix pixels per ring, and (3) the
           index of the :math:`\\varphi=180^\\circ` pixel (or rather the
           closest pixel to the left of :math:`\\varphi=180^\\circ`, if ring
           is shifted). Only those numbers are stored that are actually
           needed (therefore the dictionary approach).
        2. This is a Cython-domain only method.

        '''

        cdef:
            uint64_t ring
            uint64_t startpix, num_pix_in_ring
            bool shifted

        self.ring_info_hash_startpix.clear()
        self.ring_info_hash_num_pix_in_ring.clear()
        self.ring_info_hash_shifted.clear()

        for ring in range(1, 4 * self._nside):
            self._get_ring_info_small(ring, startpix, num_pix_in_ring, shifted)

            self.ring_info_hash_startpix.insert(
                pair[uint64_t, uint64_t](ring, startpix)
                )
            self.ring_info_hash_num_pix_in_ring.insert(
                pair[uint64_t, uint64_t](ring, num_pix_in_ring)
                )
            self.ring_info_hash_shifted.insert(
                pair[uint64_t, bool](ring, shifted)
                )

        return

    cdef void _compute_target_hpx_pixels(
            self,
            uint64_t[::1] xpix_fs,
            uint64_t[::1] ypix_fs,
            double[::1] xwcs_fs,
            double[::1] ywcs_fs,
            ) noexcept nogil:
        '''
        Helper to create a lookup for target pixels in HEALPix representation.

        Parameters
        ----------
        xpix_fs/ypix_fs : `~numpy.array` [1D] of unsigned 64-bit ints
            Flat lists of the target-map pixel coordinates :math:`(x, y)`.
        xwcs_fs/ywcs_fs : `~numpy.array` [1D] of unsigned 64-bit ints
            Flat lists of the target-map world coordinates :math:`(l, b)`.

        Notes
        -----
        1. For each target-map pixel `(x_\\mathrm{index}, y_\\mathrm{index})`
           a hash::

               xy_hash = xpix * MAX_Y + ypix  # with MAX_Y == 2**30

           is computed. These are stored as the values of a dictionary the key
           of which is the HEALPix index of the world coordinates of the
           respective pixel pair.
        2. For sanity a vector for the pixel centers is used, in case multiple
           pixels from the WCS map share the same HEALPix index.
        3. This is a Cython-domain only method.

        '''

        cdef:
            uint64_t out_x, out_y
            uint64_t xy_hash  # this is out_x * MAX_Y + out_y
            # i.e., no more than MAX_Y times y-pixels allowed!
            uint64_t hpx_idx, i

            uint64_t index_len = xpix_fs.shape[0]
            pair[unordered_map[uint64_t, vector[uint64_t]].iterator, bool] p_it

        self.target_dict.clear()

        for i in range(index_len):
            hpx_idx = self._ang2pix(
                HALFPI - DEG2RAD * ywcs_fs[i], DEG2RAD * xwcs_fs[i]
                )
            xy_hash = xpix_fs[i] * MAX_Y + ypix_fs[i]

            # this only inserts, if key is not yet present!
            p_it = self.target_dict.insert(
                pair[uint64_t, vector[uint64_t]](hpx_idx, vector[uint64_t]())
                )
            deref(p_it.first).second.push_back(xy_hash)

        return

    cdef void _fill_disc_hash(
            self,
            vector[uint64_t] urings,
            double disc_size_rad,
            ) noexcept nogil:
        '''
        Helper to create a lookup for :math:`\\varphi=180^\\circ` discs (per
        ring).

        Parameters
        ----------
        urings : std::vector of unsigned 64-bit int
            List of unique rings to fill the disc lookups for.
        disc_size_rad : double
            The requested size of the discs to store in radians.

        Notes
        -----
        1. The :math:`\\varphi=180^\\circ` discs are calculated with a
           function very similar to the C++ HEALPix library's function
           `query_disc`. These discs are subsequently shifted along phi-axis
           to get the discs around arbitrary positions.
        3. This is a cython-domain only method.
        '''
        cdef:
            int64_t i  # Windows open-mp only works with signed loop vars
            uint64_t ring, j
            uint64_t startpix, num_pix_in_ring
            vector[uint64_t] disc_indices_vec, disc_rings_vec
            vector[uint64_t] new_urings

        # first check, which of the rings are not yet in the dictionary
        # missing rings are set to a default empty vector
        # this is done to allow concurrent computation of the discs below
        for i in range(urings.size()):
            ring = urings[i]
            if self.disc_hash_indices.count(ring) > 0:
                continue
            new_urings.push_back(ring)
            self.disc_hash_indices[ring] = vector[uint64_t]()
            self.disc_hash_ring[ring] = vector[uint64_t]()

        for i in prange(
                new_urings.size(), nogil=True, schedule='static', chunksize=50
                ):
            ring = new_urings[i]
            disc_indices_vec = self._query_disc_phi180(
                disc_size_rad, ring
                )

            disc_rings_vec = disc_indices_vec
            for j in range(disc_indices_vec.size()):
                disc_rings_vec[j] = \
                    <uint64_t> self._pix2ring(disc_indices_vec[j])

            # we store both, the HPX indices, as well as the ring number
            # for each pixel in the disc, for convenience
            self.disc_hash_indices.at(ring).insert(
                self.disc_hash_indices.at(ring).end(),
                disc_indices_vec.begin(),
                disc_indices_vec.end()
                )
            self.disc_hash_ring.at(ring).insert(
                self.disc_hash_ring.at(ring).end(),
                disc_rings_vec.begin(),
                disc_rings_vec.end()
                )

        return

    cdef void _compute_input_output_mapping(
            self,
            double[::1] lons,
            double[::1] lats,
            double disc_size_rad,
            vector[uint64_t] &intermediary_hpidxs_vec,
            unordered_map[uint64_t, vector[uint64_t]] &hpxidx_output_map,
            vector[TimingInfo] &timing_info,
            ) noexcept nogil:
        '''
        Helper to create the `input_output_mapping`.

        This is the most computationally demanding function, probably because
        random access to `std::unordered_map` is pretty slow for large dicts
        (at least in recent GCC versions, i.e., 4.7+).

        Parameters
        ----------
        lons/lats : `~numpy.array` [1D] of doubles
            Input coordinates of the points to be gridded.
        disc_size_rad : double
            The requested size of the discs to store in radians.

        Returns
        -------
        intermediary_hpidxs_vec : std::vector of unsigned 64-bit integer
            (Intermediate) HEALPix index that contains the input coordinate
        hpxidx_output_map : std::unordered_map
            keys : unsigned 64-bit integer
                (Intermediate) HEALPix index that contains the input
                coordinate
            values : std::vector of unsigned 64-bit integer
                List of target-map pixels (as xy-hash) that lie within a disc
                of size "disc_size_rad" around the (intermediate) HEALPix
                index and hence the input coordinate pair (within errors)

        Notes
        -----
        1. Using the :math:`\\varphi=180^\\circ` discs, by shifting along the
           phi-axis we get the discs around each input position.
        2. By using an intermediary HEALPix index lookup, we can lower the
           necessary memory of the lookup tables. Since the intermediary
           HEALPix pixel size is coupled to the kernel size, large smoothing
           radii (or rather a large number of pixels per kernel sphere) could
           otherwise lead to high overhead, when for each coord a disc needs
           to be computed.
        3. This is a Cython-domain only method.

        '''

        cdef:
            int64_t i  # Windows open-mp only works with signed loop vars
            uint64_t j
            # need temporary x, y, z coords for healpy
            double theta  # need this only temporary
            vector[double] phis_vec
            vector[uint64_t] intermediary_unique_hpidxs_vec
            unordered_set[uint64_t] intermediary_hpidxs_set
            cpp_list[uint64_t] intermediary_unique_hpidxs_list
            unordered_set[uint64_t].iterator _set_it
            vector[uint64_t] rings_vec, urings_vec
            cpp_set[uint64_t] urings_set

            uint64_t hpidx, ring, startpix, num_pix_in_ring, pix_in_ring
            double phi, dphi

            vector[uint64_t] disc_indices_vec, disc_rings_vec
            uint64_t disc_idx, disc_ring, j_startpix, j_num_pix_in_ring
            double dshift
            int64_t ishift, j_pix_in_ring  # this must be signed

            vector[uint64_t] tmp_output_pixels
            vector[uint64_t] pixels
            uint64_t hpxpixel
            int64_t coords_len = lons.shape[0]
            int64_t intermediary_len

            unordered_map[uint64_t, vector[uint64_t]] input_output_mapping

        intermediary_hpidxs_vec.resize(coords_len)
        for i in prange(coords_len, nogil=True, schedule='static', chunksize=500):
            theta = HALFPI - DEG2RAD * lats[i]
            phi = DEG2RAD * lons[i]
            hpidx = self._ang2pix(theta, phi)
            # this list is used for assigning each input coord a hpx index
            intermediary_hpidxs_vec[i] = hpidx

        fill_time_info(
            timing_info, '- intermediary hpx index calculation', P_MULTI
            )
        for i in range(coords_len):
            # this set stores the unique hpxidx (for which we have to calculate
            # discs)
            # Note: this routine can be made much faster, if an
            # unordered_set is used; however, this will cost a lot of
            # efficiency in the heavy-worker prange routine below;
            # so we have to sort the set afterwards
            # (yields about 25% speed for this block)
            intermediary_hpidxs_set.insert(intermediary_hpidxs_vec[i])

        _set_it = intermediary_hpidxs_set.begin()
        while _set_it != intermediary_hpidxs_set.end():
            intermediary_unique_hpidxs_list.push_back(deref(_set_it))
            inc(_set_it)
        intermediary_unique_hpidxs_list.sort()

        # convert set to a unique vector for convenience
        intermediary_unique_hpidxs_vec.clear()
        intermediary_unique_hpidxs_vec.insert(
            intermediary_unique_hpidxs_vec.end(),
            intermediary_unique_hpidxs_list.begin(),
            intermediary_unique_hpidxs_list.end(),
            )
        intermediary_len = intermediary_unique_hpidxs_vec.size()
        fill_time_info(
            timing_info,
            '- constructing/sorting unique hpx index vector',
            P_SINGLE
            )

        # find the (unique) HPX rings needed for each intermediary hpx index
        # this is the list of rings for which disc_info dictionary is filled
        # also prepare the HPX index, ring index and phi value for each of
        # the intermediary hpx indices
        for i in range(intermediary_len):
            hpidx = intermediary_unique_hpidxs_vec[i]
            # get theta phi for each hpx idx (center value!)
            self._pix2ang(hpidx, theta, phi)
            phis_vec.push_back(phi)
            rings_vec.push_back(self._pix2ring(hpidx))
            urings_set.insert(rings_vec[i])

        urings_vec.clear()
        urings_vec.insert(
            urings_vec.end(),
            urings_set.begin(), urings_set.end()
            )

        fill_time_info(
            timing_info, '- find unique hpx rings', P_SINGLE
            )
        # fill the disc info dicts
        self._fill_disc_hash(urings_vec, disc_size_rad)
        fill_time_info(timing_info, '- fill disc hashes', P_SINGLE)

        # now compute the input-output mapping
        hpxidx_output_map.clear()
        # need to prepare the dictionary to avoid race conditions
        for i in range(intermediary_len):
            hpidx = intermediary_unique_hpidxs_vec[i]
            tmp_output_pixels = vector[uint64_t]()
            hpxidx_output_map[hpidx] = tmp_output_pixels
        fill_time_info(timing_info, '- prepare <hpxidx_output_map>', P_SINGLE)

        for i in prange(
                intermediary_len, nogil=True, schedule='static', chunksize=500
                ):
            hpidx = intermediary_unique_hpidxs_vec[i]
            ring = <uint64_t> rings_vec[i]
            phi = phis_vec[i]

            startpix = self.ring_info_hash_startpix.at(ring)
            num_pix_in_ring = self.ring_info_hash_num_pix_in_ring.at(ring)

            disc_indices_vec = self.disc_hash_indices.at(ring)
            disc_rings_vec = self.disc_hash_ring.at(ring)

            # amount of shifting is simply the difference of phi w.r.t. to 180d
            dphi = phi - PI
            pix_in_ring = hpidx - startpix

            for j in range(disc_indices_vec.size()):
                disc_idx = disc_indices_vec.at(j)
                disc_ring = disc_rings_vec.at(j)

                j_startpix = self.ring_info_hash_startpix.at(disc_ring)
                j_num_pix_in_ring = self.ring_info_hash_num_pix_in_ring.at(disc_ring)
                j_pix_in_ring = disc_idx - j_startpix

                # shifting is easy, because the number of disc-pixels per ring
                # is constant! just need to get the correct HPX pixel indices
                # by looking at the difference of phi from 180d
                dshift = j_num_pix_in_ring * dphi / 2. / PI
                if dshift >= 0:
                    ishift = <int64_t> (dshift+0.5)
                else:
                    ishift = <int64_t> (dshift-0.5)
                # j_pix_in_ring += ishift  # inplace not possible with prange
                j_pix_in_ring = j_pix_in_ring + ishift
                hpxpixel = imod(j_pix_in_ring, j_num_pix_in_ring) + j_startpix

                # only add those pixels to the hpxidx_output_map that are
                # actually present on the target dict
                if self.target_dict.count(hpxpixel) < 1:
                    continue

                pixels = self.target_dict.at(hpxpixel)
                hpxidx_output_map.at(hpidx).insert(
                    hpxidx_output_map.at(hpidx).end(),
                    pixels.begin(), pixels.end()
                    )

        fill_time_info(timing_info, '- compute <hpxidx_output_map>', P_MULTI)
        return

    cdef void _compute_output_input_mapping(
            self,
            vector[uint64_t] &intermediary_hpidxs_vec,
            unordered_map[uint64_t, vector[uint64_t]] & hpxidx_output_map,
            unordered_map[uint64_t, vector[uint64_t]] & output_input_map,
            vector[TimingInfo] &timing_info,
            ) noexcept nogil:
        '''
        Helper to invert the `hpxidx_output_map`.

        Parameters
        ----------
        intermediary_hpidxs_vec : std::vector of unsigned 64-bit integer
            (Intermediate) HEALPix index that contains the input coordinate
        hpxidx_output_map : std::unordered_map
            keys : unsigned 64-bit integer
                Index over the intermediary HEALPix indices
            values : std::vector of unsigned 64-bit integer
                List of target-map pixels (as xy-hash) that lie within a disc
                of size `disc_size_rad` around the input coordinate pair
        output_input_mapping : std::unordered_map; return value (call-by-reference)
            keys : unsigned 64-bit integer
                Target-map pixels (as xy-hash)
            values : std::vector of unsigned 64-bit integer
                List of input coordinate pair indices contributing to the
                according target-map pixel

        Notes
        -----
        This is a Cython-domain only method.
        '''

        cdef:
            vector[uint64_t] tmp_output_pixels
            vector[uint64_t].iterator pixvec_it
            unordered_map[uint64_t, vector[uint64_t]].iterator _mit
            pair[unordered_map[uint64_t, vector[uint64_t]].iterator, bool] _mit_p
            uint64_t _pix
            int64_t pixel  # Windows open-mp only works with signed loop vars

            uint64_t j, max_threads = openmp.omp_get_max_threads()
            vector[unordered_map[uint64_t, vector[uint64_t]]] tmp_maps

        # we parallelize by doing the inversion for local dicts, later we
        # have to merge them
        for j in range(max_threads - 1):
            tmp_maps.push_back(unordered_map[uint64_t, vector[uint64_t]]())

        for pixel in prange(
                intermediary_hpidxs_vec.size(),
                nogil=True, schedule='dynamic'
                ):

            j = threadid()

            tmp_output_pixels = hpxidx_output_map.at(
                intermediary_hpidxs_vec[pixel]
                )

            # this is ugly :-(
            # tried to do this with a pointer to output_input_map and
            # tmp_maps.at(j), respectively, but it didn't compile
            # perhaps due to the prange (auto) thread-local determination
            if j == 0:
                pixvec_it = tmp_output_pixels.begin()
                while pixvec_it != tmp_output_pixels.end():
                    _pix = deref(pixvec_it)
                    _mit = output_input_map.find(_pix)
                    if _mit == output_input_map.end():
                        # Should we handle the bool? Can insertion really fail?
                        _mit_p = output_input_map.insert(
                            pair[uint64_t, vector[uint64_t]](
                                _pix, vector[uint64_t]()
                                )
                            )
                        _mit = _mit_p.first

                    deref(_mit).second.push_back(pixel)

                    inc(pixvec_it)
            else:
                pixvec_it = tmp_output_pixels.begin()
                while pixvec_it != tmp_output_pixels.end():
                    _pix = deref(pixvec_it)
                    _mit = tmp_maps.at(j - 1).find(_pix)
                    if _mit == tmp_maps.at(j - 1).end():
                        # Should we handle the bool? Can insertion really fail?
                        _mit_p = tmp_maps.at(j - 1).insert(
                            pair[uint64_t, vector[uint64_t]](
                                _pix, vector[uint64_t]()
                                )
                            )
                        _mit = _mit_p.first

                    deref(_mit).second.push_back(pixel)

                    inc(pixvec_it)

        fill_time_info(
            timing_info,
            '- invert <hpxidx_output_map> to compute <output_input_map>',
            P_MULTI
            )

        # now merge
        for j in range(1, max_threads):

            o_it = tmp_maps.at(j - 1).begin()
            while o_it != tmp_maps.at(j - 1).end():
                key = deref(o_it).first
                values = deref(o_it).second

                m_it = output_input_map.find(key)
                if m_it == output_input_map.end():
                    output_input_map.insert(
                        pair[uint64_t, vector[uint64_t]](
                            key, vector[uint64_t]()
                            )
                        )
                output_input_map.at(key).insert(
                    output_input_map.at(key).end(),
                    values.begin(),
                    values.end(),
                    )

                inc(o_it)

        fill_time_info(timing_info, '- merge <output_input_map>', P_SINGLE)

        return

    cdef void _get_output_pixels(
            self,
            unordered_map[uint64_t, vector[uint64_t]] & output_input_mapping,
            vector[uint64_t] & output_pixels,
            ) noexcept nogil:
        '''
        Convenience function to obtain the keys of `output_input_mapping`.

        Parameters
        ----------
        output_input_mapping : std::unordered_map
            keys : unsigned 64-bit integer
                Target-map pixels (as xy-hash)
            values : std::vector of unsigned 64-bit integer
                List of input coordinate pair indices contributing to the
                according target-map pixel
        output_pixels : std::vector of unsigned 64-bit integer; return value (call-by-reference)
            Keys of `output_input_mapping`

        Notes
        -----
        This is a Cython-domain only method.
        '''

        cdef:
            unordered_map[uint64_t, vector[uint64_t]].iterator oi_map_it

        oi_map_it = output_input_mapping.begin()
        while oi_map_it != output_input_mapping.end():
            output_pixels.push_back(deref(oi_map_it).first)
            inc(oi_map_it)

        return
