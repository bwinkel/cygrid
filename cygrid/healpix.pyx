#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

# ####################################################################
#
# title                  :healpix.pyx
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
from cython.parallel import prange
from cython.operator cimport dereference as deref, preincrement as inc
from cpython cimport bool as python_bool

# import C/C++ modules
from libc.math cimport exp, cos, sin, sqrt, asin, acos, atan2, fabs, fmod
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set as cpp_set
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


__all__ = ['Healpix']


from .helpers cimport (
    ilog2, isqrt, imod, fmodulo, nside_to_order,
    npix_to_nside, true_angular_distance
    )
from .constants cimport NESTED, RING, MAX_Y
from .constants cimport DEG2RAD, RAD2DEG
from .constants cimport PI, TWOTHIRD, TWOPI, HALFPI, INV_TWOPI, INV_HALFPI


cdef class Healpix(object):
    '''
    Helper class for HEALPix operations.

    Implements conversion between HEALPix world and pixel coordinates ('RING'
    representation only). Also provides a `query_disc` method that returns
    a disc footprint for :math:`\\varphi = 180^\\circ` for a given ring.

    Parameters
    ----------
    nside : unsigned 64-bit int
        The HEALPix `nside` parameter.
    scheme : enum
        The HEALPix `scheme`. Currently, only 'RING' is supported.

    Notes
    -----
    A lot of the code in this class was adapted from the HEALPix C++ library.
    (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
    see `<http://healpix.sourceforge.net>`_)
    '''

    def __init__(self, uint64_t nside=1, uint32_t scheme=RING):
        self._nside = nside
        assert scheme in [RING, NESTED], 'Scheme must be RING or NESTED.'
        if scheme == NESTED:
            raise NotImplementedError(
                'Currently, only RING scheme is supported.'
                )
        self._scheme = scheme
        self._params_dirty = True

        self._update_params()

    property nside:
        '''
        property nside : unsigned 64-bit int
            The HEALPix 'nside' parameter. (Re-)setting 'nside' updates other
            properties.
        '''

        def __get__(self):
            return self._nside

        def __set__(self, value):
            self._params_dirty = self._nside != value
            self._nside = value
            self._update_params()

    property scheme:
        '''
        property scheme : enum
            The HEALPix 'scheme'. Currently only 'RING' is supported.
        '''

        def __get__(self):
            return self._scheme

        def __set__(self, value):
            assert value in [RING, NESTED], 'Scheme must be RING or NESTED.'
            if value == NESTED:
                raise NotImplementedError(
                    'Currently, only RING scheme is supported.'
                    )
            self._scheme = value

    property npix:
        '''
        property npix : unsigned 64-bit int
            Return the HEALPix 'npix' parameter.
        '''

        def __get__(self):
            return self._npix

    property nrings:
        '''
        property nrings : unsigned 64-bit int
            Return the HEALPix 'nrings' parameter.
        '''

        def __get__(self):
            return self._nrings

    property order:
        '''
        property order : unsigned 64-bit int
            Return the HEALPix 'order' parameter.
        '''

        def __get__(self):
            return self._order

    property npface:
        '''
        property nface : unsigned 64-bit int
            Return the HEALPix 'nface' parameter.
        '''

        def __get__(self):
            return self._npface

    property ncap:
        '''
        property ncap : unsigned 64-bit int
            Return the HEALPix 'ncap' parameter.
        '''

        def __get__(self):
            return self._ncap

    property omega:
        '''
        property omega : double
            Return the solid angle, 'omega' of the HEALPix pixels
            for current `nside`.
        '''

        def __get__(self):
            return self._omega

    property resolution:
        '''
        property resolution : double
            Return the angular size of the HEALPix pixels for current `nside`.
        '''

        def __get__(self):
            return self._resolution

    cpdef _update_params(self):
        '''
        Update HEALPix parameters if necessary (when `_params_dirty`
        is set to True).

        Note, another private method (`_on_nside_changed`) is called
        that can be used by derived classes to hook-in.
        '''

        if self._params_dirty:

            self._order = nside_to_order(self._nside)
            assert self._order >= 0, 'Nside must be a power of 2.'
            self._nrings = 4 * self._nside - 1
            self._max_npix_per_ring = 4 * self._nside
            self._npface = self._nside ** 2
            self._ncap = (self._npface - self._nside) << 1
            self._npix = 12 * self._npface
            self._fact2 = 4. / self._npix
            self._fact1 = (self._nside << 1) * self._fact2
            self._omega = 4. * PI / self._npix
            self._resolution = sqrt(self._omega)

            self._on_nside_changed()

    cpdef _on_nside_changed(self):
        '''Can be used in derived classes to react to `nside` changes.'''
        pass

    cdef inline void _get_ring_info_small(
            self,
            uint64_t ring,
            uint64_t & startpix,
            uint64_t & num_pix_in_ring,
            bool & shifted,
            ) noexcept nogil:
        '''
        Return start index and number of pixels per healpix ring.

        Parameters
        ----------
        ring : unsigned 64-bit int
            The HEALPix ring parameter to get information on.
        startpix : unsigned 64-bit int; return value (call-by-reference)
            The HEALPix index of the first pixel in ring.
        num_pix_in_ring : unsigned 64-bit int; return value (call-by-reference)
            The number of HEALPix pixels in ring.
        shifted : bool; return value (call-by-reference)
            Whether the central HEALPix pixel in that ring is shifted from
            :math:`\\varphi=180^\\circ`.

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a Cython-domain only method.
        '''

        cdef uint64_t nr

        # Note: we need to use the idiom (&refval)[0] to modifiy the
        # call-by-reference values
        # otherwise a compiler error "Assignment to reference 'value'"
        # is thrown

        if ring < self._nside:

            (&shifted)[0] = <bool> True
            (&num_pix_in_ring)[0] = 4 * ring
            (&startpix)[0] = 2 * ring * (ring - 1)

        elif ring < 3 * self._nside:

            (&shifted)[0] = <bool> (((ring - self._nside) & 1) == 0)
            (&num_pix_in_ring)[0] = 4 * self._nside
            (&startpix)[0] = self._ncap + (ring - self._nside) * num_pix_in_ring

        else:

            (&shifted)[0] = <bool> True
            nr = 4 * self._nside - ring
            (&num_pix_in_ring)[0] = 4 * nr
            (&startpix)[0] = self._npix - 2 * nr * (nr + 1)

    def get_ring_info_small(self, uint64_t ring):
        '''
        Return start index and number of pixels per healpix ring.

        Wrapper around the Cython-only (private) method
        `_get_ring_info_small`.

        Parameters
        ----------
        ring : unsigned 64-bit int
            The HEALPix ring parameter to get information on.

        Returns
        -------
        startpix : unsigned 64-bit int
            The HEALPix index of the first pixel in ring.
        num_pix_in_ring : unsigned 64-bit int
            The number of HEALPix pixels in ring.
        shifted : bool
            Whether the central HEALPix pixel in that ring is shifted from
            :math:`\\varphi=180^\\circ`.

        Raises
        ------
        ValueError
            Ring index not in valid range.
        '''

        cdef:
            uint64_t startpix, num_pix_in_ring
            bool shifted,

        if not (ring > 0 and ring <= 4 * self._nside - 1):
            raise ValueError('Ring index not in valid range.')

        self._get_ring_info_small(ring, startpix, num_pix_in_ring, shifted)

        return startpix, num_pix_in_ring, python_bool(shifted)

    cdef uint64_t _pix2ring(self, uint64_t pix) noexcept nogil:
        '''
        Return 'ring' number that contains the HEALPix pixel, 'pix'.

        Parameters
        ----------
        pix : unsigned 64-bit int
            The HEALPix index.

        Returns
        -------
        ring : unsigned 64-bit int
            The HEALPix ring.

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a Cython-domain only method.
        '''

        if pix < self._ncap:

            # North Polar cap, counted from North pole
            return (1 + isqrt(1 + 2 * pix)) >> 1

        elif pix < self._npix - self._ncap:

            # Equatorial region, counted from North pole
            return (pix - self._ncap) // (4 * self._nside) + self._nside

        else:

            # South Polar cap
            return 4 * self._nside - (
                (1 + isqrt(2 * (self._npix - pix) - 1)) >> 1
                )

    def pix2ring(self, uint64_t pix):
        '''
        Return 'ring' number that contains the HEALPix pixel, 'pix'.

        Wrapper around the Cython-only (private) method _pix2ring.

        Parameters
        ----------
        pix : unsigned 64-bit int
            The HEALPix index.

        Returns
        -------
        ring : unsigned 64-bit int
            The HEALPix ring.

        Raises
        ------
        ValueError
            Pixel index not in valid range.
        '''

        if not (pix >= 0 and pix < self._npix):
            raise ValueError('Pixel index not in valid range.')

        return self._pix2ring(pix)

    def pix2ring_many(self, long[::1] pix):
        '''
        Return 'ring' numbers containing HEALPix pixels given in array, 'pix'.

        Wrapper around the Cython-only (private) method _pix2ring.

        Parameters
        ----------
        pix : `~numpy.array` of unsigned 64-bit ints
            The HEALPix indices.

        Returns
        -------
        ring : `~numpy.array` of unsigned 64-bit ints
            The HEALPix rings.
        '''

        cdef:

            np.ndarray[long, ndim=1] rings = np.empty(
                (pix.size,), dtype=np.int64
                )
            long[::1] rings_v = rings
            long i

        for i in range(pix.size):

            rings_v[i] = <long> self._pix2ring(pix[i])

        return rings

    cdef uint64_t _loc2pix(
            self,
            double z,
            double phi,
            double sin_theta,
            bool have_sin_theta
            ) noexcept nogil:
        '''
        Convert location :math:`(z, \\varphi)` to HEALPix index.

        Parameters
        ----------
        z : double
            :math:`z` coordinate (:math:`z = \\cos\\vartheta`).
        phi : double
            :math:`\\varphi` coordinate.
        sin_theta : double
            For very small :math:`\\vartheta`, one should provide
            additionally the sine of :math:`\\vartheta` to avoid numerical
            inaccuracy. In that case, set `have_sin_theta` to true.
        have_sin_theta : bool
            See `sin_theta`.

        Returns
        -------
        pix : unsigned 64-bit int
            The HEALPix pixel index.

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a Cython-domain only method.
        '''

        cdef:
            double za = fabs(z)
            double tt = fmodulo(phi * INV_HALFPI, 4.0)  # in [0,4)

            double temp1, temp2, tp, tmp
            uint64_t nl4, jp, jm, ir, kshift, t1, ip

        if za <= TWOTHIRD:

            # Equatorial region
            nl4 = 4 * self._nside
            temp1 = self._nside * (0.5 + tt)
            temp2 = self._nside * z * 0.75
            jp = <uint64_t> (temp1 - temp2)  # index of  ascending edge line
            jm = <uint64_t> (temp1 + temp2)  # index of descending edge line

            # ring number counted from z = 2/3
            ir = self._nside + 1 + jp - jm  # in {1,2n+1}
            kshift = 1 - (ir & 1)  # kshift = 1 if ir even, 0 otherwise

            t1 = jp + jm + kshift + 1 + nl4 + nl4 - self._nside
            if self._order > 0:
                ip = (t1 >> 1) & (nl4 - 1)  # in {0,4n-1}
            else:
                ip = (t1 >> 1) % nl4  # in {0,4n-1}

            return self._ncap + (ir - 1) * nl4 + ip

        else:

            # North & South polar caps
            tp = tt - (<uint64_t> tt)
            if (za < 0.99) or (not have_sin_theta):
                tmp = self._nside * sqrt(3. * (1. - za))
            else:
                tmp = self._nside * sin_theta / sqrt((1. + za) / 3.)

            jp = <uint64_t> (tp * tmp)  # increasing edge line index
            jm = <uint64_t> ((1.0 - tp) * tmp)  # decreasing edge line index

            ir = jp + jm + 1  # ring number counted from the closest pole
            ip = <uint64_t> (tt * ir)  # in {0,4*ir-1}

            if z > 0:
                return 2 * ir * (ir - 1) + ip
            else:
                return self._npix - 2 * ir * (ir + 1) + ip

    cdef uint64_t _ang2pix(self, double theta, double phi) noexcept nogil:
        '''
        Return the pixel index containing the angular coordinates :math:`(\\varphi, \\vartheta)`.

        Parameters
        ----------
        theta : double
            :math:`\\vartheta` coordinate (:math:`\\vartheta = \\pi/2 - \\mathrm{latitude}`).
        phi : double
            :math:`\\varphi` coordinate (longitude).

        Returns
        -------
        pix : unsigned 64-bit int
            The HEALPix pixel index.

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a cython-domain only method.
        '''

        if theta < 0.01 or theta > PI - 0.01:
            return self._loc2pix(cos(theta), phi, sin(theta), <bool> True)
        else:
            return self._loc2pix(cos(theta), phi, 0., <bool> False)

    def ang2pix(self, double theta, double phi):
        '''
        Return the pixel index containing the angular coordinates :math:`(\\varphi, \\vartheta)`.

        Wrapper around the Cython-only (private) method _ang2pix.

        Parameters
        ----------
        theta : double
            :math:`\\vartheta` coordinate (:math:`\\vartheta = \\pi/2 - \\mathrm{latitude}`).
        phi : double
            :math:`\\varphi` coordinate (longitude).

        Returns
        -------
        pix : unsigned 64-bit int
            The HEALPix pixel index.

        Raises
        ------
        ValueError
            Invalid theta value.
        '''

        if not (theta >= 0 and theta <= PI):
            raise ValueError('Invalid theta value.')

        return self._ang2pix(theta, phi)

    def ang2pix_many(self, double[::1] theta, double[::1] phi):
        '''
        Return the pixel indices containing the angular coordinates of :math:`(\\varphi, \\vartheta)` arrays.

        Wrapper around the Cython-only (private) method _ang2pix.

        Parameters
        ----------
        theta : `~numpy.array` of doubles
            :math:`\\vartheta` coordinate (:math:`\\vartheta = \\pi/2 - \\mathrm{latitude}`).
        phi : `~numpy.array` of doubles
            :math:`\\varphi` coordinate (longitude).

        Returns
        -------
        pix : `~numpy.array` of unsigned 64-bit ints
            The HEALPix pixel index.
        '''

        cdef:

            np.ndarray[long, ndim=1] hpx = np.empty(
                (theta.size,), dtype=np.int64
                )
            long[::1] hpx_v = hpx
            long i

        for i in range(theta.size):

            hpx_v[i] = <long> self._ang2pix(theta[i], phi[i])

        return hpx

    cdef void _pix2loc (
            self,
            uint64_t pix,
            double &z,
            double &phi,
            double &sin_theta,
            bool &have_sin_theta
            ) noexcept nogil:
        '''
        Convert HEALPix index to location :math:`(z, \\varphi)`.

        Parameters
        ----------
        pix : unsigned 64-bit int
            The HEALPix pixel index.
        z : double; return value (call-by-reference)
            :math:`z` coordinate (:math:`z = \\cos\\vartheta`).
        phi : double; return value (call-by-reference)
            :math:`\\varphi` coordinate.
        sin_theta : double; return value (call-by-reference)
            For very small :math:`\\vartheta`, the function additionally
            returns the sine of `\\vartheta`. In that case, `have_sin_theta`
            is returning `true`.
        have_sin_theta : bool; return value (call-by-reference)
            See `sin_theta`.

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a Cython-domain only method.
        '''

        cdef:
            double _z, _phi, _sin_theta
            bool _have_sin_theta = False
            uint64_t ip, itmp, nl4, iphi, iring
            double tmp, fodd

        if pix < self._ncap:

            # North Polar cap
            iring = (1 + isqrt(1+2*pix)) >> 1  # counted from North pole
            iphi = (pix + 1) - 2 * iring * (iring - 1)

            tmp = (iring * iring) * self._fact2
            _z = 1.0 - tmp
            if _z > 0.99:
                _sin_theta = sqrt(tmp * (2. - tmp))
                _have_sin_theta = True
            _phi = (iphi - 0.5) * HALFPI / iring

        elif pix < self._npix - self._ncap:

            # Equatorial region
            nl4 = 4 * self._nside
            ip = pix - self._ncap
            if self._order >= 0:
                itmp = ip >> (self._order + 2)
            else:
                itmp = ip // nl4

            iring = itmp + self._nside
            iphi = ip - nl4 * itmp + 1
            # 1 if iring+nside is odd, 1/2 otherwise
            if (iring + self._nside) & 1 > 0:
                fodd = 1.0
            else:
                fodd = 0.5

            _z = (2. * self._nside - iring) * self._fact1
            _phi = (iphi - fodd) * PI * 0.75 * self._fact1

        else:

            # South Polar cap
            ip = self._npix - pix
            iring = (1 + isqrt(2 * ip - 1)) >> 1 # counted from South pole
            iphi = 4 * iring + 1 - ((<int64_t> ip) - 2 * iring * (iring - 1))

            tmp = (iring * iring) * self._fact2
            _z = tmp - 1.0
            if _z < -0.99:
                _sin_theta = sqrt(tmp * (2. - tmp))
                _have_sin_theta = True
            _phi = (iphi - 0.5) * HALFPI / iring

        (&have_sin_theta)[0] = _have_sin_theta
        (&sin_theta)[0] = _sin_theta
        (&z)[0] = _z
        (&phi)[0] = _phi

    cdef void _pix2ang(
            self, uint64_t pix, double & theta, double & phi
            ) noexcept nogil:
        '''
        Convert HEALPix index to angular coordinates :math:`(\\varphi, \\vartheta)`.

        Parameters
        ----------
        pix : unsigned 64-bit int
            The HEALPix pixel index.
        theta : double; return value (call-by-reference)
            :math:`\\vartheta` coordinate (:math:`\\vartheta = \\pi/2 - \\mathrm{latitude}`).
        phi : double; return value (call-by-reference)
            :math:`\\varphi` coordinate (longitude).

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a Cython-domain only method.
        '''

        cdef:
            double _z, _phi, _sin_theta
            bool _have_sin_theta

        self._pix2loc(pix, _z, _phi, _sin_theta, _have_sin_theta)

        if _have_sin_theta:
            (&theta)[0] = atan2(_sin_theta, _z)
            (&phi)[0] = _phi
        else:
            (&theta)[0] = acos(_z)
            (&phi)[0] = _phi

    def pix2ang(self, uint64_t pix):
        '''
        Convert HEALPix index to angular coordinates :math:`(\\varphi, \\vartheta)`.

        Wrapper around the Cython-only (private) method `_pix2ang`.

        Parameters
        ----------
        pix : unsigned 64-bit int
            The HEALPix pixel index.

        Returns
        -------
        theta : double
            :math:`\\vartheta` coordinate (:math:`\\vartheta = \\pi/2 - \\mathrm{latitude}`).
        phi : double
            :math:`\\varphi` coordinate (longitude).

        Raises
        ------
        ValueError
            `pix` not in valid range.
        '''

        if not (pix >= 0 and pix < self._npix):
            raise ValueError('pix not in valid range.')

        cdef:
            double phi, theta

        self._pix2ang(pix, theta, phi)
        return theta, phi

    def pix2ang_many(self, long[::1] pix):
        '''
        Convert HEALPix indices to angular coordinates of :math:`(\\varphi, \\vartheta)` arrays.

        Wrapper around the Cython-only (private) method `_pix2ang`.

        Parameters
        ----------
        pix : `~numpy.array` of unsigned 64-bit ints
            The HEALPix pixel index.

        Returns
        -------
        theta : `~numpy.array` of doubles
            :math:`\\vartheta` coordinate (:math:`\\vartheta = \\pi/2 - \\mathrm{latitude}`).
        phi : `~numpy.array` of doubles
            :math:`\\varphi` coordinate (longitude).
        '''

        cdef:

            np.ndarray[double, ndim=1] theta = np.empty(
                (pix.size,), dtype=np.float64
                )
            np.ndarray[double, ndim=1] phi = np.empty(
                (pix.size,), dtype=np.float64
                )
            double[::1] theta_v = theta
            double[::1] phi_v = phi
            double _t, _p
            long i

        for i in range(pix.size):

            self._pix2ang(pix[i], _t, _p)
            theta[i] = _t
            phi[i] = _p

        return theta, phi

    cdef vector[uint64_t] _query_disc_phi180(
            self, double disc_size_rad, uint64_t disc_ring
            ) noexcept nogil:
        '''
        Return HEALPix indices of a disc around :math:`\\varphi=180^\\circ` for 'disc_ring'.

        This is a modified version of the HEALPix library `~healpy.query_disc`
        routine that returns the HEALPix indices of a disc around arbitrary
        coordinates. Here, it was modified to only work for coordinates having
        :math:`\\varphi=180^\\circ` and a :math:`\\vartheta` that coincides
        with the latitude of a HEALPix ring. These discs can be shifted along
        :math:`\\varphi` - which is faster than calling `~healpy.query_disc`
        if one cashes the :math:`\\varphi=180^\\circ` discs.

        Parameters
        ----------
        disc_size_rad : double
            Size of the disc in radians.
        disc_ring : unsigned 64-bit int
            HEALPix ring index of the requested disc center.

        Returns
        -------
        disc_indices : std::vector[unsigned 64-bit int]
            The HEALPix indices of the disc.

        Notes
        -----
        1. This function was adapted from the HEALPix C++ library.
         (Copyright (C) 2003-2012 Max-Planck-Society; author Martin Reinecke;
         see `<http://healpix.sourceforge.net>`_)
        2. This is a Cython-domain only method.
        '''
        cdef:
            uint64_t i
            uint64_t ring
            int64_t dring, ip
            uint64_t startpix, num_pix_in_ring, mid_pixel
            bool shifted
            double disc_theta, disc_phi, theta, phi, utheta  # in rad
            double d  # various angular distances

            uint64_t l, u
            uint64_t slice_min_pix, slice_max_pix

            double z, z0, cosrbig, xa, x, ysq, dphi

            vector[uint64_t] disc_indices

        # query disc center coords for later use
        ring = disc_ring

        self._get_ring_info_small(ring, startpix, num_pix_in_ring, shifted)
        self._pix2ang(startpix + num_pix_in_ring // 2, disc_theta, disc_phi)

        disc_phi = PI

        utheta = disc_theta - disc_size_rad

        if utheta < 1.e-30:
            utheta = 1.e-30
        startpix = self._ang2pix(utheta, PI)
        dring = self._pix2ring(startpix) - ring + 2

        # find the ring index of the upper edge of the sphere
        # note, this will be the smallest ring index in sphere (North)
        while disc_ring + dring > 1:
            dring -= 1
            ring = disc_ring + dring
            self._get_ring_info_small(ring, startpix, num_pix_in_ring, shifted)
            self._pix2ang(startpix + num_pix_in_ring // 2, theta, phi)

            d = true_angular_distance(
                phi, HALFPI - theta,
                disc_phi, HALFPI - disc_theta,
                )

            if d > disc_size_rad:
                break

        # now, starting from the Northern edge, go down ring by ring
        # for each ring, compute the best candidate for the left edge, using
        # the trigonometric formula from HealPIX library
        # go further left until the distance from disc center is too large
        # (note, the latter step makes this function slightly slower
        # than the HEALPix lib version, but gives exact results.)

        z0 = cos(disc_theta)
        cosrbig = cos(disc_size_rad)
        xa = 1. / sqrt((1 - z0) * (1 + z0))

        # dring is the difference between current ring and disc-center ring
        # dring > 0 means, that we are in the southern half of the disc
        dring -= 1
        while True:
            dring += 1
            ring = disc_ring + dring

            if ring > self._nrings:
                # need to sanitize, because the cdef'ed methods don't check
                break

            self._get_ring_info_small(ring, startpix, num_pix_in_ring, shifted)

            # check whether mid point (in ring) is within disc
            # if this is not true (and we are in the southern part already),
            # we break out of the loop, because no pixel is contained in disc
            # note, the mid pixel might be displaced, but then
            # the mirrored one will also be outside (in case)
            mid_pixel = startpix + num_pix_in_ring // 2
            self._pix2ang(mid_pixel, theta, phi)

            d = true_angular_distance(
                phi, HALFPI - theta,
                disc_phi, HALFPI - disc_theta,
                )

            if d > disc_size_rad and dring >= 0:
                break

            # some fancy trigonometry on the sphere to calculate the potential
            # edge of the disc from disc_radius and vertical offset
            z = cos(theta)
            x = (cosrbig - z*z0) * xa
            ysq = 1 - z*z - x*x
            dphi = atan2(sqrt(ysq), x)
            if ysq <= 0: dphi = 0.

            # ip is the difference in pixels between the left edge and phi=180d
            ip = <int64_t> (num_pix_in_ring * dphi / TWOPI)
            if ip < 0:
                ip = 0

            if ip > mid_pixel:
                ip = 0

            # search until angular distance is larger than disc_radius
            while mid_pixel - ip >= startpix:
                self._pix2ang(mid_pixel - ip, theta, phi)

                ip += 1
                if ip > mid_pixel:
                    break

                d = true_angular_distance(
                    phi, HALFPI - theta,
                    disc_phi, HALFPI - disc_theta,
                    )

                if d > disc_size_rad:
                    break

            # using symmetry, we don't need to compute the right edges
            # depending on a shift of the phi=180d pixel, we need to subtract 1
            ip -= 1
            for i in range(
                    mid_pixel - ip,
                    mid_pixel + ip + 1 - <uint64_t> shifted
                    ):
                disc_indices.push_back(i)

        return disc_indices

    def query_disc_phi180(
            self, double disc_size_rad, uint64_t disc_ring
            ):
        '''
        Return HEALPix indices of a disc around :math:`\\varphi=180^\\circ` for 'disc_ring'.

        This is a modified version of the HEALPix library `~healpy.query_disc`
        routine that returns the HEALPix indices of a disc around arbitrary
        coordinates. Here, it was modified to only work for coordinates having
        :math:`\\varphi=180^\\circ` and a :math:`\\vartheta` that coincides
        with the latitude of a HEALPix ring. These discs can be shifted along
        :math:`\\varphi` - which is faster than calling `~healpy.query_disc`
        if one cashes the :math:`\\varphi=180^\\circ` discs.

        Wrapper around the Cython-only (private) method `_query_disc_phi180`.

        Parameters
        ----------
        disc_size_rad : double
            Size of the disc in radians.
        disc_ring : unsigned 64-bit int
            HEALPix ring index of the requested disc center.

        Returns
        -------
        disc_indices : `~numpy.array` of unsigned 64-bit ints
            The HEALPix indices of the disc.
        '''

        cdef:
            uint64_t i
            vector[uint64_t] disc_indices_vec
            np.ndarray[uint64_t, ndim=1] disc_indices

        disc_indices_vec = self._query_disc_phi180(disc_size_rad, disc_ring)
        disc_indices = np.empty(disc_indices_vec.size(), dtype=UINT64)
        for i in range(disc_indices_vec.size()):
            disc_indices[i] = disc_indices_vec[i]

        return disc_indices

    cdef vector[uint64_t] _query_disc(
            self, double theta, double phi, double disc_size_rad
            ) noexcept nogil:
        '''
        Return hpx indices of a disc around :math:`(\\vartheta, \\varphi)`.

        This uses `_query_disc_phi180` and shifts the pixels along ring.

        Parameters
        ----------
        theta/phi : double
            Coordinates :math:`(\\vartheta, \\varphi)` in radians.
        disc_size_rad : double
            Size of the disc in radians.

        Returns
        -------
        disc_indices : std::vector[unsigned 64-bit int]
            The HEALPix indices of the disc.

        Notes
        -----
        1. This is a Cython-domain only method.
        '''

        cdef:
            uint64_t ring, i, j, hpxidx
            uint64_t startpix, num_pix_in_ring
            vector[uint64_t] disc_indices_vec
            bool shifted
            double dphi, dshift
            int64_t ishift, pix_in_ring

        hpxidx = self._ang2pix(theta, phi)
        ring = self._pix2ring(hpxidx)

        disc_indices_vec = self._query_disc_phi180(disc_size_rad, ring)

        for i in range(disc_indices_vec.size()):

            hpxidx = disc_indices_vec[i]
            ring = self._pix2ring(hpxidx)
            self._get_ring_info_small(ring, startpix, num_pix_in_ring, shifted)

            dphi = phi - PI
            pix_in_ring = hpxidx - startpix

            dshift = num_pix_in_ring * dphi / 2. / PI
            if dshift >= 0:
                ishift = <int64_t> (dshift+0.5)
            else:
                ishift = <int64_t> (dshift-0.5)

            pix_in_ring = pix_in_ring + ishift
            disc_indices_vec[i] = imod(pix_in_ring, num_pix_in_ring) + startpix

        return disc_indices_vec

    def query_disc(
            self, double theta, double phi, double disc_size_rad
            ):
        '''
        Return hpx indices of a disc around :math:`(\\vartheta, \\varphi)`.

        This uses `query_disc_phi180` and shifts the pixels along ring.

        Wrapper around the cython-only (private) method _query_disc.

        Parameters
        ----------
        theta/phi : double
            Coordinates :math:`(\\vartheta, \\varphi)` in radians.
        disc_size_rad : double
            Size of the disc in radians.

        Returns
        -------
        disc_indices : `~numpy.array` of unsigned 64-bit ints
            The HEALPix indices of the disc.
        '''

        cdef:
            uint64_t i
            vector[uint64_t] disc_indices_vec
            np.ndarray[uint64_t, ndim=1] disc_indices

        disc_indices_vec = self._query_disc(theta, phi, disc_size_rad)
        disc_indices = np.empty(disc_indices_vec.size(), dtype=UINT64)
        for i in range(disc_indices_vec.size()):
            disc_indices[i] = disc_indices_vec[i]

        return disc_indices
