#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.utils.misc import NumpyRNGContext
import importlib
from ..healpix import Healpix


# skip over healpy related tests (currently all), if package not present:
skip_healpy = pytest.mark.skipif(
    importlib.util.find_spec('healpy') is None,
    reason='"healpy" package not installed'
    )


# for order in range(0, 12):
#     nside = 2 ** order
#     hpx = cygrid.Healpix(nside)
#     print hpx.nside, hpx.npface, hpx.npix, hpx.nrings, hpx.omega, hpx.order, hpx.resolution

hpx_props = [
    (1, 1, 12, 3, 1.0471975512, 0, 1.02332670795),
    (2, 4, 48, 7, 0.261799387799, 1, 0.511663353973),
    (4, 16, 192, 15, 0.0654498469498, 2, 0.255831676987),
    (8, 64, 768, 31, 0.0163624617374, 3, 0.127915838493),
    (16, 256, 3072, 63, 0.00409061543436, 4, 0.0639579192467),
    (32, 1024, 12288, 127, 0.00102265385859, 5, 0.0319789596233),
    (64, 4096, 49152, 255, 0.000255663464648, 6, 0.0159894798117),
    (128, 16384, 196608, 511, 6.39158661619e-05, 7, 0.00799473990583),
    (256, 65536, 786432, 1023, 1.59789665405e-05, 8, 0.00399736995292),
    (512, 262144, 3145728, 2047, 3.99474163512e-06, 9, 0.00199868497646),
    (1024, 1048576, 12582912, 4095, 9.9868540878e-07, 10, 0.000999342488229),
    (2048, 4194304, 50331648, 8191, 2.49671352195e-07, 11, 0.000499671244114),
    ]


class TestHealpix:

    def setup_method(self):

        pass
        # self.nside = 1024
        # self.hpx = Healpix(self.nside)

    def teardown_method(self):

        pass

    def test_properties(self):

        eps = 1.e-10

        for nside, npface, npix, nrings, omega, order, reso in hpx_props:

            hpx = Healpix(nside)

            assert hpx.nside == nside
            assert hpx.npface == npface
            assert hpx.npix == npix
            assert hpx.nrings == nrings
            assert abs(hpx.omega - omega) < eps
            assert hpx.order == order
            assert abs(hpx.resolution - reso) < eps

    def test_get_ring_info_small(self):

        hpx = Healpix(64)

        assert hpx.get_ring_info_small(1) == (0, 4, True)
        assert hpx.get_ring_info_small(10) == (180, 40, True)
        assert hpx.get_ring_info_small(127) == (24192, 256, False)
        assert hpx.get_ring_info_small(128) == (24448, 256, True)
        assert hpx.get_ring_info_small(255) == (49148, 4, True)

        with pytest.raises(ValueError):
            hpx.get_ring_info_small(0)
        with pytest.raises(ValueError):
            hpx.get_ring_info_small(256)

    @skip_healpy
    def test_pix2ring(self):

        import healpy as hp

        with NumpyRNGContext(12345):

            nside = 1024
            hpx = Healpix(nside)
            npix = hpx.npix

            hpx_idx = np.random.randint(0, npix - 1, 10000)

            assert_equal(
                hp.pix2ring(nside, hpx_idx),
                hpx.pix2ring_many(hpx_idx)
                )

    @skip_healpy
    def test_ang2pix(self):

        import healpy as hp

        with NumpyRNGContext(12345):

            nside = 1024
            hpx = Healpix(nside)

            theta = np.random.uniform(0, np.pi, 10000)
            phi = np.random.uniform(0, 2 * np.pi, 10000)

            assert_equal(
                hp.ang2pix(nside, theta, phi),
                hpx.ang2pix_many(theta, phi)
                )

    @skip_healpy
    def test_pix2ang(self):

        import healpy as hp

        with NumpyRNGContext(12345):

            nside = 1024
            hpx = Healpix(nside)
            npix = hpx.npix

            hpx_idx = np.random.randint(0, npix - 1, 10000)

            assert_allclose(
                hp.pix2ang(nside, hpx_idx),
                hpx.pix2ang_many(hpx_idx)
                )

    @skip_healpy
    def test_query_disc(self):

        import healpy as hp

        with NumpyRNGContext(12345):

            nside = 1024
            hpx = Healpix(nside)
            disc_size = np.radians(0.5 / 60.)

            thetas = np.random.uniform(0, np.pi, 10)
            phis = np.random.uniform(0, 2 * np.pi, 10)

            for theta, phi in zip(thetas, phis):

                _vec = hp.ang2vec(theta, phi)
                hp_disc = hp.query_disc(nside, _vec, disc_size)
                assert_equal(
                    hp_disc,
                    hpx.query_disc(theta, phi, disc_size)
                    )
