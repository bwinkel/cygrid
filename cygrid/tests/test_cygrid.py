#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.utils.misc import NumpyRNGContext
from astropy.utils.data import get_pkg_data_filename
from .. import cygrid


class TestWcsGrid:

    def setup_method(self):

        mapcenter = (131., 50.)
        mapsize = (1., 1.)  # degrees
        self.beamsize_fwhm = 0.05  # degrees

        avg_num_pixels_x = 5 * mapsize[0] / self.beamsize_fwhm
        avg_num_pixels_y = 5 * mapsize[1] / self.beamsize_fwhm

        scale = np.cos(np.radians(mapcenter[1]))

        with NumpyRNGContext(0):
            self.xcoords = np.random.uniform(
                mapcenter[0] - mapsize[0] / 2. / scale,
                mapcenter[0] + mapsize[0] / 2. / scale,
                int(avg_num_pixels_x * avg_num_pixels_y),
                ).astype(np.float64)
            self.ycoords = np.random.uniform(
                mapcenter[1] - mapsize[1] / 2.,
                mapcenter[1] + mapsize[1] / 2.,
                int(avg_num_pixels_x * avg_num_pixels_y),
                ).astype(np.float64)
            self.signal = np.random.normal(0., .01, len(self.xcoords))

        self.signal2 = np.column_stack([self.signal, self.signal ** 2])
        self.signal3 = np.column_stack([
            self.signal, self.signal ** 2,
            self.signal ** 3, self.signal ** 4
            ]).reshape((-1, 2, 2))
        pixsize = self.beamsize_fwhm / 3.
        dnaxis1 = int(mapsize[0] / pixsize + 0.5)
        dnaxis2 = int(mapsize[1] / pixsize + 0.5)
        self.yx_shape = dnaxis2, dnaxis1
        projection = 'SIN'

        self.header = {
            'NAXIS': 2,
            'NAXIS1': dnaxis1,
            'NAXIS2': dnaxis2,
            'CTYPE1': 'RA---{}'.format(projection),
            'CTYPE2': 'DEC--{}'.format(projection),
            'CUNIT1': 'deg',
            'CUNIT2': 'deg',
            'CDELT1': -pixsize,
            'CDELT2': pixsize,
            'CRPIX1': dnaxis1 / 2.,
            'CRPIX2': dnaxis2 / 2.,
            'CRVAL1': mapcenter[0],
            'CRVAL2': mapcenter[1],
            }

        kernelsize_fwhm = self.beamsize_fwhm / 2
        kernelsize_sigma = kernelsize_fwhm / 2.35
        support_radius = 3. * kernelsize_sigma
        hpx_min_res = kernelsize_sigma / 2.

        self.kernel_args = (
            'gauss1d',
            (kernelsize_sigma, ),
            support_radius,
            hpx_min_res,
            )

        self.test_maps = np.load(get_pkg_data_filename(
            'data/cygrid_test_maps.npy'
            ))

    def teardown_method(self):

        pass

    def test_gridding2d(self):

        mygridder = cygrid.WcsGrid(self.header)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, self.signal)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_maps[0, 0],
            atol=1.e-6
            )

    def test_gridding3d(self):

        mygridder = cygrid.WcsGrid(self.header)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, self.signal2)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_maps[0],
            atol=1.e-6
            )

    def test_gridding4d(self):

        mygridder = cygrid.WcsGrid(self.header)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, self.signal3)

        if False:
            np.save('/tmp/cygrid_test_maps.npy', mygridder.get_datacube())

        assert_allclose(
            mygridder.get_datacube(),
            self.test_maps,
            atol=1.e-6
            )

    def test_kernel_not_set_error(self):

        mygridder = cygrid.WcsGrid(self.header)

        with pytest.raises(RuntimeError):
            mygridder.grid(self.xcoords, self.ycoords, self.signal)

    def test_shape_error(self):
        '''
        Test for various shape errors
        '''

        mygridder = cygrid.WcsGrid(self.header)
        mygridder.set_kernel(*self.kernel_args)

        with pytest.raises(cygrid.ShapeError):
            mygridder.grid(
                self.xcoords, self.ycoords[:, np.newaxis], self.signal
                )

        with pytest.raises(cygrid.ShapeError):
            # this should not work, as the second call to grid has
            # incompatible shape
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal
                )
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal[:, np.newaxis]
                )

        # now test with user-provided data cube
        dcube = np.zeros_like(self.test_maps[0, 0])
        mygridder = cygrid.WcsGrid(self.header, datacube=dcube)
        mygridder.set_kernel(*self.kernel_args)

        with pytest.raises(cygrid.ShapeError):
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal[:, np.newaxis]
                )

    def test_dtype_warning(self):
        '''
        Test for dtype warning

        In fact, at the moment there is only one possible warning, as
        everything else should be properly casted, etc.
        '''

        dcube = np.zeros_like(self.test_maps[0, 0])  # float32
        mygridder = cygrid.WcsGrid(
            self.header, datacube=dcube, dtype=np.float64
            )
        mygridder.set_kernel(*self.kernel_args)

        with pytest.warns(UserWarning):
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal
                )

    def test_byteorder_warning(self):
        '''
        Test for byteorder warning

        Apparently astroquery.skyview can return wrong (non-native) byteorder.
        '''

        dcube = np.zeros_like(self.test_maps[0, 0])  # float32
        mygridder = cygrid.WcsGrid(
            self.header, datacube=dcube, dtype=np.float64
            )
        mygridder.set_kernel(*self.kernel_args)

        signal_swapped = self.signal.byteswap().newbyteorder()

        with pytest.warns(UserWarning):
            mygridder.grid(
                self.xcoords, self.ycoords, signal_swapped
                )

    def test_c_contiguous(self):
        '''
        Cygrid should autocast to C-contiguous if necessary and raise
        an error if user-provided datacube is not C-contiguous
        '''

        signal2_f_cont = np.require(self.signal2, self.signal2.dtype, 'F')

        mygridder = cygrid.WcsGrid(self.header)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, signal2_f_cont)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_maps[0],
            atol=1.e-6
            )

        dcube_f_cont = np.require(np.zeros_like(self.test_maps[0, 0]), 'F')
        mygridder = cygrid.WcsGrid(self.header, datacube=dcube_f_cont)
        mygridder.set_kernel(*self.kernel_args)

        with pytest.raises(TypeError):
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal
                )

    def test_user_datacube_memorycell(self):
        '''
        If user provides a data cube, it must be made sure that cygrid
        writes to the correct memory cell (even though, the
        `get_unweighted_datacube` method can have a different object id
        due to internal reshaping)
        '''

        dcube = np.zeros_like(self.test_maps[0, 0])  # float64
        mygridder = cygrid.WcsGrid(self.header, datacube=dcube)
        mygridder.set_kernel(*self.kernel_args)
        mygridder.grid(self.xcoords, self.ycoords, self.signal)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_maps[0, 0],
            atol=1.e-6
            )
        assert_allclose(
            mygridder.get_unweighted_datacube(),
            dcube,
            atol=1.e-6
            )


class TestSlGrid:

    def setup_method(self):

        mapcenter = (131., 50.)
        mapsize = (1., 1.)  # degrees
        self.beamsize_fwhm = 0.05  # degrees

        avg_num_pixels_x = 5 * mapsize[0] / self.beamsize_fwhm
        avg_num_pixels_y = 5 * mapsize[1] / self.beamsize_fwhm

        scale = np.cos(np.radians(mapcenter[1]))

        with NumpyRNGContext(0):
            self.xcoords = np.random.uniform(
                mapcenter[0] - mapsize[0] / 2. / scale,
                mapcenter[0] + mapsize[0] / 2. / scale,
                int(avg_num_pixels_x * avg_num_pixels_y),
                ).astype(np.float64)
            self.ycoords = np.random.uniform(
                mapcenter[1] - mapsize[1] / 2.,
                mapcenter[1] + mapsize[1] / 2.,
                int(avg_num_pixels_x * avg_num_pixels_y),
                ).astype(np.float64)
            self.signal = np.random.normal(0., .01, len(self.xcoords))

            self.target_x = np.random.uniform(
                mapcenter[0] - mapsize[0] / 2. / scale,
                mapcenter[0] + mapsize[0] / 2. / scale,
                1000,
                ).astype(np.float64)
            self.target_y = np.random.uniform(
                mapcenter[1] - mapsize[1] / 2.,
                mapcenter[1] + mapsize[1] / 2.,
                1000,
                ).astype(np.float64)

        self.signal2 = np.column_stack([self.signal, self.signal ** 2])
        self.signal3 = np.column_stack([
            self.signal, self.signal ** 2,
            self.signal ** 3, self.signal ** 4
            ]).reshape((-1, 2, 2))

        kernelsize_fwhm = self.beamsize_fwhm / 2
        kernelsize_sigma = kernelsize_fwhm / 2.35
        support_radius = 3. * kernelsize_sigma
        hpx_min_res = kernelsize_sigma / 2.

        self.kernel_args = (
            'gauss1d',
            (0.5 / kernelsize_sigma ** 2,),
            support_radius,
            hpx_min_res,
            )

        self.test_sls = np.load(get_pkg_data_filename(
            'data/cygrid_test_sightlines.npy'
            ))

    def teardown_method(self):

        pass

    def test_gridding1d(self):

        mygridder = cygrid.SlGrid(self.target_x, self.target_y)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, self.signal)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_sls[0, 0],
            atol=1.e-6
            )

    def test_gridding2d(self):

        mygridder = cygrid.SlGrid(self.target_x, self.target_y)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, self.signal2)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_sls[0],
            atol=1.e-6
            )

    def test_gridding3d(self):

        mygridder = cygrid.SlGrid(self.target_x, self.target_y)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, self.signal3)

        if False:
            np.save(
                '/tmp/cygrid_test_sightlines.npy', mygridder.get_datacube()
                )

        assert_allclose(
            mygridder.get_datacube(),
            self.test_sls,
            atol=1.e-6
            )

    def test_kernel_not_set_error(self):

        mygridder = cygrid.SlGrid(self.target_x, self.target_y)

        with pytest.raises(RuntimeError):
            mygridder.grid(self.xcoords, self.ycoords, self.signal)

    def test_shape_error(self):
        '''
        Test for various shape errors
        '''

        mygridder = cygrid.SlGrid(self.target_x, self.target_y)
        mygridder.set_kernel(*self.kernel_args)

        with pytest.raises(cygrid.ShapeError):
            mygridder.grid(
                self.xcoords, self.ycoords[:, np.newaxis], self.signal
                )

        with pytest.raises(cygrid.ShapeError):
            # this should not work, as the second call to grid has
            # incompatible shape
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal
                )
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal[:, np.newaxis]
                )

        # now test with user-provided data cube
        dcube = np.zeros_like(self.test_sls[0])
        mygridder = cygrid.SlGrid(
            self.target_x, self.target_y, datacube=dcube
            )
        mygridder.set_kernel(*self.kernel_args)

        with pytest.raises(cygrid.ShapeError):
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal[:, np.newaxis]
                )

    def test_dtype_warning(self):
        '''
        Test for dtype warning

        In fact, at the moment there is only one possible warning, as
        everything else should be properly casted, etc.
        '''

        dcube = np.zeros_like(self.test_sls[0, 0])  # float32
        mygridder = cygrid.SlGrid(
            self.target_x, self.target_y,
            datacube=dcube, dtype=np.float64
            )
        mygridder.set_kernel(*self.kernel_args)

        with pytest.warns(UserWarning):
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal
                )

    def test_byteorder_warning(self):
        '''
        Test for byteorder warning

        Apparently astroquery.skyview can return wrong (non-native) byteorder.
        '''

        dcube = np.zeros_like(self.test_sls[0, 0])  # float32
        mygridder = cygrid.SlGrid(
            self.target_x, self.target_y,
            datacube=dcube, dtype=np.float64
            )
        mygridder.set_kernel(*self.kernel_args)

        signal_swapped = self.signal.byteswap().newbyteorder()

        with pytest.warns(UserWarning):
            mygridder.grid(
                self.xcoords, self.ycoords, signal_swapped
                )
    def test_c_contiguous(self):
        '''
        Cygrid should autocast to C-contiguous if necessary and raise
        an error if user-provided datacube is not C-contiguous
        '''

        signal2_f_cont = np.require(self.signal2, self.signal2.dtype, 'F')

        mygridder = cygrid.SlGrid(self.target_x, self.target_y)
        mygridder.set_kernel(*self.kernel_args)

        mygridder.grid(self.xcoords, self.ycoords, signal2_f_cont)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_sls[0],
            atol=1.e-6
            )

        dcube_f_cont = np.require(np.zeros_like(self.test_sls[0, 0]), 'F')
        mygridder = cygrid.SlGrid(
            self.target_x, self.target_y, datacube=dcube_f_cont
            )
        mygridder.set_kernel(*self.kernel_args)

        with pytest.raises(TypeError):
            mygridder.grid(
                self.xcoords, self.ycoords, self.signal
                )

    def test_user_datacube_memorycell(self):
        '''
        If user provides a data cube, it must be made sure that cygrid
        writes to the correct memory cell (even though, the
        `get_unweighted_datacube` method can have a different object id
        due to internal reshaping)
        '''

        dcube = np.zeros_like(self.test_sls[0, 0])  # float64
        mygridder = cygrid.SlGrid(
            self.target_x, self.target_y, datacube=dcube
            )
        mygridder.set_kernel(*self.kernel_args)
        mygridder.grid(self.xcoords, self.ycoords, self.signal)

        assert_allclose(
            mygridder.get_datacube(),
            self.test_sls[0, 0],
            atol=1.e-6
            )
        assert_allclose(
            mygridder.get_unweighted_datacube(),
            dcube,
            atol=1.e-6
            )
