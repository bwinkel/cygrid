.. user-manual:

******************
Cygrid quick tour
******************

.. currentmodule:: cygrid

The following (in-complete) code snippet demonstrates, how one would use
`~cygrid` to grid raw data, sampled at (possibly irregular) positions,
`(lon, lat)`, with values `rawdata` into a FITS/WCS image::

    from astropy.io import fits
    import matplotlib.pyplot as plt
    import cygrid

    # read-in data; assuming rawdata is a 2D array (i.e. a list of spectra)
    lon, lat, rawdata = get_data()

    # define target FITS/WCS header
    header = create_fits_header()  # assume a 3D image, need only spatial part

    # prepare gridder
    kernelsize_sigma = 0.2

    kernel_type = 'gauss1d'
    kernel_params = (kernelsize_sigma, )  # must be a tuple
    kernel_support = 3 * kernelsize_sigma
    hpx_maxres = kernelsize_sigma / 2

    mygridder = cygrid.WcsGrid(header)
    mygridder.set_kernel(
        kernel_type,
        kernel_params,
        kernel_support,
        hpx_maxres,
        )

    # do the actual gridding
    mygridder.grid(lon, lat, rawdata)

    # query result and store to disk ...
    data_cube = mygridder.get_datacube()
    fits.writeto('example.fits', header=header, data=data_cube)

    # ... or plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=WCS(header).celestial)
    # plot first plane of data cube
    ax.imshow(
        data_cube[0], origin='lower', interpolation='nearest'
        )
    plt.show()

Here, we have omitted the code to read-in the raw data samples, `lon`, `lat`,
and `rawdata`. The `lon` and `lat` need to be one-dimensional arrays (or
lists), while in this example `rawdata` has two dimensions (first dimension
needs to have the same length as `lon` and `lat`, second dimension is the
number of spectral channels). It would also be possible that `rawdata` is a
1D array, in this case the output `data_cube` would be 2D, i.e., a map.
Likewise, the code to setup the target FITS/WCS header is not explicitly
shown. Anything that `~astropy.WCS` would accept to create a spatial WCS is
sufficient.

The next part in the little program is about setting up the gridder class
(`~cygrid.WCSGrid`). The constructor only needs the aforementioned WCS header.
But before any gridding job would work, it is necessary to define the
weighting kernel parameters. Details about a proper choice of these values
is contained in the user manual (:ref:`kernel-parameters-label`). In most
cases you will probably use a radial-symmetric Gaussian (therefore,
`kernel_type = 'gauss1d'`) with a certain width (`kernelsize_sigma`).
The `kernel_support` parameter defines out to which angular distance the
kernel is computed. For Gaussians :math:`3\ldots4\sigma` is sufficient.
The `hpx_maxres` is a very technical parameter. It tells `~cygrid`, how fine
the internal HEALPix cache needs to be. We recommend to set this parameter to
half the kernel size.

The gridding itself is done by calling the `~cygrid.WCSGrid.grid` method.
Depending on the size of the target FITS image and the number of raw data
samples, this might take a while. Once the gridding is finished, you
can obtain the gridded data array with the `~cygrid.WCSGrid.get_datacube`
method and do whatever you want with it, e.g., store it to a FITS file.

.. note::

    This "minimal example" is also contained in a fully-working `Jupyter
    tutorial notebook <https://github.com/bwinkel/cygrid/blob/master/notebooks/01_minimal_example.ipynb>`_.


Gridding large data sets
========================

If you have a huge amount of raw data to grid, which might not fit into your
computer's memory, `~cygrid` allows you to process the data in smaller
chunks::


    mygridder = cygrid.WcsGrid(header)
    mygridder.set_kernel(...)

    for chunk in chunks:
        lon, lat, rawdata = get_data(chunk)
        mygridder.grid(lon, lat, rawdata)

    data_cube = mygridder.get_datacube()

More information can be found in the user manual (:ref:`serialization-label`).



Sight-line gridding
===================

Sometimes, it can be useful to re-sample the input raw data set for a given
list of coordinate pairs, e.g., if you want to grid to a funky coordinate
system or projection, which is not supported by FITS/WCS. For this, one can
use the `~cygrid.SlGrid` class::

    mygridder = cygrid.SlGrid(target_lon, target_lat)
    mygridder.set_kernel(...)

    mygridder.grid(input_lon, input_lat, input_signal)

    target_signal = gridder.get_datacube()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(target_lon, target_lat, c=target_signal)
    plt.show()
