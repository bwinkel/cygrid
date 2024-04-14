******
cygrid
******

- *Version:* 2.0
- *Authors:* Benjamin Winkel, Lars Flöer, Daniel Lenz
- *User manual:* `stable <https://bwinkel.github.io/cygrid/stable/>`__ |
  `developer <https://bwinkel.github.io/cygrid/latest/>`__

.. image:: https://img.shields.io/pypi/v/cygrid.svg
    :target: https://pypi.python.org/pypi/cygrid
    :alt: PyPI tag

.. image:: https://img.shields.io/badge/license-GPL-blue.svg
    :target: https://www.github.com/bwinkel/cygrid/blob/master/COPYING
    :alt: License

.. image:: http://img.shields.io/badge/arXiv-1604.06667-blue.svg
    :target: https://arxiv.org/abs/1604.06667
    :alt: Publication

.. image:: https://img.shields.io/badge/ascl-1606.003-blue.svg?colorB=262255
   :target: http://ascl.net/1606.003

Project Status
==============

.. image:: https://travis-ci.org/bwinkel/cygrid.svg?branch=master
    :target: https://travis-ci.org/bwinkel/cygrid
    :alt: cygrid's Travis CI Status

.. image:: https://ci.appveyor.com/api/projects/status/1ydk0hjf04t90aw5?svg=true
    :target: https://ci.appveyor.com/project/bwinkel/cygrid
    :alt: cygrid's AppVeyor CI Status

.. image:: https://coveralls.io/repos/github/bwinkel/cygrid/badge.svg?branch=master
    :target: https://coveralls.io/github/bwinkel/cygrid?branch=master
    :alt: cygrid's Coveralls Status

`Cygrid` is already used in several "production" systems, for example it was
utilized for two major 21-cm HI surveys, EBHIS and HI4PI. Nevertheless,
we cannot guarantee that it's completely bug-free. We kindly invite you to
use the library and we are grateful for feedback. Note, that work on the documentation is still ongoing.

Purpose
=======

`cygrid` allows to resample a number of spectra (or data points) to a regular
grid - a data cube - using any valid astronomical FITS/WCS projection (see
http://docs.astropy.org/en/stable/wcs/).

The method is a based on serialized convolution with finite gridding kernels.
Currently, only Gaussian (radial-symmetric or elliptical) kernels are provided
(which has the drawback of slight degradation of the effective resolution).
The algorithm has very small memory footprint, allows easy parallelization,
and is very fast.

A detailed description of the algorithm is given in `Winkel, Lenz & Flöer
(2016) <http://adsabs.harvard.edu/abs/2016A%26A...591A..12W>`_, which we
kindly ask to be used as reference if you found `cygrid` useful for your
research.

Features
========

- Supports any WCS projection system as target.
- Conserves flux.
- Low memory footprint.
- Scales very well on multi-processor/core platforms.

Installation
============

We highly recommend to use `cygrid` with the `Anaconda Python distribution <https://www.anaconda.com/>`_, in which
case installiation is as easy as ::

    conda install -c conda-forge cygrid

Otherwise, you should install cygrid via `pip`::

    python -m pip install cygrid

The installation is also possible from source, but you'll need a C++
compiler. Download the tar.gz-file, extract (or clone from GitHub) and
execute (in project directory)::

    python -m pip install .

Dependencies
------------

We kept the dependencies as minimal as possible. The following packages are
required:

- `Python 3.8` or later (`cygrid` versions prior to v1.0 support `Python 2.7`)
- `NumPy 1.13` or later
- `Cython 3` or later (if you want to build `cygrid` yourself)
- `Astropy 4.0` or later

(Older versions of these libraries may work, but we didn't test this!)

If you want to run the notebooks yourself, you will also need the Jupyter
server, matplotlib and wcsaxes packages. To run the tests, you'll need HealPy.

Note, for compiling the C-extension, openmp is used for parallelization and
some C++11 language features are necessary. If you use gcc, for example, you
need at least version 4.8 otherwise the setup-script will fail. It is highly
recommended to use Anaconda, which offers the proper compilers for many
platforms.

Usage
=====

Minimal example
---------------

Using `cygrid` is extremely simple. Just define a FITS header (with valid
WCS), define gridding kernel and run the grid function:

.. code-block:: python

    from astropy.io import fits
    import cygrid

    # read-in data
    glon, glat, signal = get_data(...)

    # define target FITS/WCS header
    header = {
        'NAXIS': 3,
        'NAXIS1': 101,
        'NAXIS2': 101,
        'NAXIS3': 1024,
        'CTYPE1': 'GLON-SFL',
        'CTYPE2': 'GLAT-SFL',
        'CDELT1': -0.1,
        'CDELT2': 0.1,
        'CRPIX1': 51,
        'CRPIX2': 51,
        'CRVAL1': 12.345,
        'CRVAL2': 3.14,
        }

    # prepare gridder
    kernelsize_sigma = 0.2

    kernel_type = 'gauss1d'
    kernel_params = (kernelsize_sigma, )
    kernel_support = 3 * kernelsize_sigma
    hpx_maxres = kernelsize_sigma / 2

    mygridder = cygrid.WcsGrid(header)
    mygridder.set_kernel(
        kernel_type,
        kernel_params,
        kernel_support,
        hpx_maxres
        )

    # do the gridding
    mygridder.grid(glon, glat, signal)

    # query result and store to disk
    data_cube = mygridder.get_datacube()
    fits.writeto(
        'example.fits',
        header=header, data=data_cube
        )


More use-cases and tutorials
----------------------------

Check out the `user manual <https://bwinkel.github.io/cygrid/latest/>`_ or the
`Jupyter tutorial notebooks <https://github.com/bwinkel/cygrid/tree/master/notebooks>`_
in the repository for further examples of how to use `cygrid`. Note that you
can only view the notebooks on GitHub, if you want to edit something
it is necessary to clone the repository or download a notebook to run it on
your machine.

Who do I talk to?
=================

If you encounter any problems or have questions, do not hesitate to raise an
issue or make a pull request. Moreover, you can contact the devs directly:

- <bwinkel@mpifr.de>
- <mail@daniellenz.org>


Preferred citation method
=========================

Please cite our `paper <http://adsabs.harvard.edu/abs/2016A%26A...591A..12W>`_
if you use `cygrid` for your projects.

.. code-block:: latex

    @ARTICLE{2016A&A...591A..12W,
        author = {{Winkel}, B. and {Lenz}, D. and {Fl{\"o}er}, L.},
        title = "{Cygrid: A fast Cython-powered convolution-based gridding module for Python}",
        journal = {\aap},
        archivePrefix = "arXiv",
        eprint = {1604.06667},
        primaryClass = "astro-ph.IM",
        keywords = {methods: numerical, techniques: image processing},
        year = 2016,
        month = jun,
        volume = 591,
        eid = {A12},
        pages = {A12},
        doi = {10.1051/0004-6361/201628475},
        adsurl = {http://adsabs.harvard.edu/abs/2016A%26A...591A..12W},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
