
:tocdepth: 3

#####################
Cygrid Documentation
#####################

Cygrid allows to resample a number of spectra (or data points) to a regular
grid - a data cube - using any valid astronomical FITS/WCS projection. The
method is a based on serialized convolution with finite gridding kernels.
The underlying algorithm has very small memory footprint, was designed to allow parallelization, and is very fast.

The cygrid package is available for Linux, Windows, and MacOS operating
systems. To improve computation speed, the `OpenMP technology
<https://www.openmp.org/>`_ is used. Therefore, a suitable C++ compiler must
be installed on your system, if you build from source. For convenience, we
also provide packages for the `Anaconda Python distribution
<https://www.anaconda.com/>`_.

***************
Getting Started
***************

.. toctree::
   :maxdepth: 1

   install
   importing_cygrid
   quick_tour

******************
User Documentation
******************

.. toctree::
   :maxdepth: 1

   user_manual
   Tutorials (Jupyter notebooks) <http://nbviewer.jupyter.org/github/bwinkel/cygrid/blob/master/notebooks/>


***************
License
***************

.. toctree::
   :maxdepth: 1

   license


*************************
Preferred citation method
*************************

.. toctree::
   :maxdepth: 1

Please cite our `paper <http://adsabs.harvard.edu/abs/2016A%26A...591A..12W>`_ if you use `~cygrid` for your projects.

***************
Acknowledgments
***************

This code makes use of the excellent work provided by the
`Astropy <http://www.astropy.org/>`__ community. Cygrid uses the Astropy package and also the
`Astropy Package Template <https://github.com/astropy/package-template>`__
for the packaging.
