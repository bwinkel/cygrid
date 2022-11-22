2.0.3 (2022-11-22)
=======================
This is a pure maintenance release providing cygrid for newer Python versions.

2.0.2 (2021-01-25)
=======================

- Python3.5 support ceased (it may still work, but is not tested anymore)

Bugfixes
~~~~~~~~~~
- Fix a typo (conversion between FWHM and sigma) in the manual and notebooks.
  Thanks to Yoko Okada for pointing this out.

2.0.1 (2020-10-19)
=======================

Bugfixes
~~~~~~~~~~
- Fix (stable) HTML manual

2.0.0 (2020-10-17)
=======================

- Major overhaul of the handling of array shapes. No need to manually
  fiddle with FITS headers (`NAXIS3` key) anymore. `cygrid` will now do
  everything automatically behind the curtain. [#12]

Bugfixes
~~~~~~~~~~

- Fix byteorder of input data/weights if not native. (Apparently,
  astroquery.skyview can deliver non-native byteorders.) [#8dfc]


1.0.1 (2019-01-19)
=======================

Packaging/Installation
~~~~~~~~~~~~~~~~~~~~~~
- Use clang/LLVM for the compilation on MacOS and use this path to generate
  MacOS binary wheels for PyPI. #5b32

Documentation
~~~~~~~~~~~~~
- Add bibtex item to README. #74c7
- Fix some URLs in the documention. #a42b


1.0.0 (2019-01-18)
=======================

Other
-----
- Add a user manual to supplement the existing tutorial notebooks.
- Now using Astropy package template.
