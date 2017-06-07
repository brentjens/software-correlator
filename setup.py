#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
from softwarecorrelator import __version__

ext_modules = [
    Extension(
        "softwarecorrelator.sumthreshold",
        ["softwarecorrelator/sumthreshold.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    )
]


setup(name='software-correlator',
      version=__version__,
      description='Python based software correlator for LOFAR complex voltage HDF5 files',
      author='Michiel Brentjens',
      author_email='brentjens@astron.nl',
      url='https://www.astron.nl/~brentjens/',
      packages=['softwarecorrelator'],
      scripts=['bin/cvqa', 'bin/cvcorr'],
      ext_modules=cythonize(ext_modules, force=True),
     )
