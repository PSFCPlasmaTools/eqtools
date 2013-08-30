from numpy.distutils.core import setup,Extension
from distutils.core import setup as pysetup
tricub = Extension('eqtools._tricub',['eqtools/_tricub.pyf','eqtools/_tricub.c'])

pysetup(name='eqtools',
        version='1.0',
        description='Tokamak Flux mapping utility',
        author=['Mark Chilenski','Ian Faust','John Walk'],
        author_email='tbd@psfc.mit.edu',
        url='https://github.com/PSFCPlasmaTools/eqtools/',
        packages=['eqtools',],
        )
setup(ext_modules = [tricub,])
