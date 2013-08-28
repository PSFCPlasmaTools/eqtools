from distutils.core import setup as pysetup
from numpy.distutils.core import setup,Extention

tricub = Extension('_tricub',['_tricub.pyf','_tricub.c'])

setup(ext_modules = [tricub,])


pysetup(name='EqTools',
        version='1.0',
        description='Tokamak Flux mapping utility',
        author=['Mark Chilenski','Ian Faust','John Walk']
        author_email='tbd@psfc.mit.edu'
        url='https://github.com/PSFCPlasmaTools/EqTools/'
        packages='eqtools'
        )  
