from numpy.distutils.core import setup,Extension

tricub = Extension('eqtools._tricub',['eqtools/_tricub.pyf','eqtools/_tricub.c'])

setup(name='eqtools',
        version='1.0',
        description='Tokamak Flux mapping utility',
        author=['Mark Chilenski','Ian Faust','John Walk'],
        author_email='tbd@psfc.mit.edu',
        url='https://github.com/PSFCPlasmaTools/EqTools/',
        packages=['eqtools',],
        ext_modules = [tricub,]
        )  
