import numpy
from setuptools import Extension, setup

tricub = Extension(
    name='eqtools._tricub',
    sources=['eqtools/_tricubmodule.c', 'eqtools/_tricub.c'],
    include_dirs=[numpy.get_include(), './eqtools'],
)

setup(
    name='eqtools',
    version='1.4.0',
    packages=['eqtools'],
    install_requires=['scipy', 'numpy', 'matplotlib'],
    author=['Mark Chilenski', 'Ian Faust', 'John Walk'],
    author_email='psfcplasmatools@mit.edu',
    url='https://github.com/PSFCPlasmaTools/eqtools/',
    description='Python tools for magnetic equilibria in tokamak plasmas',
    long_description=open('README.md', 'r').read(),
    ext_modules=[tricub],
    license='GPL'
)
