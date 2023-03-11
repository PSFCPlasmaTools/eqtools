from numpy.distutils.core import setup, Extension

tricub = Extension(
    'eqtools._tricub', ['eqtools/_tricub.pyf', 'eqtools/_tricub.c']
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
