.. eqtools documentation master file, created by
   sphinx-quickstart on Wed Sep  4 17:55:14 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

eqtools: Tools for interacting with magnetic equilibria
=======================================================

Homepage: https://github.com/PSFCPlasmaTools/eqtools

Overview
--------

:py:mod:`eqtools` is a Python package for working with magnetic equilibrium reconstructions from magnetic plasma confinement devices. At present, interfaces exist for data from the Alcator C-Mod and NSTX MDSplus trees as well as eqdsk a- and g-files. :py:mod:`eqtools` is designed to be flexible and extensible such that it can become a uniform interface to perform mapping operations and accessing equilibrium data for any magnetic confinement device, regardless of how the data are accessed.

The main class of :py:mod:`eqtools` is the :py:class:`~eqtools.core.Equilibrium`, which contains all of the coordinate mapping functions as well as templates for methods to fetch data (primarily dictated to the quantities computed by EFIT). Subclasses such as :py:class:`~eqtools.EFIT.EFITTree`, :py:class:`~eqtools.CModEFIT.CModEFITTree`, :py:class:`~eqtools.NSTXEFIT.NSTXEFITTree` and :py:class:`~eqtools.eqdskreader.EqdskReader` implement specific methods to access the data and convert it to the form needed for the routines in :py:class:`~eqtools.core.Equilibrium`. These classes are smart about caching intermediate results, so you will get a performance boost by using the same instance throughout your analysis of a given shot.

Installation
------------

The easiest way to install the latest release version is with `pip`::
    
    pip install eqtools

To install from source, uncompress the source files and, from the directory containing `setup.py`, run the following command::
    
    python setup.py install

Or, to build in place, run::
    
    python setup.py build_ext --inplace

Tutorial: Performing Coordinate Transforms on Alcator C-Mod Data
----------------------------------------------------------------

The basic class for manipulating EFIT results stored in the Alcator C-Mod MDSplus tree is :py:class:`~eqtools.CModEFIT.CModEFITTree`. To load the data from a specific shot, simply create the :py:class:`~eqtools.CModEFIT.CModEFITTree` object with the shot number as the argument::

    e = eqtools.CModEFITTree(1140729030)

The default EFIT to use is "ANALYSIS." If you want to use a different tree, such as "EFIT20," then you simply set this with the `tree` keyword::
    
    e = eqtools.CModEFITTree(1140729030, tree='EFIT20')

:py:mod:`eqtools` understands units. The default is to convert all lengths to meters (whereas quantities in the tree are inconsistent -- some are meters, some centimeters). If you want to specify a different default unit, use the `length_unit` keyword::

    e = eqtools.CModEFITTree(1140729030, length_unit='cm')

Once this is loaded, you can access the data you would normally have to pull from specific nodes in the tree using convenient getter methods. For instance, to get the elongation as a function of time, you can run::

    kappa = e.getElongation()

The timebase used for quantities like this is accessed with::

    t = e.getTimeBase()

For length/area/volume quantities, :py:mod:`eqtools` understands units. The default is to return in whatever units you specified when creating the :py:class:`~eqtools.CModEFIT.CModEFITTree`, but you can override this with the `length_unit` keyword. For instance, to get the vertical position of the magnetic axis in mm, you can run::

    Z_mag = e.getMagZ(length_unit='mm')

:py:mod:`eqtools` can map from almost any coordinate to any common flux surface label. For instance, say you want to know what the square root of normalized toroidal flux corresponding to a normalized flux surface volume of 0.5 is at t=1.0s. You can simply call::

    rho = e.volnorm2phinorm(0.5, 1.0, sqrt=True)

If a list of times is provided, the default behavior is to evaluate all of the points to be converted at each of the times. So, to follow the mapping of normalized poloidal flux values [0.1, 0.5, 1.0] to outboard midplane major radius at time points [1.0, 1.25, 1.5, 1.75], you could call::

    psinorm = e.psinorm2rmid([0.1, 0.5, 1.0], [1.0, 1.25, 1.5, 1.75])

This will return a 4-by-3 array: one row for each time, one column for each location. If you want to override this behavior and instead consider a sequence of (psi, t) points, set the `each_t` keyword to False::

    psinorm = e.psinorm2rmid([0.3, 0.35], [1.0, 1.1], each_t=False)

This will return a two-element array with the Rmid values for (psinorm=0.3, t=1.0) and (psinorm=0.35, t=1.1).

For programmatically mapping between coordinates, the :py:meth:`~eqtools.core.Equilibrium.rho2rho` method is quite useful. To map from outboard midplane major radius to normalized flux surface volume, you can simply call::

    e.rho2rho('Rmid', 'volnorm', 0.75, 1.0)

Finally, to get a look at the flux surfaces, simply run::

    e.plotFlux()

Package Reference
-----------------

.. toctree::
   :maxdepth: 4
   
   eqtools


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

