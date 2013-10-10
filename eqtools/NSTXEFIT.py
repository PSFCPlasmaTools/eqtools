# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This file is part of EqTools.
#
# EqTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EqTools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

"""This module provides classes for working with NSTX EFIT data.
"""

import scipy

from .EFIT import EFITTree
from .core import PropertyAccessMixin, ModuleWarning

import warnings

try:
    import MDSplus
    from MDSplus._treeshr import TreeException
    _has_MDS = True
except Exception as _e_MDS:
    if isinstance(_e_MDS, ImportError):
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work.",
                      ModuleWarning)
    else:
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work. Exception raised "
                      "was of type %s, message was '%s'."
                      % (_e_MDS.__class__, _e_MDS.message),
                      ModuleWarning)
    _has_MDS = False

class NSTXEFITTree(EFITTree):
    """Inherits :py:class:`~gptools.EFIT.EFITTree` class. Machine-specific data
    handling class for the National Spherical Torus Experiment (NSTX). Pulls EFIT
    data from selected MDS tree and shot, stores as object attributes. Each EFIT
    variable or set of variables is recovered with a corresponding getter method.
    Essential data for EFIT mapping are pulled on initialization (e.g. psirz grid).
    Additional data are pulled at the first request and stored for subsequent usage.
    
    Intializes NSTX version of EFITTree object.  Pulls data from MDS tree for storage
    in instance attributes.  Core attributes are populated from the MDS tree on initialization.
    Additional attributes are initialized as None, filled on the first request to the object.

    Args:
        shot: (long) int
            NSTX shot index (long)
    
    Kwargs:
        tree: str
            Optional input for EFIT tree, defaults to 'EFIT01'
            (i.e., EFIT data are under \\EFIT01::top.results).
        length_unit: str
            Sets the base unit used for any quantity whose
            dimensions are length to any power. Valid options are:
                
                ===========  ===========================================================================================
                'm'          meters
                'cm'         centimeters
                'mm'         millimeters
                'in'         inches
                'ft'         feet
                'yd'         yards
                'smoot'      smoots
                'cubit'      cubits
                'hand'       hands
                'default'    whatever the default in the tree is (no conversion is performed, units may be inconsistent)
                ===========  ===========================================================================================
                
            Default is 'm' (all units taken and returned in meters).
        gfile: str
            Optional input for EFIT geqdsk location name, defaults to 'g_eqdsk'
            (i.e., EFIT data are under \\tree::top.results.G_EQDSK)
        afile: str
            Optional input for EFIT aeqdsk location name, defaults to 'a_eqdsk'
            (i.e., EFIT data are under \\tree::top.results.A_EQDSK)
        tspline: Boolean
            Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic: Boolean
            Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
    """
    def __init__(self, shot, tree='EFIT01', length_unit='m', gfile='geqdsk', afile='aeqdsk', tspline=False, monotonic=False):

        root = '\\'+tree+'::top.results.'

        if not _has_MDS:
            print("ERROR: MDSplus module did not load properly. Exception is below:")
            raise _e_MDS

        super(EFITTree, self).__init__(length_unit=length_unit, tspline=tspline, monotonic=monotonic)
        
        self._shot = shot
        self._tree = tree
        self._root = root
        self._gfile = gfile
        self._afile = afile

        self._MDSTree = MDSplus.Tree(self._tree, self._shot)
        
        self._defaultUnits = {}
        
        #initialize None for non-essential data

        #flux-surface pressure
        self._fluxPres = None                                                #pressure on flux surface (psi,t)

        #fields
        self._btaxp = None                                                   #Bt on-axis, with plasma (t)
        self._btaxv = None                                                   #Bt on-axis, vacuum (t)
        self._bpolav = None                                                  #avg poloidal field (t)

        #plasma current
        self._IpCalc = None                                                  #EFIT-calculated plasma current (t)
        self._IpMeas = None                                                  #measured plasma current (t)
        self._Jp = None                                                      #grid of current density (r,z,t)
        self._currentSign = None                                             #sign of current for entire shot (calculated in moderately kludgey manner)

        #safety factor parameters
        self._q0 = None                                                      #q on-axis (t)
        self._q95 = None                                                     #q at 95% flux (t)
        self._qLCFS = None                                                   #q at LCFS (t)
        self._rq1 = None                                                     #outboard-midplane minor radius of q=1 surface (t)
        self._rq2 = None                                                     #outboard-midplane minor radius of q=2 surface (t)
        self._rq3 = None                                                     #outboard-midplane minor radius of q=3 surface (t)

        #shaping parameters
        self._kappa = None                                                   #LCFS elongation (t)
        self._dupper = None                                                  #LCFS upper triangularity (t)
        self._dlower = None                                                  #LCFS lower triangularity (t)

        #(dimensional) geometry parameters
        self._rmag = None                                                    #major radius, magnetic axis (t)
        self._zmag = None                                                    #Z magnetic axis (t)
        self._aLCFS = None                                                   #outboard-midplane minor radius (t)
        self._RmidLCFS = None                                                #outboard-midplane major radius (t)
        self._areaLCFS = None                                                #LCFS surface area (t)
        self._RLCFS = None                                                   #R-positions of LCFS (t,n)
        self._ZLCFS = None                                                   #Z-positions of LCFS (t,n)
        
        #machine geometry parameters
        self._Rlimiter = None                                                #R-positions of vacuum-vessel wall (t)
        self._Zlimiter = None                                                #Z-positions of vacuum-vessel wall (t)

        #calc. normalized-pressure values
        self._betat = None                                                   #EFIT-calc toroidal beta (t)
        self._betap = None                                                   #EFIT-calc avg. poloidal beta (t)
        self._Li = None                                                      #EFIT-calc internal inductance (t)

        #diamagnetic measurements
        self._diamag = None                                                  #diamagnetic flux (t)
        self._betatd = None                                                  #diamagnetic toroidal beta (t)
        self._betapd = None                                                  #diamagnetic poloidal beta (t)
        self._WDiamag = None                                                 #diamagnetic stored energy (t)
        self._tauDiamag = None                                               #diamagnetic energy confinement time (t)

        #energy calculations
        self._WMHD = None                                                    #EFIT-calc stored energy (t)
        self._tauMHD = None                                                  #EFIT-calc energy confinement time (t)
        self._Pinj = None                                                    #EFIT-calc injected power (t)
        self._Wbdot = None                                                   #EFIT d/dt magnetic stored energy (t)
        self._Wpdot = None                                                   #EFIT d/dt plasma stored energy (t)

        #load essential mapping data
        # Set the variables to None first so the loading calls will work right:
        self._time = None                                                    #EFIT timebase
        self._psiRZ = None                                                   #EFIT flux grid (r,z,t)
        self._rGrid = None                                                   #EFIT R-axis (t)
        self._zGrid = None                                                   #EFIT Z-axis (t)
        self._psiLCFS = None                                                 #flux at LCFS (t)
        self._psiAxis = None                                                 #flux at magnetic axis (t)
        self._fluxVol = None                                                 #volume within flux surface (psi,t)
        self._volLCFS = None                                                 #volume within LCFS (t)
        self._qpsi = None                                                    #q profile (psi,t)
        self._RmidPsi = None                                                 #max major radius of flux surface (t,psi)
        
        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        self.getTimeBase()
        self.getFluxGrid() # loads _psiRZ, _rGrid and _zGrid at once.
        self.getFluxLCFS()
        self.getFluxAxis()
        self.getVolLCFS()
        self.getQProfile()
        self.getRmidPsi()
        
        
    def getFluxGrid(self):
        """returns EFIT flux grid, [t,z,r]
        """
        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.getNode(self._root+self._gfile+':psirz')
                self._psiRZ = psinode.data()
                self._rGrid = psinode.dim_of(1).data()[0]
                self._zGrid = psinode.dim_of(2).data()[0]
                self._defaultUnits['_psiRZ'] = psinode.units
                self._defaultUnits['_rGrid'] = psinode.dim_of(1).units
                self._defaultUnits['_zGrid'] = psinode.dim_of(2).units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiRZ.copy()
        
        

    def getFluxVol(self): 
        """
        Not implemented in NSTXEFIT tree.
        
        Returns volume within flux surface [psi,t]
        """
        raise NotImplementedError()
        
    def getRmidPsi(self, length_unit=1):
        """ returns maximum major radius of each flux surface [t,psi]
        """
        
        if self._RmidPsi is None:
            try:
                RmidPsiNode = self._MDSTree.getNode(self._root+'derived:psivsrz0')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = RmidPsiNode.units
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        
        if self._defaultUnits['_RmidPsi'] != 'Wb/rad':
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidPsi'], length_unit)
        else:
            unit_factor = scipy.array([1.])
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=RuntimeWarning)
            return unit_factor * self._RmidPsi.copy()
        
        
    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS [t]
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree.getNode(self._root+self._afile+':volume')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = volLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()

    def rz2volnorm(self,*args,**kwargs):
        """ Calculated normalized volume of flux surfaces not stored in NSTX EFIT. All maping with Volnorm
        not implemented"""
        raise NotImplementedError()

    def psinorm2volnorm(self,*args,**kwargs):
        """ Calculated normalized volume of flux surfaces not stored in NSTX EFIT. All maping with Volnorm
        not implemented"""
        raise NotImplementedError()

class NSTXEFITTreeProp(NSTXEFITTree, PropertyAccessMixin):
    """NSTXEFITTree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass
