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

"""This module provides classes inheriting :py:class:`eqtools.EFIT.EFITTree` for 
working with NSTX EFIT data.
"""

import scipy

from .EFIT import EFITTree
from .core import PropertyAccessMixin, ModuleWarning

import warnings

try:
    import MDSplus
    try: 
        from MDSplus._treeshr import TreeException
    except: 
        from MDSplus.mdsExceptions.treeshrExceptions import TreeException

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
        shot (integer): NSTX shot index (long)
    
    Keyword Args:
        tree (string): Optional input for EFIT tree, defaults to 'EFIT01'
            (i.e., EFIT data are under \\EFIT01::top.results).
        length_unit (string): Sets the base unit used for any quantity whose
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
        gfile (string): Optional input for EFIT geqdsk location name,
            defaults to 'geqdsk' (i.e., EFIT data are under 
            \\tree::top.results.GEQDSK)
        afile (string): Optional input for EFIT aeqdsk location name,
            defaults to 'aeqdsk' (i.e., EFIT data are under 
            \\tree::top.results.AEQDSK)
        tspline (Boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic (Boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
    """
    def __init__(self, shot, tree='EFIT01', length_unit='m', gfile='geqdsk', afile='aeqdsk', tspline=False, monotonic=True):

        root = '\\'+tree+'::top.results.'

        if not _has_MDS:
            print("MDSplus module did not load properly. Exception is below:")
            print(_e_MDS.__class__)
            print(_e_MDS.message)
            print(
                "Most functionality will not be available! (But pickled data "
                "will still be accessible.)"
            )

        super(NSTXEFITTree, self).__init__(shot,
                                           tree,
                                           root,
                                           length_unit=length_unit,
                                           gfile=gfile,
                                           afile=afile,
                                           tspline=tspline,
                                           monotonic=monotonic)
        
    def getFluxGrid(self):
        """returns EFIT flux grid.

        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
         """        

        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.getNode(self._root+self._gfile+':psirz')
                self._psiRZ = psinode.data()
                self._rGrid = psinode.dim_of(1).data()[0]
                self._zGrid = psinode.dim_of(2).data()[0]
                self._defaultUnits['_psiRZ'] = str(psinode.units)
                self._defaultUnits['_rGrid'] = str(psinode.dim_of(1).units)
                self._defaultUnits['_zGrid'] = str(psinode.dim_of(2).units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiRZ.copy()
        
    def getMachineCrossSection(self):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting routines.

        Returns:    
            The requested data.
        """
        if self._Rlimiter is None or self._Zlimiter is None:
            try:
                limitr = self._MDSTree.getNode(self._root+self._gfile+':limitr').data()[0]
                xlim = self._MDSTree.getNode(self._root+self._gfile+':rlim').data()[0]
                ylim = self._MDSTree.getNode(self._root+self._gfile+':zlim').data()[0]
                npts = len(xlim)
                
                if npts < limitr:
                    raise ValueError("Dimensions inconsistent in limiter array lengths.")
                    
                self._Rlimiter = xlim[0:limitr]
                self._Zlimiter = ylim[0:limitr]
            except (TreeException, AttributeError):
                raise ValueError("data retrieval failed.")
        return (self._Rlimiter,self._Zlimiter)        

    def getFluxVol(self): 
        """Not implemented in NSTXEFIT tree.
        
        Returns:
            volume within flux surface [psi,t]
        """
        super(EFITTree,self).getFluxVol()
        
        
    def getRmidPsi(self, length_unit=1):
        """returns maximum major radius of each flux surface.

        Keyword Args:
            length_unit (String or 1): unit of Rmid.  Defaults to 1, indicating 
                the default parameter unit (typically m).

        Returns:
            Rmid (Array): [nt,npsi] array of maximum (outboard) major radius of 
            flux surface psi.

        Raises:
            Value Error: if module cannot retrieve data from MDS tree.
        """
        
        if self._RmidPsi is None:
            try:
                RmidPsiNode = self._MDSTree.getNode(self._root+'derived:psivsrz0')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = str(RmidPsiNode.units)
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
        
    def getIpCalc(self):
        """returns EFIT-calculated plasma current.

        Returns:
            IpCalc (Array): [nt] array of EFIT-reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """

        if self._IpCalc is None:
            try:
                IpCalcNode = self._MDSTree.getNode(self._root+self._gfile+':cpasma')
                self._IpCalc = IpCalcNode.data()
                self._defaultUnits['_IpCalc'] = str(IpCalcNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpCalc.copy()

        
    def getVolLCFS(self, length_unit=3):
        """returns volume within LCFS.

        Keyword Args:
            length_unit (String or 3): unit for LCFS volume.  Defaults to 3, 
                denoting default volumetric unit (typically m^3).

        Returns:
            volLCFS (Array): [nt] array of volume within LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """

        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree.getNode(self._root+self._afile+':volume')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = str(volLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()
 
    def getJp(self):
        """Not implemented in NSTXEFIT tree.

        Returns:
            EFIT-calculated plasma current density Jp on flux grid [t,r,z]
        """
        super(EFITTree,self).getJp()

    def rz2volnorm(self,*args,**kwargs):
        """ Calculated normalized volume of flux surfaces not stored in NSTX EFIT.

        Returns:
            All mapping with Volnorm not implemented
        """
        raise NotImplementedError()


    def psinorm2volnorm(self,*args,**kwargs):
        """ Calculated normalized volume of flux surfaces not stored in NSTX EFIT. 

        Returns:
            All maping with Volnorm not implemented
        """
        raise NotImplementedError()


class NSTXEFITTreeProp(NSTXEFITTree, PropertyAccessMixin):
    """NSTXEFITTree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass
