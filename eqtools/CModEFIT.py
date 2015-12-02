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
working with C-Mod EFIT data.
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

class CModEFITTree(EFITTree):
    """Inherits :py:class:`eqtools.EFIT.EFITTree` class. Machine-specific data
    handling class for Alcator C-Mod. Pulls EFIT data from selected MDS tree
    and shot, stores as object attributes. Each EFIT variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for EFIT mapping are pulled on initialization (e.g. psirz grid). Additional
    data are pulled at the first request and stored for subsequent usage.
    
    Intializes C-Mod version of EFITTree object.  Pulls data from MDS tree for 
    storage in instance attributes.  Core attributes are populated from the MDS 
    tree on initialization.  Additional attributes are initialized as None, 
    filled on the first request to the object.

    Args:
        shot (integer): C-Mod shot index.
    
    Keyword Args:
        tree (string): Optional input for EFIT tree, defaults to 'ANALYSIS'
            (i.e., EFIT data are under \\analysis::top.efit.results).
            For any string TREE (such as 'EFIT20') other than 'ANALYSIS',
            data are taken from \\TREE::top.results.
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
            defaults to 'g_eqdsk' (i.e., EFIT data are under
            \\tree::top.results.G_EQDSK)
        afile (string): Optional input for EFIT aeqdsk location name,
            defaults to 'a_eqdsk' (i.e., EFIT data are under 
            \\tree::top.results.A_EQDSK)
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
    def __init__(self, shot, tree='ANALYSIS', length_unit='m', gfile='g_eqdsk', 
                 afile='a_eqdsk', tspline=False, monotonic=True):
        if tree.upper() == 'ANALYSIS':
            root = '\\analysis::top.efit.results.'
        else:
            root = '\\'+tree+'::top.results.'

        super(CModEFITTree, self).__init__(shot, tree, root, 
              length_unit=length_unit, gfile=gfile, afile=afile, 
              tspline=tspline, monotonic=monotonic)
        
        self.getFluxVol() #getFluxVol is called due to wide use on C-Mod

    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface.

        Keyword Args:
            length_unit (String or 3): unit for plasma volume.  Defaults to 3,
                indicating default volumetric unit (typically m^3).

        Returns:
            fluxVol (Array): [nt,npsi] array of volume within flux surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fluxVol is None:
            try:
                fluxVolNode = self._MDSTree.getNode(self._root+'fitout:volp')
                self._fluxVol = fluxVolNode.data().T
                # Units aren't properly stored in the tree for this one!
                if fluxVolNode.units != ' ':
                    self._defaultUnits['_fluxVol'] = str(fluxVolNode.units)
                else:
                    self._defaultUnits['_fluxVol'] = 'm^3'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units are m^3, but aren't stored in the tree!
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_fluxVol'], length_unit)
        return unit_factor * self._fluxVol.copy()

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
                RmidPsiNode = self._MDSTree.getNode(self._root+'fitout:rpres')
                self._RmidPsi = RmidPsiNode.data().T
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = str(RmidPsiNode.units)
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidPsi'], length_unit)
        return unit_factor * self._RmidPsi.copy()

    def getF(self):
        """returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov 
        solutions.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fpol is None:
            try:
                fNode = self._MDSTree.getNode(self._root+self._gfile+':fpol')
                self._fpol = fNode.data().T
                self._defaultUnits['_fpol'] = str(fNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._fpol.copy()
    
    def getFluxPres(self):
        """returns pressure at flux surface.

        Returns:
            p (Array): [nt,npsi] array of pressure on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fluxPres is None:
            try:
                fluxPresNode = self._MDSTree.getNode(self._root+self._gfile+':pres')
                self._fluxPres = fluxPresNode.data().T
                self._defaultUnits['_fluxPres'] = str(fluxPresNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._fluxPres.copy()

    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FFprime (Array): [nt,npsi] array of FF' fromgrad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._ffprim is None:
            try:
                FFPrimeNode = self._MDSTree.getNode(self._root+self._gfile+':ffprim')
                self._ffprim = FFPrimeNode.data().T
                self._defaultUnits['_ffprim'] = str(FFPrimeNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._ffprim.copy()

    def getPPrime(self):
        """returns plasma pressure gradient as a function of psi.

        Returns:
            pprime (Array): [nt,npsi] array of pressure gradient on flux surface 
            psi from grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._pprime is None:
            try:
                pPrimeNode = self._MDSTree.getNode(self._root+self._gfile+':pprime')
                self._pprime = pPrimeNode.data().T
                self._defaultUnits['_pprime'] = str(pPrimeNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._pprime.copy()

    def getQProfile(self):
        """returns profile of safety factor q.

        Returns:
            qpsi (Array): [nt,npsi] array of q on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qpsi is None:
            try:
                qpsiNode = self._MDSTree.getNode(self._root+self._gfile+':qpsi')
                self._qpsi = qpsiNode.data().T
                self._defaultUnits['_qpsi'] = str(qpsiNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    def getRLCFS(self, length_unit=1):
        """returns R-values of LCFS position.

        Returns:
            RLCFS (Array): [nt,n] array of R of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._RLCFS is None:
            try:
                RLCFSNode = self._MDSTree.getNode(self._root+self._gfile+':rbbbs')
                self._RLCFS = RLCFSNode.data().T
                self._defaultUnits['_RLCFS'] = str(RLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RLCFS'], length_unit)
        return unit_factor * self._RLCFS.copy()

    def getZLCFS(self, length_unit=1):
        """returns Z-values of LCFS position.

        Returns:
            ZLCFS (Array): [nt,n] array of Z of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._ZLCFS is None:
            try:
                ZLCFSNode = self._MDSTree.getNode(self._root+self._gfile+':zbbbs')
                self._ZLCFS = ZLCFSNode.data().T
                self._defaultUnits['_ZLCFS'] = str(ZLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_ZLCFS'], length_unit)
        return unit_factor * self._ZLCFS.copy()
        

    def getMachineCrossSectionFull(self):
        """Pulls C-Mod cross-section data from tree, converts to plottable
        vector format for use in other plotting routines

        Returns:
            (`x`, `y`)

            * **x** (`Array`) - [n] array of x-values for machine cross-section.
            * **y** (`Array`) - [n] array of y-values for machine cross-section.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        #pull cross-section from tree
        try:
            ccT = MDSplus.Tree('analysis', self._shot)
            path = '\\analysis::top.limiters.tiles:'
            xvctr = ccT.getNode(path+'XTILE').data()
            yvctr = ccT.getNode(path+'YTILE').data()
            nvctr = ccT.getNode(path+'NSEG').data()
            lvctr = ccT.getNode(path+'PTS_PER_SEG').data()
        except MDSplus._treeshr.TreeException:
            raise ValueError('data load failed.')

        #xvctr, yvctr stored as [nvctr,npts] ndarray.  Each of [nvctr] rows
        #represents the x- or y- coordinate of a line segment; lvctr stores the length
        #of each line segment row.  x/yvctr rows padded out to npts = max(lvctr) with
        #uniform zeros.  To rapidly plot this, we want to flatten xvctr,yvctr down to
        #a single x,y vector set.  We'll use Nones to separate line segments (as these
        #break the continuous plotting joints).
        x = []
        y = []
        for i in range(nvctr):
            length = lvctr[i]
            xseg = xvctr[i,0:length]
            yseg = yvctr[i,0:length]
            x.extend(xseg)
            y.extend(yseg)
            if i != nvctr-1:
                x.append(None)
                y.append(None)

        x = scipy.array(x)
        y = scipy.array(y)
        return (x, y)

    def getRCentr(self, length_unit=1):
        """returns EFIT radius where Bcentr evaluated

        Returns:
            R: Radial position where Bcent calculated [m]

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._RCentr is None:
            try:
                RCentrNode = self._MDSTree.getNode(self._root+self._afile+':RCENCM')
                self._RCentr = RCentrNode.data()
                self._defaultUnits['_RCentr'] = str(RCentrNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')        
    
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RCentr'], length_unit)
        return unit_factor * self._RCentr.copy()

class CModEFITTreeProp(CModEFITTree, PropertyAccessMixin):
    """CModEFITTree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass
