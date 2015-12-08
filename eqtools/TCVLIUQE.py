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
working with TCV LIUQE Equilibrium.
"""

import scipy
from collections import namedtuple
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

try:
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mplw
    import matplotlib.gridspec as mplgs
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    from matplotlib.ticker import MaxNLocator
    import matplotlib._cntr as cntr

except Exception:
    warnings.warn("matplotlib modules could not be loaded -- plotting and gfile"
                  " writing will not be available.",
                  ModuleWarning)

# we need to define the green function area from the polygon
# see http://stackoverflow.com/questions/22678990/how-can-i-calculate-the-area-within-a-contour-in-python-using-the-matplotlib
# see also http://stackoverflow.com/questions/18304722/python-find-contour-lines-from-matplotlib-pyplot-contour
# for how to compute the contours without calling matplotlib contours

def greenArea(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a


class TCVLIUQETree(EFITTree):
    """Inherits :py:class:`eqtools.EFIT.EFITTree` class. Machine-specific data
    handling class for TCV Machine. Pulls LIUQUE data from selected MDS tree
    and shot, stores as object attributes eventually transforming it in the
    equivalent quantity for EFIT. Each  variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for LIUQUE mapping are pulled on initialization (e.g. psirz grid). Additional
    data are pulled at the first request and stored for subsequent usage.
    
    Intializes TCV version of EFITTree object.  Pulls data from MDS tree for
    storage in instance attributes.  Core attributes are populated from the MDS 
    tree on initialization.  Additional attributes are initialized as None, 
    filled on the first request to the object.

    Args:
        shot (integer): TCV shot index.
    
    Keyword Args:
        tree (string): Optional input for LIUQE tree, defaults to 'RESULTS'
            (i.e., LIUQE data are under \\results::).
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
    def __init__(self, shot, tree='tcv_shot', length_unit='m', gfile='g_eqdsk',
                 afile='a_eqdsk', tspline=False, monotonic=True):
        # this is the root tree where all the LIUQE results are stored
        root = r'\results'

        super(TCVLIUQETree, self).__init__(shot, tree, root, 
                                           length_unit=length_unit, gfile=gfile, afile=afile, 
                                           tspline=tspline, monotonic=monotonic)

    # ---  1
    def getInfo(self):
        """returns namedtuple of shot information
        
        Returns:
            namedtuple containing
                
                =====   ===============================
                shot    TCV shot index (long)
                tree    LIUQE tree (string)
                nr      size of R-axis for spatial grid
                nz      size of Z-axis for spatial grid
                nt      size of timebase for flux grid
                =====   ===============================
        """
        try:
            nt = len(self._time)
            nr = len(self._rGrid)
            nz = len(self._zGrid)
        except TypeError:
            nt, nr, nz = 0, 0, 0
            print 'tree has failed data load.'

        data = namedtuple('Info',['shot','tree','nr','nz','nt'])
        return data(shot=self._shot,tree=self._tree,nr=nr,nz=nz,nt=nt)

    # ---  2
    def getTimeBase(self):
        """returns LIUQE time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._time is None:
            try:
                psinode = self._MDSTree.getNode(self._root+'::psi')
                self._time = psinode.getDimensionAt(2).data()
                self._defaultUnits['_time'] = 's'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._time.copy()

    # ---  3
    def getFluxGrid(self):
        """returns LIUQE flux grid.

        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.
        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
         """

        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.getNode(self._root+'::psi')
                self._psiRZ = psinode.data() / (2.*scipy.pi)
                self._rGrid = psinode.dim_of(0).data()
                self._zGrid = psinode.dim_of(1).data()
                self._defaultUnits['_psiRZ'] = str(psinode.units)
                self._defaultUnits['_rGrid'] = 'm'
                self._defaultUnits['_zGrid'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # the transpose is needed as psi is saved as (R, Z, t) in the pulse file
        return self._psiRZ.copy()

    # ---  4
    def getRGrid(self, length_unit=1):
        """returns LIUQE R-axis.

        Returns:
            rGrid (Array): [nr] array of R-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'],
                                                      length_unit)
        return unit_factor * self._rGrid.copy()

    # ---  5
    def getZGrid(self, length_unit=1):
        """returns LIUQE Z-axis.

        Returns:
            zGrid (Array): [nz] array of Z-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zGrid'],
                                                      length_unit)
        return unit_factor * self._zGrid.copy()
    
    # ---  6
    def getFluxAxis(self):
        """returns psi on magnetic axis.

        Returns:
            psiAxis (Array): [nt] array of psi on magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiAxis is None:
            try:
                psiAxisNode = self._MDSTree.getNode(self._root+'::psi_axis')
                self._psiAxis =  psiAxisNode.data() / (2.*scipy.pi)
                self._defaultUnits['_psiAxis'] = str(psiAxisNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiAxis.copy()
    
    # ---  7
    def getFluxLCFS(self):
        """returns psi at separatrix. 

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiLCFS is None:
            try:
                # psiLCFSNode = self._MDSTree.getNode(self._root+'::surface_flux')
                # self._psiLCFS = psiLCFSNode.data()
                # self._defaultUnits['_psiLCFS'] = str(psiLCFSNode.units)
                self._psiLCFS = scipy.zeros(self.getTimeBase().size)
                self._defaultUnits['_psiLCFS'] = 'T*m^2'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiLCFS.copy()

    
    # ---  8
    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface. This is not implemented in LIUQE
        as default output. So we use contour and GREEN theorem to get the area
        within a default grid of the PSI. Then we compute the volume by multipling
        for 2pi * VolLCFS / AreaLCFS.  

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

                # first determine npsi
                nPsi = self.getRmidPsi().shape[1]
                # then the psi from psiGrid
                psiRZ = self.getFluxGrid()
                # the rGrid, zGrid in an appropriate mesh
                R, Z = scipy.meshgrid(self.getRGrid(), self.getZGrid())
                # read the LCFS Volume and Area and compute the appropriate twopi R
                rUsed = self.getVolLCFS() / self.getAreaLCFS()
                # define the output
                volumes = scipy.zeros((psiRZ.shape[0], nPsi))
                outArea = scipy.zeros(nPsi)
                # now we start to iterate over the times
                for i in range(psiRZ.shape[0]):
                    psi = psiRZ[i]
                    # define the levels
                    levels = scipy.linspace(psi.max(), 0, nPsi)
                    c = cntr.Cntr(R, Z, psi)
                    for j in range(nPsi - 1):
                        nlist = c.trace(levels[j + 1])
                        segs = nlist[: len(nlist) // 2]
                        outArea[j + 1] = abs(greenArea(segs[0]))
                    volumes[i,: ] = outArea *  rUsed[i]
                # then the levels for the contours
                self._fluxVol = volumes
                # Units aren't properly stored in the tree for this one!
                self._defaultUnits['_fluxVol'] = 'm^3'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units are m^3, but aren't stored in the tree!
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_fluxVol'], length_unit)
        return unit_factor * self._fluxVol.copy()

    # ---  9
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
                volLCFSNode = self._MDSTree.getNode(self._root+'::volume')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = str(volLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()

    # ---  10
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
                RmidPsiNode = self._MDSTree.getNode(self._root+'::r_max_psi')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = str(RmidPsiNode.units)
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidPsi'], length_unit)
        return unit_factor * self._RmidPsi.copy()

    # ---  11
    def getRLCFS(self, length_unit=1):
        """returns R-values of LCFS position.

        Returns:
            RLCFS (Array): [nt,n] array of R of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._RLCFS is None:
            try:
                RLCFSNode = self._MDSTree.getNode(self._root+'::r_contour')
                self._RLCFS = RLCFSNode.data()
                self._defaultUnits['_RLCFS'] = str(RLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RLCFS'], length_unit)
        return unit_factor * self._RLCFS.copy()

    # ---  12
    def getZLCFS(self, length_unit=1):
        """returns Z-values of LCFS position.

        Returns:
            ZLCFS (Array): [nt,n] array of Z of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._ZLCFS is None:
            try:
                ZLCFSNode = self._MDSTree.getNode(self._root+'::z_contour')
                self._ZLCFS = ZLCFSNode.data()
                self._defaultUnits['_ZLCFS'] = str(ZLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_ZLCFS'], length_unit)
        return unit_factor * self._ZLCFS.copy()

    # ---  13
    def getF(self):
        """returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov 
        solutions. Not implemented on LIUQE

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)
            Not stored on LIUQE nodes
        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
    
    # ---  14
    def getFluxPres(self):
        """returns pressure at flux surface. Not implemented. We have pressure
           saved on the same grid of psi

        Returns:
            p (Array): [nt,npsi] array of pressure on flux surface psi.
            Not implemented on LIUQE nodes. We have pressure on the grid use for psi
        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fluxPres is None:
            try:
                fluxPPresNode = self._MDSTree.getNode(self._root+'::ppr_coeffs')
                duData = fluxPPresNode.data()
                # then we build an appropriate grid 
                nPsi = self.getRmidPsi().shape[1]
                psiV = scipy.linspace(1,0,nPsi)

                rad = [psiV]
                for i in range(duData.shape[1]-1):
                    rad += [rad[-1]*psiV*(i+1)/(i+2)]
                rad = scipy.vstack(rad)
                self._fluxPres = scipy.reshape(self.getFluxAxis(),(self.getFluxAxis().size,1))*scipy.dot(duData,rad)/(2*scipy.pi)

                self._defaultUnits['_fluxPres'] = 'Pa'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._fluxPres.copy()

    # ---  15
    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FFprime (Array): [nt,npsi] array of FF' fromgrad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # if self._ffprim is None:
        #     try:
        #         FFPrimeNode = self._MDSTree.getNode(self._root+self._gfile+':ffprim')
        #         self._ffprim = FFPrimeNode.data()
        #         self._defaultUnits['_ffprim'] = str(FFPrimeNode.units)
        #     except TreeException:
        #         raise ValueError('data retrieval failed.')
        # return self._ffprim.copy()

    # ---  16
    def getPPrime(self):
        """returns plasma pressure gradient as a function of psi.

        Returns:
            pprime (Array): [nt,npsi] array of pressure gradient on flux surface 
            psi from grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # raise NotImplementedError()
        # in Liuqe pprime is not given in the appropriate flux surface but it is saved as coefficients
        # ppr_coeffs. So we need to build the derivative as
        # p' = p0 + p1 * phi + p2 * phi^2 + p3 * phi^3 with phi = (psi - psi_edge) / (psi_axis - psi_edge)
        # But conventionally psi_edge on TCV = 0 -->  phi = psi / psi_axis
        if self._pprime is None:
            try:
                fluxPPresNode = self._MDSTree.getNode(self._root+'::ppr_coeffs')
                duData = fluxPPresNode.data()

                # then we build an appropriate grid 
                nPsi = self.getRmidPsi().shape[1]
                psiV = scipy.linspace(1, 0, nPsi)

                # This should be faster through some vectorization /
                # slowing down to fortran matrix multiplication subroutines
                rad = [scipy.ones(psiV.size)]
                for i in range(duData.shape[1]-1):
                    rad += [rad[-1]*psiV]
                rad = scipy.vstack(rad)

                self._pprime = scipy.dot(duData, rad)
                self._defaultUnits['_fluxPres'] = 'A/m^3'
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._pprime.copy()

    # ---  17
    def getElongation(self):
        """returns LCFS elongation.

        Returns:
            kappa (Array): [nt] array of LCFS elongation.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._kappa is None:
            try:
                kappaNode = self._MDSTree.getNode(self._root+'::kappa_edge')
                self._kappa = kappaNode.data()
                self._defaultUnits['_kappa'] = ' '
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._kappa.copy()

    # ---  18
    def getUpperTriangularity(self):
        """returns LCFS upper triangularity.

        Returns:
            deltau (Array): [nt] array of LCFS upper triangularity.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._dupper is None:
            try:
                dupperNode = self._MDSTree.getNode(self._root+'::delta_ed_top')
                self._dupper = dupperNode.data()
                self._defaultUnits['_dupper'] = str(dupperNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dupper.copy()
        
    # ---  19
    def getLowerTriangularity(self):
        """returns LCFS lower triangularity.

        Returns:
            deltal (Array): [nt] array of LCFS lower triangularity.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._dlower is None:
            try:
                dlowerNode = self._MDSTree.getNode(self._root+'::delta_ed_bot')
                self._dlower = dlowerNode.data()
                self._defaultUnits['_dlower'] = str(dlowerNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dlower.copy()

    # ---  21
    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius.

        Returns:
            magR (Array): [nt] array of major radius of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rmag is None:
            try:
                rmagNode = self._MDSTree.getNode(self._root+'::r_axis')
                self._rmag = rmagNode.data()
                self._defaultUnits['_rmag'] = str(rmagNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()

    # ---  22
    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zmag is None:
            try:
                zmagNode = self._MDSTree.getNode(self._root+'::z_axis')
                self._zmag = zmagNode.data()
                self._defaultUnits['_zmag'] = str(zmagNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zmag'], length_unit)
        return unit_factor * self._zmag.copy()
    
    # ---  23
    def getAreaLCFS(self, length_unit=2):
        """returns LCFS cross-sectional area.

        Keyword Args:
            length_unit (String or 2): unit for LCFS area.  Defaults to 2, 
                denoting default areal unit (typically m^2).

        Returns:
            areaLCFS (Array): [nt] array of LCFS area.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._areaLCFS is None:
            try:
                areaLCFSNode = self._MDSTree.getNode(self._root+'::area')
                self._areaLCFS = areaLCFSNode.data()
                self._defaultUnits['_areaLCFS'] = str(areaLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Units should be cm^2:
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_areaLCFS'], length_unit)
        return unit_factor * self._areaLCFS.copy()

    # ---  24
    def getAOut(self, length_unit=1):
        """returns outboard-midplane minor radius at LCFS. In LIUQE it is the last value
        of \results::r_max_psi

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            aOut (Array): [nt] array of LCFS outboard-midplane minor radius.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # defin a simple way to fin the closest index to zero
        if self._aLCFS is None:
            try:
                _dummy = self.getRmidPsi()
                self._aLCFS = _dummy[:, _dummy.shape[1] - 1]
                self._defaultUnits['_aLCFS']='m'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_aLCFS'], length_unit)
        return unit_factor * self._aLCFS.copy()

    # ---  25
    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius. It uses getA

        Keyword Args:
            length_unit (String or 1): unit for major radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            RmidOut (Array): [nt] array of major radius of LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._RmidLCFS is None:
            try:
                # this variable is not saved in the pulse file.
                # we compute this by adding the Major radius of the machine to the computed AOut()
                RMaj = 0.88/0.996 # almost 0.88
                self._RmidLCFS = self.getAOut()+RMaj
                # The units aren't properly stored in the tree for this one!
                # Should be meters.
                self._defaultUnits['_RmidLCFS'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidLCFS'], length_unit)
        return unit_factor * self._RmidLCFS.copy()

    # ---  27
    def getQProfile(self):
        """returns profile of safety factor q.

        Returns:
            qpsi (Array): [nt,npsi] array of q on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qpsi is None:
            try:
                qpsiNode = self._MDSTree.getNode(self._root+'::q_psi')
                self._qpsi = qpsiNode.data()
                self._defaultUnits['_qpsi'] = str(qpsiNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    # ---  28
    def getQ0(self):
        """returns q on magnetic axis,q0.

        Returns:
            q0 (Array): [nt] array of q(psi=0).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._q0 is None:
            try:
                q0Node = self._MDSTree.getNode(self._root+'::q_zero')
                self._q0 = q0Node.data()
                self._defaultUnits['_q0'] = str(q0Node.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q0.copy()

    # ---  29
    def getQ95(self):
        """returns q at 95% flux surface.

        Returns:
            q95 (Array): [nt] array of q(psi=0.95).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._q95 is None:
            try:
                q95Node = self._MDSTree.getNode(self._root+'::q_95')
                self._q95 = q95Node.data()
                self._defaultUnits['_q95'] = str(q95Node.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q95.copy()

    # ---  30
    def getQLCFS(self):
        """returns q on LCFS (interpolated).

        Returns:
            qLCFS (Array): [nt] array of q* (interpolated).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qLCFS is None:
            try:
                qLCFSNode = self._MDSTree.getNode(self._root+'::q_edge')
                self._qLCFS = qLCFSNode.data()
                self._defaultUnits['_qLCFS'] = str(qLCFSNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._qLCFS.copy()
    
    # ---  35
    def getBtVac(self):
        """Returns vacuum toroidal field on-axis. We use MDSplus.Connection
        for a proper use of the TDI function tcv_eq()

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._btaxv is None:
            try:
                # constant is due to a detailed measurements on the vacuum vessel major radius
                # introduce to be consistent with TDI function tcv_eq.fun
                RMaj = 0.88/0.996 # almost 0.88 m
                # open a connection
                conn = MDSplus.Connection('tcvdata.epfl.ch')
                conn.openTree('tcv_shot', self._shot)
                bt = conn.get('tcv_eq("BZERO")').data()[0]/RMaj
                btTime = conn.get('dim_of(tcv_eq("BZERO"))').data()
                conn.closeTree(self._tree, self._shot)
                # we need to interpolate on the time basis of LIUQE
                self._btaxv = scipy.interp(self.getTimeBase(), btTime, bt)
                self._defaultUnits['_btaxv'] = 'T'
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxv.copy()

    # ---  36
    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Returns:
            BtPla (Array): [nt] array of toroidal field including plasma effects.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        raise NotImplementedError()
        # if self._btaxp is None:
        #     try:
        #         btaxpNode = self._MDSTree.getNode(self._root+self._afile+':btaxp')
        #         self._btaxp = btaxpNode.data()
        #         self._defaultUnits['_btaxp'] = str(btaxpNode.units)
        #     except (TreeException,AttributeError):
        #         raise ValueError('data retrieval failed.')
        # return self._btaxp.copy()
        
    # ---  39
    def getIpCalc(self):
        """returns EFIT-calculated plasma current.

        Returns:
            IpCalc (Array): [nt] array of EFIT-reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpCalc is None:
            try:
                IpCalcNode = self._MDSTree.getNode(self._root + '::i_p')
                self._IpCalc = IpCalcNode.data()
                self._defaultUnits['_IpCalc'] = str(IpCalcNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpCalc.copy()
        
    # ---  40
    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpMeas is None:
            try:
                conn = MDSplus.Connection('tcvdata.epfl.ch')
                conn.openTree('tcv_shot', self._shot)
                ip = conn.get('tcv_ip()').data()
                ipTime = conn.get('dim_of(tcv_ip())').data()
                conn.closeTree(self._tree, self._shot)
                self._IpMeas = scipy.interp(self.getTimeBase(), ipTime, ip)
                self._defaultUnits['_IpMeas'] = 'A'
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpMeas.copy()

    # ---  42
    def getBetaT(self):
        """returns LIUQE-calculated toroidal beta.

        Returns:
            BetaT (Array): [nt] array of LIUQE-calculated average toroidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betat is None:
            try:
                betatNode = self._MDSTree.getNode(self._root+'::beta_tor')
                self._betat = betatNode.data()
                self._defaultUnits['_betat'] = str(betatNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betat.copy()

    # ---  43
    def getBetaP(self):
        """returns LIUQE-calculated poloidal beta.

        Returns:
            BetaP (Array): [nt] array of LIUQE-calculated average poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betap is None:
            try:
                betapNode = self._MDSTree.getNode(self._root+'::beta_pol')
                self._betap = betapNode.data()
                self._defaultUnits['_betap'] = str(betapNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betap.copy()
    
        
    # ---  44
    def getLi(self):
        """returns LIUQE-calculated internal inductance.

        Returns:
            Li (Array): [nt] array of LIUQE-calculated internal inductance.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Li is None:
            try:
                LiNode = self._MDSTree.getNode(self._root+'::l_i')
                self._Li = LiNode.data()
                self._defaultUnits['_Li'] = str(LiNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Li.copy()
    
    # ---  50
    def getDiamagWp(self):
        """returns diamagnetic-loop plasma stored energy.

        Returns:
            Wp (Array): [nt] array of measured plasma stored energy.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._WDiamag is None:
            try:
                WDiamagNode = self._MDSTree.getNode(self._root+'::total_energy')
                self._WDiamag = WDiamagNode.data()
                self._defaultUnits['_WDiamag'] = str(WDiamagNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WDiamag.copy()

    # ---  53
    def getTauMHD(self):
        """returns LIUQE-calculated MHD energy confinement time.

        Returns:
            tauMHD (Array): [nt] array of LIUQE-calculated energy confinement time.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._tauMHD is None:
            try:
                tauMHDNode = self._MDSTree.getNode(self._root+'::tau_e')
                self._tauMHD = tauMHDNode.data()
                self._defaultUnits['_tauMHD'] = str(tauMHDNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._tauMHD.copy()

    # ---  59
    def getMachineCrossSection(self):
        """Pulls TCV cross-section data from tree, converts to plottable
        vector format for use in other plotting routines

        Returns:
            (`x`, `y`)

            * **x** (`Array`) - [n] array of x-values for machine cross-section.
            * **y** (`Array`) - [n] array of y-values for machine cross-section.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # pull cross-section from tree
        try:
            self._Rlimiter = MDSplus.Data.execute('static("r_t")').getValue().data()
            self._Zlimiter = MDSplus.Data.execute('static("z_t")').getValue().data()
        except MDSplus._treeshr.TreeException:
            raise ValueError('data load failed.')

        return (self._Rlimiter,self._Zlimiter)

    # ---  60
    def getMachineCrossSectionPatch(self):
        """Pulls TCV cross-section data from tree, converts it directly to
        a matplotlib patch which can be simply added to the approriate axes
        call in plotFlux()

        Returns:
            tiles matplotlib Patch, vessel matplotlib Patch

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        # pull cross-section from tree
        try:
            Rv_in = MDSplus.Data.execute('static("r_v:in")').getValue().data()
            Rv_out = MDSplus.Data.execute('static("r_v:out")').getValue().data()
            Zv_in = MDSplus.Data.execute('static("z_v:in")').getValue().data()
            Zv_out = MDSplus.Data.execute('static("z_v:out")').getValue().data()
        except MDSplus._treeshr.TreeException:
            raise ValueError('data load failed.')

        # this is for the vessel
        verticesIn = [r for r in zip(Rv_in, Zv_in)]
        verticesIn.append(verticesIn[0])
        codesIn = [Path.MOVETO] + (len(verticesIn) - 1) * [Path.LINETO]
        verticesOut = [r for r in zip(Rv_out, Zv_out)][::-1]
        verticesOut.append(verticesOut[0])
        codesOut = [Path.MOVETO] + (len(verticesOut) - 1) * [Path.LINETO]
        vessel_path = Path(verticesIn + verticesOut, codesIn + codesOut)
        vessel_patch = PathPatch(vessel_path, facecolor=(0.6, 0.6, 0.6),
                                 edgecolor='black')
        # this is for the tiles
        x, y = self.getMachineCrossSection()
        verticesIn = [r for r in zip(x, y)][::- 1]
        verticesIn.append(verticesIn[0])
        codesIn = [Path.MOVETO] + (len(verticesIn)-1) * [Path.LINETO]
        verticesOut = [r for r in zip(Rv_in, Zv_in)]
        verticesOut.append(verticesOut[0])
        codesOut = [Path.MOVETO] + (len(verticesOut) - 1) * [Path.LINETO]
        tiles_path = Path(verticesIn + verticesOut, codesIn + codesOut)
        tiles_patch = PathPatch(tiles_path, facecolor=(0.75, 0.75, 0.75),
                                edgecolor='black')

        return (tiles_patch , vessel_patch)

    # ---  61
    def plotFlux(self, fill=True, mask=False):
        """Plots LIQUE TCV flux contours directly from psi grid.
        
        Returns the Figure instance created and the time slider widget (in case
        you need to modify the callback). `f.axes` contains the contour plot as
        the first element and the time slice slider as the second element.
        
        Keyword Args:
            fill (Boolean):
                Set True to plot filled contours.  Set False (default) to plot white-background
                color contours.
        """
        
        try:
            psiRZ = self.getFluxGrid()
            rGrid = self.getRGrid(length_unit = 'm')
            zGrid = self.getZGrid(length_unit='m')
            t = self.getTimeBase()

            RLCFS = self.getRLCFS(length_unit='m')
            ZLCFS = self.getZLCFS(length_unit='m')
        except ValueError:
            raise AttributeError('cannot plot EFIT flux map.')
        try:
            limx, limy = self.getMachineCrossSection()
        except NotImplementedError:
            if self._verbose:
                print('No machine cross-section implemented!')
            limx = None
            limy = None
        try:
            macx, macy = self.getMachineCrossSectionFull()
        except:
            macx = None
            macy = None

        # event handler for arrow key events in plot windows.  Pass slider object
        # to update as masked argument using lambda function
        # lambda evt: arrow_respond(my_slider,evt)
        def arrowRespond(slider, event):
            if event.key == 'right':
                slider.set_val(min(slider.val+1, slider.valmax))
            if event.key == 'left':
                slider.set_val(max(slider.val-1, slider.valmin))

        # make time-slice window
        fluxPlot = plt.figure(figsize=(6, 11))
        gs = mplgs.GridSpec(2, 1, height_ratios=[30, 1])
        psi = fluxPlot.add_subplot(gs[0,0])
        psi.set_aspect('equal')
        try:
            tilesP, vesselP = self.getMachineCrossSectionPatch()
            psi.add_patch(tilesP)
            psi.add_patch(vesselP)
        except NotImplementedError:
            if self._verbose:
                print('No machine cross-section implemented!')
        psi.set_xlim([0.6, 1.2])
        psi.set_ylim([-0.8, 0.8])

        timeSliderSub = fluxPlot.add_subplot(gs[1,0])
        title = fluxPlot.suptitle('')

        # dummy plot to get x,ylims
        psi.contour(rGrid,zGrid,psiRZ[0],10, colors='k')

        # generate graphical mask for limiter wall
        if mask:
            xlim = psi.get_xlim()
            ylim = psi.get_ylim()
            bound_verts = [(xlim[0],ylim[0]),(xlim[0],ylim[1]),(xlim[1],ylim[1]),(xlim[1],ylim[0]),(xlim[0],ylim[0])]
            poly_verts = [(limx[i],limy[i]) for i in range(len(limx) - 1, -1, -1)]

            bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
            poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

            path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
            patch = mpatches.PathPatch(path,facecolor='white',edgecolor='none')

        def updateTime(val):
            psi.clear()
            t_idx = int(timeSlider.val)

            psi.set_xlim([0.5, 1.2])
            psi.set_ylim([-0.8, 0.8])

            title.set_text('LIUQE Reconstruction, $t = %(t).2f$ s' % {'t':t[t_idx]})
            psi.set_xlabel('$R$ [m]')
            psi.set_ylabel('$Z$ [m]')
            if macx is not None:
                psi.plot(macx, macy, 'k', linewidth=3, zorder=5)
            elif limx is not None:
                psi.plot(limx,limy,'k',linewidth=3,zorder=5)
            # catch NaNs separating disjoint sections of R,ZLCFS in mask
            maskarr = scipy.where(scipy.logical_or(RLCFS[t_idx] > 0.0,scipy.isnan(RLCFS[t_idx])))
            RLCFSframe = RLCFS[t_idx,maskarr[0]]
            ZLCFSframe = ZLCFS[t_idx,maskarr[0]]
            psi.plot(RLCFSframe,ZLCFSframe,'r',linewidth=3,zorder=3)
            if fill:
                psi.contourf(rGrid,zGrid,psiRZ[t_idx],50,zorder=2)
                psi.contour(rGrid,zGrid,psiRZ[t_idx],50,colors='k',linestyles='solid',zorder=3)
            else:
                psi.contour(rGrid,zGrid,psiRZ[t_idx],50,colors='k')
            if mask:
                patchdraw = psi.add_patch(patch)
                patchdraw.set_zorder(4)

            psi.add_patch(tilesP)
            psi.add_patch(vesselP)
            psi.set_xlim([0.5, 1.2])
            psi.set_ylim([-0.8, 0.8])

            fluxPlot.canvas.draw()

        timeSlider = mplw.Slider(timeSliderSub,'t index',0,len(t)-1,valinit=0,valfmt="%d")
        timeSlider.on_changed(updateTime)
        updateTime(0)

        plt.ion()
        fluxPlot.show()


class TCVLIUQETreeProp(TCVLIUQETree, PropertyAccessMixin):
    """TCVLIUQETree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass
