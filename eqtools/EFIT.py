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

"""Provides class inheriting :py:class:`eqtools.core.Equilibrium` for working 
with EFIT data.
"""

import scipy
from collections import namedtuple

from .core import Equilibrium, ModuleWarning, inPolygon

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
    _has_plt = True
except:
    warnings.warn("Matplotlib.pyplot module could not be loaded -- classes that "
                  "use pyplot will not work.",ModuleWarning)
    _has_plt = False

class EFITTree(Equilibrium):
    """Inherits :py:class:`Equilibrium <eqtools.core.Equilibrium>` class. 
    EFIT-specific data handling class for machines using standard EFIT tag 
    names/tree structure with MDSplus.  Constructor and/or data loading may 
    need overriding in a machine-specific implementation.  Pulls EFIT data 
    from selected MDS tree and shot, stores as object attributes.  Each EFIT 
    variable or set of variables is recovered with a corresponding getter 
    method.  Essential data for EFIT mapping are pulled on initialization 
    (e.g. psirz grid).  Additional data are pulled at the first request and 
    stored for subsequent usage.
    
    Intializes :py:class:`EFITTree` object. Pulls data from MDS tree for 
    storage in instance attributes. Core attributes are populated from the MDS 
    tree on initialization. Additional attributes are initialized as None,
    filled on the first request to the object.

    Args:
        shot (integer): Shot number
        tree (string): MDSplus tree to open to fetch EFIT data.
        root (string): Root path for EFIT data in MDSplus tree.
    
    Keyword Args:
        length_unit (string): Sets the base unit used for any
            quantity whose dimensions are length to any power.
            Valid options are:

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
        tspline (boolean): Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic (boolean): Sets whether or not the "monotonic" form of time
            window finding is used. If True, the timebase must be 
            monotonically increasing. Default is False (use slower,
            safer method).
    """
    def __init__(self, shot, tree, root, length_unit='m', gfile = 'g_eqdsk', 
                 afile='a_eqdsk', tspline=False, monotonic=True):
        if not _has_MDS:
            print("MDSplus module did not load properly. Exception is below:")
            print(_e_MDS.__class__)
            print(_e_MDS.message)
            print(
                "Most functionality will not be available! (But pickled data "
                "will still be accessible.)"
            )

        super(EFITTree, self).__init__(length_unit=length_unit, tspline=tspline, 
                                       monotonic=monotonic)
        
        self._shot = shot
        self._tree = tree
        self._root = root
        self._gfile = gfile
        self._afile = afile

        self._MDSTree = MDSplus.Tree(self._tree, self._shot)
        
        self._defaultUnits = {}
        
        #initialize None for non-essential data

        #grad-shafranov related parameters
        self._fpol = None
        self._fluxPres = None                                                #pressure on flux surface (psi,t)
        self._ffprim = None
        self._pprime = None                                                  #pressure derivative on flux surface (t,psi)

        #fields
        self._btaxp = None                                                   #Bt on-axis, with plasma (t)
        self._btaxv = None                                                   #Bt on-axis, vacuum (t)
        self._bpolav = None                                                  #avg poloidal field (t)
        self._BCentr = None                                                  #Bt at RCentr, vacuum (for gfiles) (t)

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
        self._RCentr = None                                                  #Radius for BCentr calculation (for gfiles) (t)
        
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
        self._fluxVol = None                                                 #volume within flux surface (t,psi)
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
        
    def __str__(self):
        """string formatting for EFITTree class.
        """
        try:
            nt = len(self._time)
            nr=  len(self._rGrid)
            nz = len(self._zGrid)

            mes = 'EFIT data for shot '+str(self._shot)+' from tree '+str(self._tree.upper())+'\n'+\
                  'timebase '+str(self._time[0])+'-'+str(self._time[-1])+'s in '+str(nt)+' points\n'+\
                  str(nr)+'x'+str(nz)+' spatial grid'
            return mes
        except TypeError:
            return 'tree has failed data load.'
    
    def __getstate__(self):
        """Used to close out the MDSplus.Tree instance to make this class pickleable.
        """
        self._MDSTree_internal = None
        return super(EFITTree, self).__getstate__()
    
    @property
    def _MDSTree(self):
        """Use a property to mask the MDSplus.Tree.
        
        This is needed since it isn't pickleable, so we might need to trash it
        and restore it automatically.
        
        You should ALWAYS access _MDSTree since _MDSTree_internal is guaranteed
        to be None after pickling/unpickling.
        """
        if self._MDSTree_internal is None:
            self._MDSTree_internal = MDSplus.Tree(self._tree, self._shot)
        return self._MDSTree_internal
    
    @_MDSTree.setter
    def _MDSTree(self, v):
        self._MDSTree_internal = v
    
    @_MDSTree.deleter
    def _MDSTree(self):
        del self._MDSTree_internal
    
    def getInfo(self):
        """returns namedtuple of shot information
        
        Returns:
            namedtuple containing
                
                =====   ===============================
                shot    C-Mod shot index (long)
                tree    EFIT tree (string)
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

    def getTimeBase(self):
        """returns EFIT time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._time is None:
            try:
                timeNode = self._MDSTree.getNode(self._root+self._afile+':time')
                self._time = timeNode.data()
                self._defaultUnits['_time'] = str(timeNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._time.copy()

    def getFluxGrid(self):
        """returns EFIT flux grid.
        
        Note that this method preserves whatever sign convention is used in the
        tree. For C-Mod, this means that the result should be multiplied by
        -1 * :py:meth:`getCurrentSign()` in most cases.
        
        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.getNode(self._root+self._gfile+':psirz')
                self._psiRZ = psinode.data()
                self._rGrid = psinode.dim_of(0).data()
                self._zGrid = psinode.dim_of(1).data()
                self._defaultUnits['_psiRZ'] = str(psinode.units)
                self._defaultUnits['_rGrid'] = str(psinode.dim_of(0).units)
                self._defaultUnits['_zGrid'] = str(psinode.dim_of(1).units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiRZ.copy()

    def getRGrid(self, length_unit=1):
        """returns EFIT R-axis.

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

    def getZGrid(self, length_unit=1):
        """returns EFIT Z-axis.

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

    def getFluxAxis(self):
        """returns psi on magnetic axis.

        Returns:
            psiAxis (Array): [nt] array of psi on magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiAxis is None:
            try:
                psiAxisNode = self._MDSTree.getNode(self._root+self._afile+':simagx')
                self._psiAxis = psiAxisNode.data()
                self._defaultUnits['_psiAxis'] = str(psiAxisNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiAxis.copy()

    def getFluxLCFS(self):
        """returns psi at separatrix.

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._psiLCFS is None:
            try:
                psiLCFSNode = self._MDSTree.getNode(self._root+self._afile+':sibdry')
                self._psiLCFS = psiLCFSNode.data()
                self._defaultUnits['_psiLCFS'] = str(psiLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiLCFS.copy()

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
                self._fluxVol = fluxVolNode.data()
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
                volLCFSNode = self._MDSTree.getNode(self._root+self._afile+':vout')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = str(volLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()

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
                self._RLCFS = RLCFSNode.data()
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
                self._ZLCFS = ZLCFSNode.data()
                self._defaultUnits['_ZLCFS'] = str(ZLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_ZLCFS'], length_unit)
        return unit_factor * self._ZLCFS.copy()
        
    def remapLCFS(self,mask=False):
        """Overwrites RLCFS, ZLCFS values pulled from EFIT with 
        explicitly-calculated contour of psinorm=1 surface.  This is then masked 
        down by the limiter array using core.inPolygon, restricting the contour 
        to the closed plasma surface and the divertor legs.

        Keyword Args:
            mask (Boolean): Default False.  Set True to mask LCFS path to 
                limiter outline (using inPolygon).  Set False to draw full 
                contour of psi = psiLCFS.

        Raises:
            NotImplementedError: if :py:mod:`matplotlib.pyplot` is not loaded.
            ValueError: if limiter outline is not available.
        """
        if not _has_plt:
            raise NotImplementedError("Requires matplotlib.pyplot for contour calculation.")
            
        try:
            Rlim,Zlim = self.getMachineCrossSection()
        except:
            raise ValueError("Limiter outline (self.getMachineCrossSection) must be available.")

        plt.ioff()
            
        psiRZ = self.getFluxGrid()  # [nt,nZ,nR]
        R = self.getRGrid()
        Z = self.getZGrid()
        psiLCFS = -1.0 * self.getCurrentSign() * self.getFluxLCFS()

        RLCFS_stores = []
        ZLCFS_stores = []
        maxlen = 0
        nt = len(self.getTimeBase())
        fig = plt.figure()
        for i in range(nt):
            cs = plt.contour(R,Z,psiRZ[i],[psiLCFS[i]])
            paths = cs.collections[0].get_paths()
            RLCFS_frame = []
            ZLCFS_frame = []
            for path in paths:
                v = path.vertices
                RLCFS_frame.extend(v[:,0])
                ZLCFS_frame.extend(v[:,1])
                RLCFS_frame.append(scipy.nan)
                ZLCFS_frame.append(scipy.nan)
            RLCFS_frame = scipy.array(RLCFS_frame)
            ZLCFS_frame = scipy.array(ZLCFS_frame)

            # generate masking array to vessel
            if mask:
                maskarr = scipy.array([False for i in range(len(RLCFS_frame))])
                for i,x in enumerate(RLCFS_frame):
                    y = ZLCFS_frame[i]
                    maskarr[i] = inPolygon(Rlim,Zlim,x,y)

                RLCFS_frame = RLCFS_frame[maskarr]
                ZLCFS_frame = ZLCFS_frame[maskarr]

            if len(RLCFS_frame) > maxlen:
                maxlen = len(RLCFS_frame)
            RLCFS_stores.append(RLCFS_frame)
            ZLCFS_stores.append(ZLCFS_frame)

        RLCFS = scipy.zeros((nt,maxlen))
        ZLCFS = scipy.zeros((nt,maxlen))
        for i in range(nt):
            RLCFS_frame = RLCFS_stores[i]
            ZLCFS_frame = ZLCFS_stores[i]
            ni = len(RLCFS_frame)
            RLCFS[i,0:ni] = RLCFS_frame
            ZLCFS[i,0:ni] = ZLCFS_frame

        # store final values
        self._RLCFS = RLCFS
        self._ZLCFS = ZLCFS

        # set default unit parameters, based on RZ grid
        rUnit = self._defaultUnits['_rGrid']
        zUnit = self._defaultUnits['_zGrid']
        self._defaultUnits['_RLCFS'] = rUnit
        self._defaultUnits['_ZLCFS'] = zUnit

        # cleanup
        plt.ion()
        plt.clf()
        plt.close(fig)
        plt.ioff()

    def getF(self):
        """returns F=RB_{\Phi}(\Psi), often calculated for grad-shafranov 
        solutions.
        
        Note that this method preserves whatever sign convention is used in the
        tree. For C-Mod, this means that the result should be multiplied by
        -1 * :py:meth:`getCurrentSign()` in most cases.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._fpol is None:
            try:
                fNode = self._MDSTree.getNode(self._root+self._gfile+':fpol')
                self._fpol = fNode.data()
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
                self._fluxPres = fluxPresNode.data()
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
                self._ffprim = FFPrimeNode.data()
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
                self._pprime = pPrimeNode.data()
                self._defaultUnits['_pprime'] = str(pPrimeNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._pprime.copy()

    def getElongation(self):
        """returns LCFS elongation.

        Returns:
            kappa (Array): [nt] array of LCFS elongation.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._kappa is None:
            try:
                kappaNode = self._MDSTree.getNode(self._root+self._afile+':eout')
                self._kappa = kappaNode.data()
                self._defaultUnits['_kappa'] = str(kappaNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._kappa.copy()

    def getUpperTriangularity(self):
        """returns LCFS upper triangularity.

        Returns:
            deltau (Array): [nt] array of LCFS upper triangularity.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._dupper is None:
            try:
                dupperNode = self._MDSTree.getNode(self._root+self._afile+':doutu')
                self._dupper = dupperNode.data()
                self._defaultUnits['_dupper'] = str(dupperNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dupper.copy()

    def getLowerTriangularity(self):
        """returns LCFS lower triangularity.

        Returns:
            deltal (Array): [nt] array of LCFS lower triangularity.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._dlower is None:
            try:
                dlowerNode = self._MDSTree.getNode(self._root+self._afile+':doutl')
                self._dlower = dlowerNode.data()
                self._defaultUnits['_dlower']  = str(dlowerNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dlower.copy()

    def getShaping(self):
        """pulls LCFS elongation and upper/lower triangularity.
        
        Returns:
            namedtuple containing (kappa, delta_u, delta_l)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            kap = self.getElongation()
            du = self.getUpperTriangularity()
            dl = self.getLowerTriangularity()
            data = namedtuple('Shaping',['kappa','delta_u','delta_l'])
            return data(kappa=kap,delta_u=du,delta_l=dl)
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius.

        Returns:
            magR (Array): [nt] array of major radius of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rmag is None:
            try:
                rmagNode = self._MDSTree.getNode(self._root+self._afile+':rmagx')
                self._rmag = rmagNode.data()
                self._defaultUnits['_rmag'] = str(rmagNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()

    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._zmag is None:
            try:
                zmagNode = self._MDSTree.getNode(self._root+self._afile+':zmagx')
                self._zmag = zmagNode.data()
                self._defaultUnits['_zmag'] = str(zmagNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zmag'], length_unit)
        return unit_factor * self._zmag.copy()

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
                areaLCFSNode = self._MDSTree.getNode(self._root+self._afile+':areao')
                self._areaLCFS = areaLCFSNode.data()
                self._defaultUnits['_areaLCFS'] = str(areaLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Units should be cm^2:
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_areaLCFS'], length_unit)
        return unit_factor * self._areaLCFS.copy()

    def getAOut(self, length_unit=1):
        """returns outboard-midplane minor radius at LCFS.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            aOut (Array): [nt] array of LCFS outboard-midplane minor radius.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._aLCFS is None:
            try:
                aLCFSNode = self._MDSTree.getNode(self._root+self._afile+':aout')
                self._aLCFS = aLCFSNode.data()
                self._defaultUnits['_aLCFS'] = str(aLCFSNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_aLCFS'], length_unit)
        return unit_factor * self._aLCFS.copy()

    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius.

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
                RmidLCFSNode = self._MDSTree.getNode(self._root+self._afile+':rmidout')
                self._RmidLCFS = RmidLCFSNode.data()
                # The units aren't properly stored in the tree for this one!
                # Should be meters.
                if RmidLCFSNode.units != ' ':
                    self._defaultUnits['_RmidLCFS'] = str(RmidLCFSNode.units)
                else:
                    self._defaultUnits['_RmidLCFS'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidLCFS'], length_unit)
        return unit_factor * self._RmidLCFS.copy()

    def getGeometry(self, length_unit=None):
        """pulls dimensional geometry parameters.
        
        Returns:
            namedtuple containing (magR,magZ,areaLCFS,aOut,RmidOut)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            Rmag = self.getMagR(length_unit=(length_unit if length_unit is not None else 1))
            Zmag = self.getMagZ(length_unit=(length_unit if length_unit is not None else 1))
            AreaLCFS = self.getAreaLCFS(length_unit=(length_unit if length_unit is not None else 2))
            aOut = self.getAOut(length_unit=(length_unit if length_unit is not None else 1))
            RmidOut = self.getRmidOut(length_unit=(length_unit if length_unit is not None else 1))
            data = namedtuple('Geometry',['Rmag','Zmag','AreaLCFS','aOut','RmidOut'])
            return data(Rmag=Rmag,Zmag=Zmag,AreaLCFS=AreaLCFS,aOut=aOut,RmidOut=RmidOut)
        except ValueError:
            raise ValueError('data retrieval failed.')

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
                self._qpsi = qpsiNode.data()
                self._defaultUnits['_qpsi'] = str(qpsiNode.units)
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    def getQ0(self):
        """returns q on magnetic axis,q0.

        Returns:
            q0 (Array): [nt] array of q(psi=0).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._q0 is None:
            try:
                q0Node = self._MDSTree.getNode(self._root+self._afile+':qqmagx')
                self._q0 = q0Node.data()
                self._defaultUnits['_q0'] = str(q0Node.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q0.copy()

    def getQ95(self):
        """returns q at 95% flux surface.

        Returns:
            q95 (Array): [nt] array of q(psi=0.95).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._q95 is None:
            try:
                q95Node = self._MDSTree.getNode(self._root+self._afile+':qpsib')
                self._q95 = q95Node.data()
                self._defaultUnits['_q95'] = str(q95Node.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q95.copy()

    def getQLCFS(self):
        """returns q on LCFS (interpolated).

        Returns:
            qLCFS (Array): [nt] array of q* (interpolated).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._qLCFS is None:
            try:
                qLCFSNode = self._MDSTree.getNode(self._root+self._afile+':qout')
                self._qLCFS = qLCFSNode.data()
                self._defaultUnits['_qLCFS'] = str(qLCFSNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._qLCFS.copy()

    def getQ1Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=1 surface.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            qr1 (Array): [nt] array of minor radius of q=1 surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rq1 is None:
            try:
                rq1Node = self._MDSTree.getNode(self._root+self._afile+':aaq1')
                self._rq1 = rq1Node.data()
                self._defaultUnits['_rq1'] = str(rq1Node.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq1'], length_unit)
        return unit_factor * self._rq1.copy()

    def getQ2Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=2 surface.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            qr2 (Array): [nt] array of minor radius of q=2 surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rq2 is None:
            try:
                rq2Node = self._MDSTree.getNode(self._root+self._afile+':aaq2')
                self._rq2 = rq2Node.data()
                self._defaultUnits['_rq2'] = str(rq2Node.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq2'], length_unit)
        return unit_factor * self._rq2.copy()

    def getQ3Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=3 surface.

        Keyword Args:
            length_unit (String or 1): unit for minor radius.  Defaults to 1, 
                denoting default length unit (typically m).

        Returns:
            qr3 (Array): [nt] array of minor radius of q=3 surface.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._rq3 is None:
            try:
                rq3Node = self._MDSTree.getNode(self._root+self._afile+':aaq3')
                self._rq3 = rq3Node.data()
                self._defaultUnits['_rq3'] = str(rq3Node.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq3'], length_unit)
        return unit_factor * self._rq3.copy()

    def getQs(self, length_unit=1):
        """pulls q values.
        
        Returns:
            namedtuple containing (q0,q95,qLCFS,rq1,rq2,rq3).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            q0 = self.getQ0()
            q95 = self.getQ95()
            qLCFS = self.getQLCFS()
            rq1 = self.getQ1Surf(length_unit=length_unit)
            rq2 = self.getQ2Surf(length_unit=length_unit)
            rq3 = self.getQ3Surf(length_unit=length_unit)
            data = namedtuple('Qs',['q0','q95','qLCFS','rq1','rq2','rq3'])
            return data(q0=q0,q95=q95,qLCFS=qLCFS,rq1=rq1,rq2=rq2,rq3=rq3)
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getBtVac(self):
        """Returns vacuum toroidal field on-axis.

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._btaxv is None:
            try:
                btaxvNode = self._MDSTree.getNode(self._root+self._afile+':btaxv')
                self._btaxv = btaxvNode.data()
                self._defaultUnits['_btaxv'] = str(btaxvNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxv.copy()

    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Returns:
            BtPla (Array): [nt] array of toroidal field including plasma effects.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._btaxp is None:
            try:
                btaxpNode = self._MDSTree.getNode(self._root+self._afile+':btaxp')
                self._btaxp = btaxpNode.data()
                self._defaultUnits['_btaxp'] = str(btaxpNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxp.copy()

    def getBpAvg(self):
        """returns average poloidal field.

        Returns:
            BpAvg (Array): [nt] array of average poloidal field.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._bpolav is None:
            try:
                bpolavNode = self._MDSTree.getNode(self._root+self._afile+':bpolav')
                self._bpolav = bpolavNode.data()
                self._defaultUnits['_bpolav'] = str(bpolavNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._bpolav.copy()

    def getFields(self):
        """pulls vacuum and plasma toroidal field, avg poloidal field.
        
        Returns:
            namedtuple containing (btaxv,btaxp,bpolav).

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            btaxv = self.getBtVac()
            btaxp = self.getBtPla()
            bpolav = self.getBpAvg()
            data = namedtuple('Fields',['BtVac','BtPla','BpAvg'])
            return data(BtVac=btaxv,BtPla=btaxp,BpAvg=bpolav)
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getIpCalc(self):
        """returns EFIT-calculated plasma current.

        Returns:
            IpCalc (Array): [nt] array of EFIT-reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpCalc is None:
            try:
                IpCalcNode = self._MDSTree.getNode(self._root+self._afile+':cpasma')
                self._IpCalc = IpCalcNode.data()
                self._defaultUnits['_IpCalc'] = str(IpCalcNode.units)
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpCalc.copy()

    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._IpMeas is None:
            try:
                IpMeasNode = self._MDSTree.getNode(self._root+self._afile+':pasmat')
                self._IpMeas = IpMeasNode.data()
                self._defaultUnits['_IpMeas'] = str(IpMeasNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpMeas.copy()

    def getJp(self):
        """returns EFIT-calculated plasma current density Jp on flux grid.

        Returns:
            Jp (Array): [nt,nz,nr] array of current density.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Jp is None:
            try:
                JpNode = self._MDSTree.getNode(self._root+self._gfile+':pcurrt')
                self._Jp = JpNode.data()
                # Units come in as 'a': am I missing something about the
                # definition of this quantity?
                self._defaultUnits['_Jp'] = str(JpNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Jp.copy()

    def getBetaT(self):
        """returns EFIT-calculated toroidal beta.

        Returns:
            BetaT (Array): [nt] array of EFIT-calculated average toroidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betat is None:
            try:
                betatNode = self._MDSTree.getNode(self._root+self._afile+':betat')
                self._betat = betatNode.data()
                self._defaultUnits['_betat'] = str(betatNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betat.copy()

    def getBetaP(self):
        """returns EFIT-calculated poloidal beta.

        Returns:
            BetaP (Array): [nt] array of EFIT-calculated average poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betap is None:
            try:
                betapNode = self._MDSTree.getNode(self._root+self._afile+':betap')
                self._betap = betapNode.data()
                self._defaultUnits['_betap'] = str(betapNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betap.copy()

    def getLi(self):
        """returns EFIT-calculated internal inductance.

        Returns:
            Li (Array): [nt] array of EFIT-calculated internal inductance.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Li is None:
            try:
                LiNode = self._MDSTree.getNode(self._root+self._afile+':ali')
                self._Li = LiNode.data()
                self._defaultUnits['_Li'] = str(LiNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Li.copy()

    def getBetas(self):
        """pulls calculated betap, betat, internal inductance
        
        Returns:
            namedtuple containing (betat,betap,Li)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            betat = self.getBetaT()
            betap = self.getBetaP()
            Li = self.getLi()
            data = namedtuple('Betas',['betat','betap','Li'])
            return data(betat=betat,betap=betap,Li=Li)
        except ValueError:
                raise ValueError('data retrieval failed.')

    def getDiamagFlux(self):
        """returns measured diamagnetic-loop flux.

        Returns:
            Flux (Array): [nt] array of diamagnetic-loop flux.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._diamag is None:
            try:
                diamagNode = self._MDSTree.getNode(self._root+self._afile+':diamag')
                self._diamag = diamagNode.data()
                self._defaultUnits['_diamag'] = str(diamagNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._diamag.copy()

    def getDiamagBetaT(self):
        """returns diamagnetic-loop toroidal beta.

        Returns:
            BetaT (Array): [nt] array of measured toroidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betatd is None:
            try:
                betatdNode = self._MDSTree.getNode(self._root+self._afile+':betatd')
                self._betatd = betatdNode.data()
                self._defaultUnits['_betatd'] = str(betatdNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betatd.copy()

    def getDiamagBetaP(self):
        """returns diamagnetic-loop avg poloidal beta.

        Returns:
            BetaP (Array): [nt] array of measured poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._betapd is None:
            try:
                betapdNode = self._MDSTree.getNode(self._root+self._afile+':betapd')
                self._betapd = betapdNode.data()
                self._defaultUnits['_betapd'] = str(betapdNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betapd.copy()

    def getDiamagTauE(self):
        """returns diamagnetic-loop energy confinement time.

        Returns:
            tauE (Array): [nt] array of measured energy confinement time.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._tauDiamag is None:
            try:
                tauDiamagNode = self._MDSTree.getNode(self._root+self._afile+':taudia')
                self._tauDiamag = tauDiamagNode.data()
                self._defaultUnits['_tauDiamag'] = str(tauDiamagNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._tauDiamag.copy()

    def getDiamagWp(self):
        """returns diamagnetic-loop plasma stored energy.

        Returns:
            Wp (Array): [nt] array of measured plasma stored energy.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._WDiamag is None:
            try:
                WDiamagNode = self._MDSTree.getNode(self._root+self._afile+':wplasmd')
                self._WDiamag = WDiamagNode.data()
                self._defaultUnits['_WDiamag'] = str(WDiamagNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WDiamag.copy()

    def getDiamag(self):
        """pulls diamagnetic flux measurements, toroidal and poloidal beta, 
        energy confinement time and stored energy.
        
        Returns:
            namedtuple containing (diamag. flux, betatd, betapd, tauDiamag, WDiamag)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            dFlux = self.getDiamagFlux()
            betatd = self.getDiamagBetaT()
            betapd = self.getDiamagBetaP()
            dTau = self.getDiamagTauE()
            dWp = self.getDiamagWp()
            data = namedtuple('Diamag',['diaFlux','diaBetat','diaBetap','diaTauE','diaWp'])
            return data(diaFlux=dFlux,diaBetat=betatd,diaBetap=betapd,diaTauE=dTau,diaWp=dWp)
        except ValueError:
                raise ValueError('data retrieval failed.')

    def getWMHD(self):
        """returns EFIT-calculated MHD stored energy.

        Returns:
            WMHD (Array): [nt] array of EFIT-calculated stored energy.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._WMHD is None:
            try:
                WMHDNode = self._MDSTree.getNode(self._root+self._afile+':wplasm')
                self._WMHD = WMHDNode.data()
                self._defaultUnits['_WMHD'] = str(WMHDNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WMHD.copy()

    def getTauMHD(self):
        """returns EFIT-calculated MHD energy confinement time.

        Returns:
            tauMHD (Array): [nt] array of EFIT-calculated energy confinement time.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._tauMHD is None:
            try:
                tauMHDNode = self._MDSTree.getNode(self._root+self._afile+':taumhd')
                self._tauMHD = tauMHDNode.data()
                self._defaultUnits['_tauMHD'] = str(tauMHDNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._tauMHD.copy()

    def getPinj(self):
        """returns EFIT-calculated injected power.

        Returns:
            Pinj (Array): [nt] array of EFIT-reconstructed injected power.

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Pinj is None:
            try:
                PinjNode = self._MDSTree.getNode(self._root+self._afile+':pbinj')
                self._Pinj = PinjNode.data()
                self._defaultUnits['_Pinj'] = str(PinjNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Pinj.copy()

    def getWbdot(self):
        """returns EFIT-calculated d/dt of magnetic stored energy.

        Returns:
            dWdt (Array): [nt] array of d(Wb)/dt

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Wbdot is None:
            try:
                WbdotNode = self._MDSTree.getNode(self._root+self._afile+':wbdot')
                self._Wbdot = WbdotNode.data()
                self._defaultUnits['_Wbdot'] = str(WbdotNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Wbdot.copy()

    def getWpdot(self):
        """returns EFIT-calculated d/dt of plasma stored energy.

        Returns:
            dWdt (Array): [nt] array of d(Wp)/dt

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Wpdot is None:
            try:
                WpdotNode = self._MDSTree.getNode(self._root+self._afile+':wpdot')
                self._Wpdot = WpdotNode.data()
                self._defaultUnits['_Wpdot'] = str(WpdotNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Wpdot.copy()

    def getBCentr(self):
        """returns EFIT-Vacuum toroidal magnetic field in Tesla at Rcentr

        Returns:
            B_cent (Array): [nt] array of B_t at center [T]

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._BCentr is None:
            try:
                BCentrNode = self._MDSTree.getNode(self._root+self._gfile+':bcentr')
                self._BCentr = BCentrNode.data()
                self._defaultUnits['_BCentr'] = str(BCentrNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._BCentr.copy()

    def getRCentr(self, length_unit=1):
        """returns EFIT radius where Bcentr evaluated

        Returns:
            R: Radial position where Bcent calculated [m]

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        if self._Rcentr is None:
            try:
                RCentrNode = self._MDSTree.getNode(self._root+self._gfile+':rcentr')
                self._RCentr = RCentrNode.data()
                self._defaultUnits['_RCentr'] = str(RCentrNode.units)
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')        
    
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RCentr'], length_unit)
        return unit_factor * self._RCentr.copy()

    def getEnergy(self):
        """pulls EFIT-calculated energy parameters - stored energy, tau_E, 
        injected power, d/dt of magnetic and plasma stored energy.
        
        Returns:
            namedtuple containing (WMHD,tauMHD,Pinj,Wbdot,Wpdot)

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """
        try:
            WMHD = self.getWMHD()
            tauMHD = self.getTauMHD()
            Pinj = self.getPinj()
            Wbdot = self.getWbdot()
            Wpdot = self.getWpdot()
            data = namedtuple('Energy',['WMHD','tauMHD','Pinj','Wbdot','Wpdot'])
            return data(WMHD=WMHD,tauMHD=tauMHD,Pinj=Pinj,Wbdot=Wbdot,Wpdot=Wpdot)
        except ValueError:
            raise ValueError('data retrieval failed.')
            
    def getMachineCrossSection(self):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting 
        routines.
        
        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        if self._Rlimiter is None or self._Zlimiter is None:
            try:
                limitr = self._MDSTree.getNode(self._root+self._gfile+':limitr').data()
                xlim = self._MDSTree.getNode(self._root+self._gfile+':xlim').data()
                ylim = self._MDSTree.getNode(self._root+self._gfile+':ylim').data()
                npts = len(xlim)
                
                if npts < limitr:
                    raise ValueError("Dimensions inconsistent in limiter array lengths.")
                    
                self._Rlimiter = xlim[0:limitr]
                self._Zlimiter = ylim[0:limitr]
            except (TreeException, AttributeError):
                raise ValueError("data retrieval failed.")
        return (self._Rlimiter,self._Zlimiter)

    def getMachineCrossSectionFull(self):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """
        try:
            return self.getMachineCrossSection()
        except:
            raise NotImplementedError("self.getMachineCrossSection not implemented.")

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's 
        IDL implementation efit_rz2psi.pro.

        Returns:
            currentSign (Integer): 1 for positive-direction current, -1 for negative.
        """
        if self._currentSign is None:
            self._currentSign = 1 if scipy.mean(self.getIpMeas()) > 1e5 else -1
        return self._currentSign

    def getParam(self, path):
        """Backup function, applying a direct path input for tree-like data 
        storage access for parameters not typically found in 
        :py:class:`Equilbrium <eqtools.core.Equilbrium>` object.  
        Directly calls attributes read from g/a-files in copy-safe manner.

        Args:
            name (String): Parameter name for value stored in EqdskReader 
                instance.

        Raises:
            AttributeError: raised if no attribute is found.
        """
        if self._root in path:
            EFITpath = path
        else:
            EFITpath = self._root+path

        try:
            var = self._MDSTree.getNode(EFITpath).data()
            return var
        except AttributeError:
            raise ValueError('invalid MDS tree.')
        except TreeException:
            raise ValueError('path '+EFITpath+' is not valid.')
