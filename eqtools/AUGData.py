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

"""This module provides classes inheriting :py:class:`eqtools.Equilibrium` for 
working with ASDEX Upgrade experimental data.
"""

import scipy

from .core import PropertyAccessMixin, ModuleWarning, Equilibrium

import warnings

try:
    import dd
    from dd import PyddError
    _has_dd = True

except Exception as _e_dd:
    if isinstance(_e_dd, ImportError):
        warnings.warn("dd module could not be loaded -- classes that use "
                     "dd for data access will not work.",
                      ModuleWarning)
    else:
        warnings.warn("dd module could not be loaded -- classes that use "
                      "dd for data access will not work. Exception raised "
                      "was of type %s, message was '%s'."
                      % (_e_dd.__class__, _e_dd.message),
                      ModuleWarning)
    _has_dd = False
    
try:
    import matplotlib.pyplot as plt
    _has_plt = True
except:
    warnings.warn("Matplotlib.pyplot module could not be loaded -- classes that "
                  "use pyplot will not work.",ModuleWarning)
    _has_plt = False


class AUGDDData(Equilibrium):
    """Inherits :py:class:`eqtools.Equilibrium` class. Machine-specific data
    handling class for ASDEX Upgrade. Pulls AFS data from selected location
    and shotfile, stores as object attributes. Each data variable or set of
    variables is recovered with a corresponding getter method. Essential data
    for mapping are pulled on initialization (e.g. psirz grid). Additional
    data are pulled at the first request and stored for subsequent usage.
    
    Intializes ASDEX Upgrade version of the Equilibrium object.  Pulls data to 
    storage in instance attributes.  Core attributes are populated from the AFS 
    data on initialization.  Additional attributes are initialized as None, 
    filled on the first request to the object.

    Args:
        shot (integer): ASDEX Upgrade shot index.
    
    Keyword Args:
        shotfile (string): Optional input for alternate shotfile, defaults to 'EQH'
            (i.e., CLISTE results are in EQH,EQI with other reconstructions
            Available (FPP, EQE, ect.).
        edition (integer): Describes the edition of the shotfile to be used
        shotfile2 (string): Describes companion 0D equilibrium data, will automatically
            reference based off of shotfile, but can be manually specified for 
            unique reconstructions, etc.
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
        experiment: Used to describe the work space that the shotfile is located
            It defaults to 'AUGD' but can be set to other values
    """ 

    # its like relating g files to a files
    _relatedSVFile = {'EQI':'GQI','EQH':'GQH','EQE':'GQE','FPP':'GPI'}

   
    def __init__(self, shot, shotfile='EQH', edition = 0, shotfile2 = None, length_unit='m', tspline = False,
                 monotonic=True, experiment='AUGD'):

        if not _has_dd:
            print("dd module did not load properly")
            print(
                "Most functionality will not be available!"
            )

        super(AUGDDData, self).__init__(length_unit=length_unit, tspline=tspline, 
                                        monotonic=monotonic)
        
        self._shot = shot
        self._tree = shotfile
        print(self._shot,self._tree,edition,experiment)
        self._MDSTree = dd.shotfile(self._tree,
                                    self._shot,
                                    edition=edition,
                                    experiment=experiment)
        
        try:
            self._MDSTree2 = dd.shotfile(self._relatedSVFile[self._tree],
                                         self._shot,
                                         edition=edition,
                                         experiment=experiment)
        except KeyError:
            self._MDSTree2 = dd.shotfile(shotfile2,
                                         self._shot,
                                         edition=edition,
                                         experiment=experiment)
        except PyddError:
            raise ValueError('Specify valid companion SV shotfile.')

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
        self._IpCalc = None                                                  #calculated plasma current (t)
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
        self._betat = None                                                   #calc toroidal beta (t)
        self._betap = None                                                   #calc avg. poloidal beta (t)
        self._Li = None                                                      #calc internal inductance (t)

        #diamagnetic measurements
        self._diamag = None                                                  #diamagnetic flux (t)
        self._betatd = None                                                  #diamagnetic toroidal beta (t)
        self._betapd = None                                                  #diamagnetic poloidal beta (t)
        self._WDiamag = None                                                 #diamagnetic stored energy (t)
        self._tauDiamag = None                                               #diamagnetic energy confinement time (t)

        #energy calculations
        self._WMHD = None                                                    #calc stored energy (t)
        self._tauMHD = None                                                  #calc energy confinement time (t)
        self._Pinj = None                                                    #calc injected power (t)
        self._Wbdot = None                                                   #d/dt magnetic stored energy (t)
        self._Wpdot = None                                                   #d/dt plasma stored energy (t)

        #load essential mapping data
        # Set the variables to None first so the loading calls will work right:
        self._time = None                                                    #timebase
        self._psiRZ = None                                                   #flux grid (r,z,t)
        self._rGrid = None                                                   #R-axis (t)
        self._zGrid = None                                                   #Z-axis (t)
        self._psiLCFS = None                                                 #flux at LCFS (t)
        self._psiAxis = None                                                 #flux at magnetic axis (t)
        self._fluxVol = None                                                 #volume within flux surface (t,psi)
        self._volLCFS = None                                                 #volume within LCFS (t)
        self._qpsi = None                                                    #q profile (psi,t)
        self._RmidPsi = None                                                 #max major radius of flux surface (t,psi)
        
        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        self.getTimeBase() # check
        self._timeidxend = self.getTimeBase().size
        self.getFluxGrid() # loads _psiRZ, _rGrid and _zGrid at once. check
        self.getFluxLCFS() # check
        self.getFluxAxis() # check
        self.getFluxVol() #check
        self._lpf = self.getFluxVol().shape[1]
        self.getVolLCFS() # check
        self.getQProfile() #
        
    def __str__(self):
        """string formatting for ASDEX Upgrade Equilibrium class.
        """
        try:
            nt = len(self._time)
            nr=  len(self._rGrid)
            nz = len(self._zGrid)

            mes = 'AUG data for shot '+str(self._shot)+' from shotfile '+str(self._tree.upper())+'\n'+\
                  'timebase '+str(self._time[0])+'-'+str(self._time[-1])+'s in '+str(nt)+' points\n'+\
                  str(nr)+'x'+str(nz)+' spatial grid'
            return mes
        except TypeError:
            return 'tree has failed data load.'
        
    def getInfo(self):
        """returns namedtuple of shot information
        
        Returns:
            namedtuple containing
                
                =====   ===============================
                shot    ASDEX Upgrage shot index (long)
                tree    shotfile (string)
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
        """returns time base vector.

        Returns:
            time (array): [nt] array of time points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._time is None:
            try:
                timeNode = self._MDSTree('time')
                self._time = timeNode.data
                self._defaultUnits['_time'] = str(timeNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._time.copy()

    def getFluxGrid(self):
        """returns flux grid.
        
        Note that this method preserves whatever sign convention is used in AFS.
        
        Returns:
            psiRZ (Array): [nt,nz,nr] array of (non-normalized) flux on grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiRZ is None:
            try:
                psinode = self._MDSTree('PFM',calibrated=False) #calibrated signal causes seg faults (SERIOUSLY WHAT THE FUCK ASDEX)
                self._psiRZ = psinode.data[:self._timeidxend]
                self._defaultUnits['_psiRZ'] = 'Vs' #HARDCODED DUE TO CALIBRATED=FALSE
                psinode = self._MDSTree('Ri')
                self._rGrid = psinode.data[0] #assumes data from first is correct (WHY IS IT EVEN DUPICATED???)
                self._defaultUnits['_rGrid'] = str(psinode.unit)
                psinode = self._MDSTree('Zj')
                self._zGrid = psinode.data[0]
                self._defaultUnits['_zGrid'] = str(psinode.unit)

            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._psiRZ.copy()

    def getRGrid(self, length_unit=1):
        """returns R-axis.

        Returns:
            rGrid (Array): [nr] array of R-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._rGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'],
                                                      length_unit)
        return unit_factor * self._rGrid.copy()

    def getZGrid(self, length_unit=1):
        """returns Z-axis.

        Returns:
            zGrid (Array): [nz] array of Z-axis of flux grid.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiAxis is None:
            try:
                psiAxisNode = self._MDSTree('PFxx')
                self._psiAxis = psiAxisNode.data[:self._timeidxend,0]
                self._defaultUnits['_psiAxis'] = str(psiAxisNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._psiAxis.copy()

    def getFluxLCFS(self):
        """returns psi at separatrix.

        Returns:
            psiLCFS (Array): [nt] array of psi at LCFS.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._psiLCFS is None:
            try:
                psiLCFSNode = self._MDSTree('PFL')
                self._psiLCFS = psiLCFSNode.data[:self._timeidxend,0]
                self._defaultUnits['_psiLCFS'] = str(psiLCFSNode.unit)
            except PyddError:
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fluxVol is None:
            try:
                fluxVolNode = self._MDSTree('Vol') #Lpf is unreliable so I have to do this trick....
                temp = scipy.where(scipy.sum(fluxVolNode.data,axis=0)[::2] !=0)[0].max() + 1 #Find the where the volume is non-zero, give the maximum index and add one (for the core value)
                
                self._fluxVol = fluxVolNode.data[:self._timeidxend][:,:2*temp+1:2][:,::-1] #reverse it so that it is a monotonically increasing function
                if fluxVolNode.unit != ' ':
                    self._defaultUnits['_fluxVol'] = str(fluxVolNode.unit)
                else:
                    self._defaultUnits['_fluxVol'] = 'm^3'
            except PyddError:
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree('Vol')
                self._volLCFS = volLCFSNode.data[:self._timeidxend,0]
                self._defaultUnits['_volLCFS'] = str(volLCFSNode.unit)
            except PyddError:
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
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getRmidPsi not implemented.")
       

    def getRLCFS(self, length_unit=1):
        """returns R-values of LCFS position.

        Returns:
            RLCFS (Array): [nt,n] array of R of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._RLCFS is None:
            try:
                rgeo = self._MDSTree2('Rgeo')
                RLCFSNode = self._MDSTree2('rays')
                RLCFStemp = scipy.hstack((scipy.atleast_2d(RLCFSNode.data[:,-1]).T,RLCFSNode.data))
                templen = RLCFSNode.data.shape

                self._RLCFS = scipy.tile(rgeo.data,(templen[1]+1,1)).T + RLCFStemp*scipy.cos(scipy.tile((scipy.linspace(0,2*scipy.pi,templen[1]+1)),(templen[0],1))) #construct a 2d grid of angles, take cos, multiply by radius
                
                self._defaultUnits['_RLCFS'] = str(RLCFSNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RLCFS'], length_unit)
        return unit_factor * self._RLCFS.copy()

    def getZLCFS(self, length_unit=1):
        """returns Z-values of LCFS position.

        Returns:
            ZLCFS (Array): [nt,n] array of Z of LCFS points.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._ZLCFS is None:
            try:
                zgeo = self._MDSTree2('Zgeo')
                ZLCFSNode = self._MDSTree2('rays')
                ZLCFStemp = scipy.hstack((scipy.atleast_2d(ZLCFSNode.data[:,-1]).T,ZLCFSNode.data))
                templen = ZLCFSNode.data.shape
                
                self._ZLCFS =  scipy.tile(zgeo.data,(templen[1]+1,1)).T + ZLCFStemp*scipy.sin(scipy.tile((scipy.linspace(0,2*scipy.pi,templen[1]+1)),(templen[0],1))) #construct a 2d grid of angles, take sin, multiply by radius
                self._defaultUnits['_ZLCFS'] = str(ZLCFSNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_ZLCFS'], length_unit)
        return unit_factor * self._ZLCFS.copy()
        
    def remapLCFS(self,mask=False):
        """Overwrites RLCFS, ZLCFS values pulled with explicitly-calculated 
        contour of psinorm=1 surface.  This is then masked down by the limiter
        array using core.inPolygon, restricting the contour to the closed
        plasma surface and the divertor legs.

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
        psiLCFS = self.getFluxLCFS()

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
        
        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fpol is None:
            try:
                fNode = self._MDSTree('Jpol')#From definition of F with poloidal current
                self._fpol = fNode.data[:self._timeidxend,:2*self._lpf:2][::-1]*2e-7
                self._defaultUnits['_fpol'] = str('T m')
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._fpol.copy()

    def getFluxPres(self):
        """returns pressure at flux surface.

        Returns:
            p (Array): [nt,npsi] array of pressure on flux surface psi.

        Raises:
            ValueError: if module cannot retrieve data from AUG AFS system.
        """
        if self._fluxPres is None:
            try:
                fluxPresNode = self._MDSTree('Pres')
                self._fluxPres = fluxPresNode.data[:self._timeidxend][:,:2*self._lpf:2][:,::-1] #reverse it so that it is a monotonically increasing function
                self._defaultUnits['_fluxPres'] = str(fluxPresNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._fluxPres.copy()

    def getFPrime(self):
        """returns F', often calculated for grad-shafranov 
        solutions.

        Returns:
            F (Array): [nt,npsi] array of F=RB_{\Phi}(\Psi)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._fpol is None:
            try:
                fNode = self._MDSTree('Jpol')#From definition of F with poloidal current
                self._fpol = fNode.data[:self._timeidxend,1:2*self._lpf+1:2][::-1]*2e-7
                self._defaultUnits['_fpol'] = str('T m')
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._fpol.copy()

    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FFprime (Array): [nt,npsi] array of FF' fromgrad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._ffprim is None:
            try:
                FFPrimeNode = self._MDSTree('FFP')
                self._ffprim = FFPrimeNode.data[:self._timeidxend,:self._lpf][::-1]
                self._defaultUnits['_ffprim'] = str(FFPrimeNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._ffprim.copy()

    def getPPrime(self):
        """returns plasma pressure gradient as a function of psi.

        Returns:
            pprime (Array): [nt,npsi] array of pressure gradient on flux surface 
            psi from grad-shafranov solution.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._pprime is None:
            try:
                pPrimeNode = self._MDSTree('Pres')
                self._pprime = pPrimeNode.data[:self._timeidxend][:,1:2*self._lpf+1:2][:,::-1] #reverse it so that it is a monotonically increasing function
                self._defaultUnits['_pprime'] = str(pPrimeNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._pprime.copy()

    def getElongation(self):
        """returns LCFS elongation.

        Returns:
            kappa (Array): [nt] array of LCFS elongation.

        Raises:
            ValueError: if module cannot retrieve data from AFS.
        """
        if self._kappa is None:
            try:
                kappaNode = self._MDSTree2('k')
                self._kappa = kappaNode.data
                self._defaultUnits['_kappa'] = str(kappaNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._kappa.copy()

    def getUpperTriangularity(self):
        """returns LCFS upper triangularity.

        Returns:
            deltau (Array): [nt] array of LCFS upper triangularity.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._dupper is None:
            try:
                dupperNode = self._MDSTree2('delRoben')
                self._dupper = dupperNode.data
                self._defaultUnits['_dupper'] = str(dupperNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._dupper.copy()

    def getLowerTriangularity(self):
        """returns LCFS lower triangularity.

        Returns:
            deltal (Array): [nt] array of LCFS lower triangularity.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._dlower is None:
            try:
                dlowerNode = self._MDSTree2('delRuntn')
                self._dlower = dlowerNode.data
                self._defaultUnits['_dlower']  = str(dlowerNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._dlower.copy()

    def getShaping(self):
        """pulls LCFS elongation and upper/lower triangularity.
        
        Returns:
            namedtuple containing (kappa, delta_u, delta_l)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._rmag is None:
            try:
                rmagNode = self._MDSTree2('Rmag')
                self._rmag = rmagNode.data
                self._defaultUnits['_rmag'] = str(rmagNode.unit)
            except (PyddError,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()

    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z.

        Returns:
            magZ (Array): [nt] array of Z of magnetic axis.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._zmag is None:
            try:
                zmagNode = self._MDSTree2('Zmag')
                self._zmag = zmagNode.data
                self._defaultUnits['_zmag'] = str(zmagNode.unit)
            except PyddError:
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._areaLCFS is None:
            try:
                areaLCFSNode = self._MDSTree('Area')
                self._areaLCFS = areaLCFSNode.data[:self._timeidxend,0]
                self._defaultUnits['_areaLCFS'] = str(areaLCFSNode.unit)
            except PyddError:
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._aLCFS is None:
            try:
                aLCFSNode = self._MDSTree2('ahor')
                self._aLCFS = aLCFSNode.data
                self._defaultUnits['_aLCFS'] = str(aLCFSNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_aLCFS'], length_unit)
        return unit_factor * self._aLCFS.copy()

    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius.

        Keyword Args:
            length_unit (String or 1): unit for major radius.  Defaults to 1, 
                denoting default length unit (typically m).
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getRmidOut not implemented.")



    def getGeometry(self, length_unit=None):
        """pulls dimensional geometry parameters.
        
        Returns:
            namedtuple containing (magR,magZ,areaLCFS,aOut,RmidOut)

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
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
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._qpsi is None:
            try:
                qpsiNode = self._MDSTree('Qpsi')
                self._qpsi = qpsiNode.data[:self._timeidxend,:self._lpf]
                self._defaultUnits['_qpsi'] = str(qpsiNode.unit)
            except PyddError:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    def getQ0(self):
        """returns q on magnetic axis,q0.

        Returns:
            q0 (Array): [nt] array of q(psi=0).

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._q0 is None:
            try:
                q0Node = self._MDSTree2('q0')
                self._q0 = q0Node.data
                self._defaultUnits['_q0'] = str(q0Node.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q0.copy()

    def getQ95(self):
        """returns q at 95% flux surface.

        Returns:
            q95 (Array): [nt] array of q(psi=0.95).

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._q95 is None:
            try:
                q95Node = self._MDSTree2('q95')
                self._q95 = q95Node.data
                self._defaultUnits['_q95'] = str(q95Node.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q95.copy()


    def getQLCFS(self):
        """returns q on LCFS (interpolated).
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQLCFS not implemented.")


    def getQ1Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=1 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ1Surf not implemented.")


    def getQ2Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=2 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ2Surf not implemented.")


    def getQ3Surf(self, length_unit=1):
        """returns outboard-midplane minor radius of q=3 surface.
       
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQ3Surf not implemented.")
     

    def getQs(self, length_unit=1):
        """pulls q values.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getQs not implemented.")


    def getBtVac(self):
        """Returns vacuum toroidal field on-axis. THIS MAY BE INCORRECT

        Returns:
            BtVac (Array): [nt] array of vacuum toroidal field.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._btaxv is None:
            try:
                btaxvNode = self._MDSTree('Bave')
                #technically Bave is the average over the volume, but for the core its a singular value
                self._btaxv = btaxvNode.data[:self._timeidxend,scipy.sum(btaxvNode.data,0) != 0][:,-1] 
                self._defaultUnits['_btaxv'] = str(btaxvNode.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxv.copy()


    def getBtPla(self):
        """returns on-axis plasma toroidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getBtPla not implemented.")


    def getBpAvg(self):
        """returns average poloidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getFields not implemented.")


    def getFields(self):
        """pulls vacuum and plasma toroidal field, avg poloidal field.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getFields not implemented.")
 

    def getIpCalc(self):
        """returns Plasma Current, is the same as getIpMeas.

        Returns:
            IpCalc (Array): [nt] array of the reconstructed plasma current.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._IpCalc is None:
            try:
                IpCalcNode = self._MDSTree('IpiPSI')
                self._IpCalc = scipy.squeeze(IpCalcNode.data)[:self._timeidxend]
                self._defaultUnits['_IpCalc'] = str(IpCalcNode.unit)
            except (PyddError,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpCalc.copy()


    def getIpMeas(self):
        """returns magnetics-measured plasma current.

        Returns:
            IpMeas (Array): [nt] array of measured plasma current.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        return self.getIpCalc()

    def getJp(self):
        """returns the calculated plasma current density Jp on flux grid.

        Returns:
            Jp (Array): [nt,nz,nr] array of current density.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._Jp is None:
            try:
                JpNode = self._MDSTree('CDM',calibrated=False)
                self._Jp = JpNode.data
                self._defaultUnits['_Jp'] = str(JpNode.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Jp.copy()


    def getBetaT(self):
        """returns the calculated toroidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getBetaT not implemented.")


    def getBetaP(self):
        """returns the calculated poloidal beta.

        Returns:
            BetaP (Array): [nt] array of the calculated average poloidal beta.

        Raises:
            ValueError: if module cannot retrieve data from the AUG AFS system.
        """
        if self._betap is None:
            try:
                betapNode = self._MDSTree2('betpol')
                self._betap = betapNode.data
                self._defaultUnits['_betap'] = str(betapNode.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betap.copy()


    def getLi(self):
        """returns the calculated internal inductance.

        Returns:
            Li (Array): [nt] array of the calculated internal inductance.

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._Li is None:
            try:
                LiNode = self._MDSTree2('li')
                self._Li = LiNode.data
                self._defaultUnits['_Li'] = str(LiNode.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Li.copy()


    def getBetas(self):
        """pulls calculated betap, betat, internal inductance.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getBetas not implemented.")


    def getDiamagFlux(self):
        """returns the measured diamagnetic-loop flux.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagFlux not implemented.")
    

    def getDiamagBetaT(self):
        """returns diamagnetic-loop toroidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagBetaT not implemented.")

    def getDiamagBetaP(self):
        """returns diamagnetic-loop avg poloidal beta.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagBetaP not implemented.")


    def getDiamagTauE(self):
        """returns diamagnetic-loop energy confinement time.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagTauE not implemented.")


    def getDiamagWp(self):
        """returns diamagnetic-loop plasma stored energy.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamagWp not implemented.")


    def getDiamag(self):
        """pulls diamagnetic flux measurements, toroidal and poloidal beta, 
        energy confinement time and stored energy.
        
        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getDiamag not implemented.")


    def getWMHD(self):
        """returns calculated MHD stored energy.

        Returns:
            WMHD (Array): [nt] array of the calculated stored energy.

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._WMHD is None:
            try:
                WMHDNode = self._MDSTree2('Wmhd')
                self._WMHD = WMHDNode.data
                self._defaultUnits['_WMHD'] = str(WMHDNode.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WMHD.copy()


    def getTauMHD(self):
        """returns the calculated MHD energy confinement time.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getTauMHD not implemented.")


    def getPinj(self):
        """returns the injected power.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
            .
        """
        raise NotImplementedError("self.getPinj not implemented.")
 

    def getWbdot(self):
        """returns the calculated d/dt of magnetic stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getWbdot not implemented.")


    def getWpdot(self):
        """returns the calculated d/dt of plasma stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getWpdot not implemented.")


    def getBCentr(self):
        """returns Vacuum toroidal magnetic field at center of plasma

        Returns:
            B_cent (Array): [nt] array of B_t at center [T]

        Raises:
            ValueError: if module cannot retrieve data from the AUG afs system.
        """
        if self._BCentr is None:
            try:
                temp = dd.shotfile('MBI',self._shot)
                BCentrNode = temp('BTFABB')
                self._BCentr = BCentrNode.data[self._getNearestIdx(self.getTimeBase(),BCentrNode.time)]
                self._defaultUnits['_BCentr'] = str(BCentrNode.unit)
            except (PyddError, AttributeError):
                raise ValueError('data retrieval failed.')

        return self._BCentr

    def getRCentr(self, length_unit=1):
        """Returns Radius of BCenter measurement

        Returns:
            R: Radial position where Bcent calculated [m]
        """
        if self._RCentr is None:
            self._RCentr = 1.65 #Hardcoded from MAI file description of BTF
            self._defaultUnits['_RCentr'] = 'm'
        return self._RCentr

    def getEnergy(self):
        """pulls the calculated energy parameters - stored energy, tau_E, 
        injected power, d/dt of magnetic and plasma stored energy.

        Raises:
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """
        raise NotImplementedError("self.getEnergy not implemented.")
            
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
                self._Rlimiter, self._Zlimiter = AUGVessel.getMachineCrossSection(self._shot)

            except (PyddError, AttributeError):
                raise ValueError("data retrieval failed.")
        return (self._Rlimiter,self._Zlimiter)

    def getMachineCrossSectionFull(self):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """
        x, y = AUGVessel.getMachineCrossSectionFull(self._shot)
        x[ x > self.getRGrid().max()] = self.getRGrid().max()
        
        return (x, y)


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
            NotImplementedError: Not implemented on ASDEX-Upgrade reconstructions.
        """ 
        raise NotImplementedError("self.getEnergy not implemented.")



class YGCAUGInterface(object):

    #============================================================================================================
    #
    #                     VESSEL OUTLINE HARDCODE VALUES DUE TO ASDEX UPGRADE INCONSISTENCIES
    #
    #============================================================================================================

    #Rather than use another dependency in code, this stores all the necessary interfacing from the data structure
    # the only necessary implementation is the data handler (dd python package)
    _vessel_components = {0:(1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          948:(1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                          8650:(1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0),
                          9401:(1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1),
                          12751:(1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1),
                          14051:(1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1),
                          14601:(1,1,0,0,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1),
                          16315:(1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1),
                          18204:(1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1),
                          19551:(1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1),
                          21485:(1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1),
                          25891:(1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1),
                          30136:(1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1)
                          } 

    #counter-clockwise from the inner wall order of components
    _order = {0:(9,8,7,5,2,1,14,13,12,0,4,6),
              948:(9,8,7,5,2,1,10,0,4,6),
              8650:(9,15,16,17,18,19,20,21,22,23,24,25,26,1,10,0,4,6),
              9401:(9,15,16,17,18,19,20,21,22,23,24,25,27,10,0,4,6),
              12751:(9,29,15,16,17,18,19,20,21,22,23,24,25,27,10,0,4,6),
              14051:(9,29,15,16,17,18,19,20,21,22,23,24,25,26,28,10,0,4,6),
              14601:(9,29,15,16,17,18,19,20,21,22,23,24,25,26,28,10,0,4,30,31,32,33,34),
              16315:(9,15,16,17,18,19,20,21,22,23,24,25,26,27,1,10,0,35,36,37,38,39,30,31,32,33,34),
              18204:(9,15,16,17,18,19,20,21,22,23,24,25,26,27,1,10,0,35,36,37,38,39,30,31,32,33,34),
              19551:(9,15,16,17,18,19,20,21,22,23,24,25,26,27,1,10,0,35,36,37,38,39,30,31,32,33,34),
              21485:(9,15,16,17,18,19,20,21,22,23,24,25,26,27,1,10,0,35,36,37,38,39,30,31,32,33,34),
              25891:(9,15,16,17,18,19,20,21,22,23,24,25,26,27,10,41,42,43,36,37,38,39,30,31,32,33,34),
              30136:(9,15,16,17,18,19,20,21,22,23,24,25,26,27,10,41,42,43,36,37,38,39,30,31,32,33,34)
              } 
    
   #start location in array of values for given object closest to plasma
    _start = {0:(21,0,3,3,1,4,2,3,3,9,0,1),
              948:(21,0,3,3,1,4,2,9,0,1),
              8650:(21,0,0,0,0,0,0,0,0,0,0,0,0,4,2,9,0,1),
              9401:(21,0,0,0,0,0,0,0,0,0,0,0,0,2,9,0,1),
              12751:(21,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,0,1),
              14051:(21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,0,1),
              14601:(21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,9,0,0,0,0,0,0),
              16315:(21,0,0,0,0,0,0,0,0,0,0,0,0,13,4,2,0,0,0,0,0,0,0,0,0,0,0),
              18204:(21,0,0,0,0,0,0,0,0,0,0,0,0,13,4,2,0,0,0,0,0,0,0,0,0,0,0),
              19551:(21,0,0,0,0,0,0,0,0,0,0,0,0,13,4,2,0,0,0,0,0,0,0,0,0,0,0),
              21485:(21,0,0,0,0,0,0,0,0,0,0,0,0,0,4,2,0,0,0,1,0,0,0,0,0,0,0),
              25891:(21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0),
              30136:(21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,1,0,0,0,0,0,0,0)
              }
     
   #end location in array of values for given object closest to plasma
    _end = {0:(42,2,7,7,5,6,10,5,10,13,4,5),
            948:(42,2,7,7,5,6,34,13,4,5),
            8650:(42,4,26,25,32,2,9,8,3,35,22,28,3,5,34,13,4,5),
            9401:(42,4,26,22,32,5,9,8,5,35,22,26,5,27,13,4,5),
            12751:(39,2,4,26,25,32,2,9,8,3,35,22,28,5,25,13,4,5),
            14051:(39,2,5,5,20,26,8,11,7,5,3,4,14,4,5,25,13,4,10),
            14601:(39,2,5,5,20,26,8,11,7,5,3,4,14,4,5,25,13,4,2,2,2,2,4),
            16315:(42,5,5,20,26,8,11,7,5,3,4,14,4,16,5,34,2,2,2,2,2,2,2,2,2,2,4),
            18204:(42,5,5,20,26,8,11,7,5,3,4,14,4,16,5,34,2,2,2,2,2,2,2,2,2,2,7),
            19551:(42,5,5,20,26,8,11,7,5,3,4,14,4,16,5,34,2,2,7,2,2,2,2,2,2,2,7),
            21485:(42,6,5,15,6,6,6,7,6,6,3,12,6,3,5,34,2,2,7,2,2,2,2,2,2,2,7),
            25891:(42,6,5,15,6,6,6,7,6,6,3,18,6,3,57,2,2,18,7,2,2,2,2,2,2,2,7),
            30136:(42,6,5,15,6,6,6,7,6,9,2,18,4,2,57,2,2,17,7,2,2,2,2,2,2,2,7)
            }

   #Which objects are stored reverse of the counter-clockwise motion as described in order
    _rev = {0:(0,1,1,1,1,1,1,1,1,1,1,1),
            948:(0,1,1,1,1,1,1,1,1,1),
            8650:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1),
            9401:(0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1),
            12751:(0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1),
            14051:(0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1),
            14601:(0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0),
            16315:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
            18204:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
            19551:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
            21485:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
            25891:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
            30136:(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
            }


    # ONLY CERTAIN YGC FILES EXIST I MEAN CMON ITS NOT THAT MUCH MEMORY
    _ygc_shotfiles = scipy.array([0, 948, 8650, 9401, 12751, 14051, 14601, 16315, 18204, 19551, 21485, 25891, 30136])
                  
    def _getData(self,shot):
        try:

            self._ygc_shot = self._ygc_shotfiles[scipy.searchsorted(self._ygc_shotfiles, [shot], 'right') - 1][0] #find nearest shotfile which is the before it

            if self._ygc_shot < 8650:
                ccT = dd.shotfile('YGC', self._ygc_shotfiles[2]) # This is because of shots <8650 not having RrGC zzGC or inxbeg
            else:
                ccT = dd.shotfile('YGC', self._ygc_shot)
            xvctr = ccT('RrGC')
            yvctr = ccT('zzGC')
            nvctr = ccT('inxbeg')
            nvctr = nvctr.data.astype(int)-1
            
        except (PyddError, AttributeError):
            raise ValueError("data retrieval failed.")

        except:
            raise ValueError('data load failed.')

        return xvctr.data,yvctr.data,nvctr

  
    def getMachineCrossSection(self,shot):
        """Returns R,Z coordinates of vacuum-vessel wall for masking, plotting 
        routines.
        
        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        xvctr,yvctr,nvctr = self._getData(shot)
        x = []
        y = []
        
        #by reference to simplify coding
        start = self._start[self._ygc_shot]
        end = self._end[self._ygc_shot]
        rev = self._rev[self._ygc_shot]
        order = self._order[self._ygc_shot]

        for i in xrange(len(order)):
            idx = nvctr[order[i]]
            xseg = xvctr[idx+start[i]:idx+end[i]] 
            yseg = yvctr[idx+start[i]:idx+end[i]]

            if rev[i]:
                xseg = xseg[::-1]
                yseg = yseg[::-1]
            
            x.extend(xseg)
            y.extend(yseg)

        x.extend([x[0]])
        y.extend([y[0]])

        return (x[::-1], y[::-1])        

    def getMachineCrossSectionFull(self, shot):
        """Returns R,Z coordinates of vacuum-vessel wall for plotting routines.
        
        Absent additional vector-graphic data on machine cross-section, returns
        :py:meth:`getMachineCrossSection`.
        
        Returns:
            result from getMachineCrossSection().
        """

        xvctr,yvctr,nvctr = self._getData(shot)
        
        # get valid components which is in the data structure for some shots, but not all and had to be hardcoded
        temp = self._vessel_components[self._ygc_shot]
        
        x = []
        y = []

        for i in xrange(len(nvctr)-1):
            if temp[i]:
                xseg = xvctr[nvctr[i]:nvctr[i+1]]
                yseg = yvctr[nvctr[i]:nvctr[i+1]]
                x.extend(xseg)
                y.extend(yseg)
                x.append(None)
                y.append(None)
                    
        x = scipy.array(x[:-1])
        y = scipy.array(y[:-1])
        return (x, y)


if _has_dd:
    AUGVessel = YGCAUGInterface() #import setting necessary to get the vacuum vessel

 
        
class AUGDDDataProp(AUGDDData, PropertyAccessMixin):
    """AUGDDData with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down.
    """
    pass
