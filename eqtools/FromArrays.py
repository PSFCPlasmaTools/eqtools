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

from .core import Equilibrium

import scipy

class ArrayEquilibrium(Equilibrium):
    """Class to represent an equilibrium specified as arrays of data.
    
    Create ArrayEquilibrium instance from arrays of data.
    
    Has very little checking on the shape/type of the arrays at this point.
    
    Args:
        psiRZ: Array-like, (M, N, P).
            Flux values at M times, N Z locations and P R locations.
        rGrid: Array-like, (P,).
            R coordinates that psiRZ is given at.
        zGrid: Array-like, (N,).
            Z coordinates that psiRZ is given at.
        time: Array-like, (M,).
            Times that psiRZ is given at.
        q: Array-like, (S, M).
            q profile evaluated at S values of psinorm from 0 to 1, given at M
            times.
        fluxVol: Array-like, (S, M).
            Flux surface volumes evaluated at S values of psinorm from 0 to 1,
            given at M times.
        psiLCFS: Array-like, (M,).
            Flux at the last closed flux surface, given at M times.
        psiAxis: Array-like, (M,).
            Flux at the magnetic axis, given at M times.
        rmag: Array-like, (M,).
            Radial coordinate of the magnetic axis, given at M times.
        zmag: Array-like, (M,).
            Vertical coordinate of the magnetic axis, given at M times.
        Rout: Outboard midplane radius of the last closed flux surface.
    
    Keyword Args:
        length_unit: String.
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
        tspline: Boolean.
            Sets whether or not interpolation in time is
            performed using a tricubic spline or nearest-neighbor
            interpolation. Tricubic spline interpolation requires at least
            four complete equilibria at different times. It is also assumed
            that they are functionally correlated, and that parameters do
            not vary out of their boundaries (derivative = 0 boundary
            condition). Default is False (use nearest neighbor interpolation).
        monotonic: Boolean.
            Sets whether or not the "monotonic" form of time window
            finding is used. If True, the timebase must be monotonically
            increasing. Default is False (use slower, safer method).
        verbose: Boolean.
            Allows or blocks console readout during operation.  Defaults to True,
            displaying useful information for the user.  Set to False for quiet
            usage or to avoid console clutter for multiple instances.
    """
    def __init__(self, psiRZ, rGrid, zGrid, time, q, fluxVol, psiLCFS, psiAxis,
                 rmag, zmag, Rout, **kwargs):
        self._psiRZ = scipy.asarray(psiRZ, dtype=float)
        self._rGrid = scipy.asarray(rGrid, dtype=float)
        self._zGrid = scipy.asarray(zGrid, dtype=float)
        self._time = scipy.asarray(time, dtype=float)
        self._qpsi = scipy.asarray(q, dtype=float)
        self._fluxVol = scipy.asarray(fluxVol, dtype=float)
        self._psiLCFS = scipy.asarray(psiLCFS, dtype=float)
        self._psiAxis = scipy.asarray(psiAxis, dtype=float)
        self._rmag = scipy.asarray(rmag, dtype=float)
        self._zmag = scipy.asarray(zmag, dtype=float)
        self._RmidLCFS = scipy.asarray(Rout, dtype=float)
        
        self._defaultUnits = {}
        self._defaultUnits['_psiRZ'] = 'Wb/rad'
        self._defaultUnits['_rGrid'] = 'm'
        self._defaultUnits['_zGrid'] = 'm'
        self._defaultUnits['_time'] = 's'
        self._defaultUnits['_qpsi'] = ' '
        self._defaultUnits['_fluxVol'] = 'm^3'
        self._defaultUnits['_psiLFCS'] = 'Wb/rad'
        self._defaultUnits['_psiAxis'] = 'Wb/rad'
        self._defaultUnits['_rmag'] = 'm'
        self._defaultUnits['_zmag'] = 'm'
        self._defaultUnits['_RmidLCFS'] = 'm'
        self._defaultUnits['_RLCFS'] = 'm'
        self._defaultUnits['_ZLCFS'] = 'm'
        
        super(ArrayEquilibrium, self).__init__(**kwargs)
    
    def getTimeBase(self):
        """Returns a copy of the time base vector, array dimensions are (M,).
        """
        return self._time.copy()
    
    def getFluxGrid(self):
        """Returns a copy of the flux array, dimensions are (M, N, P), corresponding to (time, Z, R).
        """
        return self._psiRZ.copy()
    
    def getRGrid(self, length_unit=1):
        """Returns a copy of the radial grid, dimensions are (P,).
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'],
                                                      length_unit)
        return unit_factor * self._rGrid.copy()
    
    def getZGrid(self, length_unit=1):
        """Returns a copy of the vertical grid, dimensions are (N,).
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zGrid'],
                                                      length_unit)
        return unit_factor * self._zGrid.copy()
    
    def getQProfile(self):
        """Returns safety factor q profile (over Q values of psinorm from 0 to 1), dimensions are (Q, M)
        """
        return self._qpsi.copy()
    
    def getFluxVol(self, length_unit=3):
        """returns volume within flux surface [psi,t]
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_fluxVol'], length_unit)
        return unit_factor * self._fluxVol.copy()
    
    def getFluxLCFS(self):
        """returns psi at separatrix [t]
        """
        return self._psiLCFS.copy()
    
    def getFluxAxis(self):
        """returns psi on magnetic axis [t]
        """
        return self._psiAxis.copy()
    
    def getMagR(self, length_unit=1):
        """returns magnetic-axis major radius [t]
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()
    
    def getMagZ(self, length_unit=1):
        """returns magnetic-axis Z [t]
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zmag'], length_unit)
        return unit_factor * self._zmag.copy()
    
    def getRmidOut(self, length_unit=1):
        """returns outboard-midplane major radius [t]
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidLCFS'], length_unit)
        return unit_factor * self._RmidLCFS.copy()
    
    def getRLCFS(self, length_unit=1):
        raise NotImplementedError("getRLCFS not supported for ArrayEquilibrium!")
    
    def getZLCFS(self, length_unit=1):
        raise NotImplementedError("getRLCFS not supported for ArrayEquilibrium!")
    
    def getCurrentSign(self):
        return 1