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
    
    Args:
        psiRZ: Array-like, (M, N, P).
            Flux values at M times, N Z locations and P R locations.
        rGrid: Array-like, (P,).
            R coordinates that psiRZ is given at.
        zGrid: Array-like, (N,).
            Z coordinates that psiRZ is given at.
        time: Array-like, (M,).
            Times that psiRZ is given at.
        q: Array-like, (Q, M).
            q profile evaluated at Q values of psinorm from 0 to 1, given at M
            times.
        fluxVol: Array-like, (S, M).
            Flux surface volumes evaluated at S values of psinorm from 0 to 1,
            given at M times.
    
    Keyword Args:
        length_unit: String.
            Base unit for any quantity whose dimensions are length to any power.
            Default is 'm'. Valid options are:
            
            ===========  ===========
            'm'          meters
            'cm'         centimeters
            'mm'         millimeters
            'in'         inches
            'ft'         feet
            'yd'         yards
            'smoot'      smoots
            'cubit'      cubits
            'hand'       hands
            'default'    meters
            ===========  ===========
    """
    def __init__(self, psiRZ, rGrid, zGrid, time, q, fluxVol,
                 length_unit='m', tspline=False, fast=False):
        self._psiRZ = scipy.asarray(psiRZ, dtype=float)
        self._rGrid = scipy.asarray(rGrid, dtype=float)
        self._zGrid = scipy.asarray(zGrid, dtype=float)
        self._time = scipy.asarray(time, dtype=float)
        self._qpsi = scipy.asarray(q, dtype=float)
        self._fluxVol = scipy.asarray(fluxVol, dtype=float)
        
        self._defaultUnits = {}
        self._defaultUnits['_psiRZ'] = 'Wb/rad'
        self._defaultUnits['_rGrid'] = 'm'
        self._defaultUnits['_zGrid'] = 'm'
        self._defaultUnits['_time'] = 's'
        self._defaultUnits['_qpsi'] = ' '
        self._defaultUnits['_fluxVol'] = 'm^3'
    
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
        """Returns safety factor q profile (over Q values of psinorm from 0 to 1),
        dimensions are (Q, M)
        """
        return self._qpsi.copy()
