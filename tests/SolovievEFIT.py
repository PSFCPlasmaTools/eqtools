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

"""This module provides a class for constructing an :py:class:`Equilibrium` object
built on the analytic Soloviev equilibrium for testing purposes.

Classes:
    SolvievEFIT: class inheriting :py:class:`Equilibrium` for generation of and
    mapping routines using the Soloviev solution to the Grad-Shafranov equation.
"""

import scipy
from collections import namedtuple
from .core import Equilibrium, ModuleWarning, inPolygon

import warnings

try:
    import matplotlib.pyplot as plt
    _has_plt = True
except:
    warnings.warn("Matplotlib.pyplot module could not be loaded -- classes that "
                  "use pyplot will not work.",ModuleWarning)
    _has_plt = False

class CircSolovievEFIT(Equilibrium):
    """Equilibrium class working with analytic Soloviev equilibria, restricted
    to a circular cross-section.

    Generates Soloviev equilibrium from scalar-parameter inputs, provides
    mapping routines for use in equilibrium testing purposes.
    """
    def __init__(self,R,a,B0,Ip,betat,length_unit='m'):
        # instantiate superclass, forcing time splining to false (no time variation
        # in equilibrium)
        super(CircSolovievEFIT,self}).__init__(length_unit=length_unit,tspline=False)

        self._defaultUnits = {}

        self._R = R
        self._defaultUnits['_R'] = 'm'
        self._a = a
        self._defaultUnits['_a'] = 'm'
        self._B0 = B0
        self._defaultUnits['_B'] = 'T'
        self._Ip = Ip
        self._defaultUnits['_Ip'] = 'MA'
        self._betat = betat


    def __str__(self):
        """string formatting for CircSolovievEFIT class.
        """
        datadict = {'R':self._R,'a':self._a,'Ip':self_Ip,'Bt':self_B0,'betat':self_betat}
        return "Circular Soloviev equilibrium with R = %(R)s, a = %(a)s,"+\
        " Ip = %(Ip)f, Bt = %(Bt)f, betat = %(betat)f" % datadict

    def getInfo(self):
        """returns namedtuple of equilibrium information
        """
        data = namedtuple('Info',['R','a','Ip','B0','betat'])
        return data(R=self._R,a=self._a,Ip=self._Ip,B0=self._B0,betat=self_betat)

    def _RZtortheta(self,R,Z):
        """converts input RZ coordinates to polar cross-section
        """
        r = scipy.sqrt((R - self._R)**2 + (Z)**2)
        theta = scipy.arctan(Z/(R - self._R))
        return (r,theta)

    def rz2psi_analytic(self,R,Z,length_unit='m'):
        """analytic formulation for flux calculation in Soloviev equilibrium.

        Args:
            R: Array-like or scalar float.
                Values for radial coordinate to map to poloidal flux.
            Z: Array-like or scalar float.
                Values for radial coordinate to map to poloidal flux.

        Keyword Args:
            length_unit: String or 1.
                Length unit that R and Z are given in.  
                If a string is given, it must be a valid unit specifier:
                
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
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (R and Z given in meters).

        Returns:
            psi: Array or scalar float.  If all of the input arguments are scalar,
                then a scalar is returned. Otherwise, a scipy Array instance is
                returned. If R and Z both have the same shape then psi has this
                shape as well. If the make_grid keyword was True then psi has
                shape (len(Z), len(R)).
        """
        qstar = (4.*scipy.pi*1.e-7) * self._R * self._Ip / (2.*scipy.pi * self._a**2 * self._B0)
        A = 2.* self._B0 * qstar
        C = 8. * self._R * self._B0**2 * self._betat / (self._a**2 * A)

        (r,theta) = self._RZtortheta(R,Z)

        psi = A/4. * (r**2 - self._a**2) + C/8. * (r**2 - self._a**2) * r * scipy.cos(theta)

    def rz2psi(self,R,Z,*args,**kwargs):
        """Converts passed, R,Z arrays to psi values.
        
        Wrapper for Equilibrium.rz2psi masking out timebase dependence.

        Args:
            R: Array-like or scalar float.
                Values of the radial coordinate to
                map to poloidal flux. If the make_grid keyword is True, R must 
                have shape (len_R,).
            Z: Array-like or scalar float.
                Values of the vertical coordinate to
                map to poloidal flux. Must have the same shape as R unless the 
                make_grid keyword is set. If the make_grid keyword is True, Z 
                must have shape (len_Z,).
            *args:
                Slot for time input for consistent syntax with Equilibrium.rz2psi.
                will return dummy value for time if input in EqdskReader.

        Keyword Args:
            make_grid: Boolean.
                Set to True to pass R and Z through meshgrid
                before evaluating. If this is set to True, R and Z must each
                only have a single dimension, but can have different lengths.
                Default is False (do not form meshgrid).
            length_unit: String or 1.
                Length unit that R and Z are being given
                in. If a string is given, it must be a valid unit specifier:
                
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
                
                If length_unit is 1 or None, meters are assumed. The default
                value is 1 (R and Z given in meters).
            **kwargs:
                Other keywords (i.e., return_t) to rz2psi are valid
                (necessary for proper inheritance and usage in other mapping routines)
                but will return dummy values.

        Returns:
            psi: Array or scalar float. If all of the input arguments are scalar,
                then a scalar is returned. Otherwise, a scipy Array instance is
                returned. If R and Z both have the same shape then psi has this
                shape as well. If the make_grid keyword was True then psi has
                shape (len(Z), len(R)).
        """
        t = self.getTimeBase()[0]
        return super(EqdskReader,self).rz2psi(R,Z,t,**kwargs)