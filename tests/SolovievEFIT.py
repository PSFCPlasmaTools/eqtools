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

    def getFluxGrid(self):
        """returns flux grid, [R,Z]
        """
        pass