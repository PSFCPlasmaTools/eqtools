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
built on the analytic Soloviev equilibrium for testing and synthetic-diagnostic purposes.

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
    def __init__(self):
        pass
        # define/calculate psiRZ

    def __str__(self):
        """string formatting for CircSolovievEFIT class.
        """
        pass

    def getInfo(self):
        """returns namedtuple of equilibrium information
        """
        pass

    def getFluxGrid(self):
        """returns flux grid, [R,Z]
        """
        return self.psiRZ.copy()