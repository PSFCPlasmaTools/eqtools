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

"""testing script using :py:class:`CircSolovievEFIT` object to compare
numerical calculations of flux mapping to analytic forms used in
Soloviev equilibria with circular cross section.
"""
import eqtools
import SolovievEFIT as sefit
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import warnings

warnings.simplefilter("ignore")
plt.ion()

# we will start by generating a Soloviev EFIT with C-Mod-like parameters:
# R = 0.69 m, a = 0.22 m, Ip = 1.0 MA, Bt = 5.4 T.
# we'll use a volume-averaged toroidal beta of 0.2 as well.
# leave default length unit in meters, and use a 257x257 grid (default)

equil = sefit.CircSolovievEFIT(0.69,0.22,5.4,1.0,0.2)

# show off our nice equilibrium!
equil.plotFlux(False)

# next we'll generate a testing RZ grid to check flux mappings between
# the analytic and numerical solutions.  We'll fill the entire mapping
# space -- create a test point at the midpoint of each "face" in the
# defined RZ grid in equil, so it's equidistant from each of its
# nearest-neighbor points for the interpolation.

rGrid = equil.getRGrid()
zGrid = equil.getZGrid()







