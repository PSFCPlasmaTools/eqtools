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

equil = sefit.CircSolovievEFIT(0.69, 0.22, 5.4, 1.0, 0.02, npts=20)

# show off our nice equilibrium!
equil.plotFlux(False)

# next we'll generate a testing RZ grid to check flux mappings between
# the analytic and numerical solutions.  We'll fill the entire mapping
# space -- create a test point at the midpoint of each "face" in the
# defined RZ grid in equil, so it's equidistant from each of its
# nearest-neighbor points for the interpolation.

# gets axes for psiRZ in equil
R = equil.getRGrid()
Z = equil.getZGrid()
rGrid,zGrid = scipy.meshgrid(R,Z)

# construct face points as meshgrid
rFace,zFace = scipy.meshgrid((R[:-1] + R[1:]) / 2.0, (Z[:-1] + Z[1:]) / 2.0)
print rFace.shape

# plot illustration
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.plot(rGrid,zGrid,'ob',markersize=5)
ax.plot(rFace,zFace,'or',markersize=5)

# calculate fluxes!
psi_analytic = equil.rz2psi_analytic(rFace,zFace)
psi_numerical = equil.rz2psi(rFace,zFace)

fig2 = plt.figure(figsize=(10,8))
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('$R$ (m)')
ax2.set_ylabel('$Z$ (m)')
ax2.set_title("$(\\psi_{analytic} - \\psi_{numerical})/\\psi_{analytic}$, %i x %i grid" % (equil._npts,equil._npts))
cf = ax2.contourf(rFace,zFace,(psi_analytic-psi_numerical)/psi_analytic,50,zorder=2)
cb = plt.colorbar(cf)
circ = plt.Circle((equil._R,0.0),equil._a,ec='r',fc='none',linewidth=3,zorder=3)
ax2.add_patch(circ)
ax2.plot(equil._R,0.0,'rx',markersize=10,zorder=3)

fig3,(ax3,ax4) = plt.subplots(1,2,sharey=True,figsize=(16,8))
ax3.contourf(rFace,zFace,psi_analytic,50)
ax4.contourf(rFace,zFace,psi_numerical,50)
fig3.subplots_adjust(wspace=0)
ax3.set_xlabel('$R$ (m)')
ax4.set_xlabel('$R$ (m)')
ax3.set_ylabel('$Z$ (m)')
ax3.set_title('analytic calculation')
ax4.set_title('numerical calculation')

fig4 = plt.figure(figsize=(8,8))
ax5 = fig4.add_subplot(111)
ax5.contour(rGrid,zGrid,equil.rz2psi(rGrid,zGrid),50)




