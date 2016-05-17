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
    CircSolvievEFIT: class inheriting :py:class:`Equilibrium` for generation of and
    mapping routines using the Soloviev solution to the Grad-Shafranov equation.
"""

import scipy
from collections import namedtuple
from eqtools.core import Equilibrium, ModuleWarning, inPolygon

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

    Args:
        R (float): major radius of model equilibrium.
        a (float): minor radius of model equilibrium.
        B0 (float): on-axis toroidal field of model equilibrium.
        Ip (float): plasma current of model equilibrium.
        betat (float): volume-averaged toroidal beta of model equilibrium.

    Keyword Args:
        length_unit (String): Sets the base unit used for any quantity whose
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
        npts (int): grid size of model equilibrium reconstruction.
            Default is npts=257 (257x257 RZ grid)
    """
    def __init__(self,R,a,B0,Ip,betat,length_unit='m',npts=257):
        # instantiate superclass, forcing time splining to false (no time variation
        # in equilibrium)
        super(CircSolovievEFIT,self).__init__(length_unit=length_unit, tspline=False, monotonic=False)

        self._defaultUnits = {}

        self._R = R
        self._defaultUnits['_R'] = length_unit
        self._a = a
        self._defaultUnits['_a'] = length_unit
        self._B0 = B0
        self._defaultUnits['_B0'] = 'T'
        self._Ip = Ip
        self._defaultUnits['_Ip'] = 'MA'
        self._betat = betat
        self._npts = npts

        self._currentSign = -1 if Ip > 0 else 1
        # Remember: Ip is in MA.
        self._qstar = (2.*scipy.pi * a**2 * B0) / (4.*scipy.pi*1.e-1 * R * Ip)

        # flux definitions
        self._psiLCFS = scipy.array([0.0])
        self._psi0 = -0.5 * self._B0 * self._a**2 / self._qstar
        self._psi0 = scipy.array([self._psi0])

        # RZ grid
        self._rGrid = scipy.linspace(R-1.25*a,R+1.25*a,self._npts)
        self._defaultUnits['_rGrid'] = length_unit
        self._zGrid = scipy.linspace(-1.25*a,1.25*a,self._npts)
        self._defaultUnits['_zGrid'] = length_unit

        self._psiRZ = self.rz2psi_analytic(self._rGrid,self._zGrid,length_unit=length_unit,make_grid=True)
        self._psiRZ = scipy.reshape(self._psiRZ,(1,npts,npts))

    def __str__(self):
        """string formatting for CircSolovievEFIT class.
        """
        datadict = {'R':self._R,'a':self._a,'Ip':self._Ip,'Bt':self._B0,
                    'betat':self._betat}
        return ("Circular Soloviev equilibrium with R = %(R)s, a = %(a)s,"
        " Ip = %(Ip)f, Bt = %(Bt)f, betat = %(betat)f" % datadict)

    def getInfo(self):
        """returns namedtuple of equilibrium information
        """
        data = namedtuple('Info',['R','a','Ip','B0','betat','npts'])
        return data(R=self._R,a=self._a,Ip=self._Ip,B0=self._B0,betat=self._betat,npts=self._npts)

    ####################
    # mapping routines #
    ####################

    def _RZtortheta(self,R,Z,make_grid=False):
        """converts input RZ coordinates to polar cross-section coordinates.

        Args:
            R (Array-like or scalar float): 
                Values of the radial coordinate.  If `R` and `Z` are both scalar
                values, they are used as the coordinate pair for all of the
                values in `t`. Must have the same shape as `Z` unless the
                `make_grid` keyword is True. If `make_grid` is True, `R` must
                have only one dimension (or be a scalar).
            Z (Array-like or scalar float):
                Values of the vertical coordinate. If `R` and `Z` are both
                scalar values, they are used as the coordinate pair for all of
                the values in `t`. Must have the same shape as `R` unless the
                `make_grid` keyword is True. If `make_grid` is True, `Z` must
                have only one dimension.

        Keyword Args:
            make_grid (Boolean):
                Set to True to pass `R` and `Z` through :py:func:`meshgrid`
                before evaluating. If this is set to True, `R` and `Z` must each
                only have a single dimension, but can have different lengths.
                Default is False (do not form meshgrid).

        Returns:
            Tuple of:

            * **r** - array with same dimension as `R`, `Z` of minor-radius values
            * **theta** - array with same dimension as `R`, `Z` of polar angle values
        """
        (
            R,
            Z,
            t,
            time_idxs,
            unique_idxs,
            single_time,
            single_val,
            oshape
        ) = self._processRZt(
            R,
            Z,
            0.0,
            make_grid=make_grid,
            each_t=True,
            check_space=False
        )
        
        R = scipy.reshape(R, oshape)
        Z = scipy.reshape(Z, oshape)
        
        r = scipy.sqrt((R - self._R)**2 + (Z)**2)
        theta = scipy.arctan2(Z, (R - self._R))
        return (r, theta)

    def rz2psi_analytic(self,R,Z,length_unit='m',make_grid=False):
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
        A = 2.* self._B0 / self._qstar
        C = 8. * self._R * self._B0**2 * self._betat / (self._a**2 * A)

        (r,theta) = self._RZtortheta(R,Z,make_grid=make_grid)

        psi = A/4. * (r**2 - self._a**2) + C/8. * (r**2 - self._a**2) * r * scipy.cos(theta)
        return psi

    def rz2psinorm_analytic(self,R,Z,length_unit='m',make_grid=False):
        """Calculates normalized poloidal flux at given (R,Z)

        Uses the definition:

        .. math::
        
            \texttt{psi\_norm} = \frac{\psi - \psi(0)}{\psi(a) - \psi(0)}

        Args:
            R: Array-like or scalar float.
                Values of radial coordinate to map to psinorm.
            Z: Array-like or scalar float.
                Values of vertical coordinate to map to psinorm.

        Keyword Args:
            length_unit (String or 1): Length unit that `R`, `Z` are given in.
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
                value is 1 (use meters).

        Returns:
            psinorm: Array-like or scalar float.
                normalized poloidal flux.
        """
        psi = self.rz2psi_analytic(R,Z,length_unit=length_unit,make_grid=make_grid)
        psinorm = (psi - self._psi0[0]) / (self._psiLCFS[0] - self._psi0[0])

        return psinorm

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
        t = 0.0
        return super(CircSolovievEFIT,self).rz2psi(R,Z,t,**kwargs)

    def rz2psinorm(self,R,Z,*args,**kwargs):
        """Calculates the normalized poloidal flux at the given (R,Z).
        Wrapper for Equilibrium.rz2psinorm masking out timebase dependence.

        Uses the definition:
        psi_norm = (psi - psi(0)) / (psi(a) - psi(0))

        Args:
            R: Array-like or scalar float.
                Values of the radial coordinate to
                map to normalized poloidal flux. If R and Z are both scalar
                values, they are used as the coordinate pair for all of the
                values in t. Must have the same shape as Z unless the make_grid
                keyword is set. If the make_grid keyword is True, R must have
                shape (len_R,).
            Z: Array-like or scalar float.
                Values of the vertical coordinate to
                map to normalized poloidal flux. If R and Z are both scalar
                values, they are used as the coordinate pair for all of the
                values in t. Must have the same shape as R unless the make_grid
                keyword is set. If the make_grid keyword is True, Z must have
                shape (len_Z,).
            *args:
                Slot for time input for consistent syntax with Equilibrium.rz2psinorm.
                will return dummy value for time if input in EqdskReader.

        Keyword Args:
            sqrt: Boolean.
                Set to True to return the square root of normalized
                flux. Only the square root of positive psi_norm values is taken.
                Negative values are replaced with zeros, consistent with Steve
                Wolfe's IDL implementation efit_rz2rho.pro. Default is False
                (return psinorm).
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
                Other keywords passed to Equilibrium.rz2psinorm are valid,
                but will return dummy values (i.e. for timebase keywords)

        Returns:
            psinorm: Array or scalar float. If all of the input arguments are
                scalar, then a scalar is returned. Otherwise, a scipy Array
                instance is returned. If R and Z both have the same shape then
                psinorm has this shape as well. If the make_grid keyword was
                True then psinorm has shape (len(Z), len(R)).

        Examples:
            All assume that Eq_instance is a valid instance EqdskReader:

            Find single psinorm value at R=0.6m, Z=0.0m::
            
                psi_val = Eq_instance.rz2psinorm(0.6, 0)

            Find psinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m).
            Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                psi_arr = Eq_instance.rz2psinorm([0.6, 0.8], [0, 0])

            Find psinorm values on grid defined by 1D vector of radial positions R
            and 1D vector of vertical positions Z::
            
                psi_mat = Eq_instance.rz2psinorm(R, Z, make_grid=True)
        """
        t = 0.0
        return super(CircSolovievEFIT,self).rz2psinorm(R,Z,t,**kwargs)

    def rho2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rz2phinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rz2volnorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rz2rmid(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rz2roa(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rz2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rmid2roa(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rmid2psinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rmid2phinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rmid2volnorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def rmid2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def roa2rmid(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def roa2psinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def roa2phinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def roa2volnorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def roa2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def psinorm2rmid(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def psinorm2roa(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def psinorm2volnorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def psinorm2phinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def psinorm2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def phinorm2psinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def phinorm2volnorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def phinorm2rmid(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def phinorm2roa(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def phinorm2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def volnorm2psinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def volnorm2phinorm(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def volnorm2rmid(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def volnorm2roa(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def volnorm2rho(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    def getNearestIdx(self,*args,**kwargs):
        raise NotImplementedError("method not defined for Soloviev testing module")

    #################
    # data handlers #
    #################

    def getTimeBase(self):
        return scipy.array([0.0])

    def getCurrentSign(self):
        """Returns the sign of the current, for flux mapping.  Note - inverted from intuitive, due to Soloviev formalism.
        """
        return self._currentSign

    def getFluxGrid(self):
        return self._psiRZ.copy()

    def getRGrid(self,length_unit=1):
        """Returns model equilibrium grid R-axis [r]
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'],length_unit)
        return unit_factor * self._rGrid.copy()

    def getZGrid(self,length_unit=1):
        """Returns model equilibrium grid Z-axis [z]
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zGrid'],length_unit)
        return unit_factor * self._zGrid.copy()

    def getFluxAxis(self):
        """Returns psi on magnetic axis (float)
        """
        return self._psi0

    def getFluxLCFS(self):
        """Returns psi on the LCFS (float)
        """
        return self._psiLCFS

    ####################
    # plot equilibrium #
    ####################

    def plotFlux(self,fill=True):
        """Plots flux contours directly from psi grid.

        Keyword Args:
            fill (Boolean):
                Set True to plot filled contours.  Set False (default) to plot white-background color contours.
        """
        if not _has_plt:
            raise ImportError("Cannot plot without matplotlib package installed.")

        data = {'R':self._R,'eps':self._a/self._R,'Ip':self._Ip,'Bt':self._B0,'beta':self._betat,'npts':self._npts}

        plt.ion()

        psiRZ = self.getFluxGrid()[0,:,:]
        rGrid = self.getRGrid(length_unit='m')
        zGrid = self.getZGrid(length_unit='m')

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$R$ (m)')
        ax.set_ylabel('$Z$ (m)')
        ax.set_title("$R = $%(R).2f, $\\varepsilon = $%(eps).2f, $I_p = $%(Ip).2f, "
                     "$B_T = $%(Bt).2f, $\\beta_t = $%(beta).2f, n = %(npts)i" % data)

        if fill:
            ax.contourf(rGrid,zGrid,psiRZ,50,zorder=2)
        else:
            ax.contour(rGrid,zGrid,psiRZ,50,linestyles='solid',linewidth=2,zorder=2)

        circ = plt.Circle((self._R,0.0),self._a,ec='r',fc='none',linewidth=3,zorder=3)
        ax.add_patch(circ)
        ax.plot(self._R,0.0,'rx',markersize=10,zorder=3)
