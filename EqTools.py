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

#development code for EFIT tree

import scipy
import scipy.interpolate
import scipy.integrate
import re

try:
    import trispline
except ImportError:
    print("WARNING: trispline module could not be loaded -- tricubic spline interpolation will not be available.")
    _has_trispline = False

try:
    import MDSplus
    from MDSplus._treeshr import TreeException
    _has_MDS = True
except Exception as _e_MDS:
    print("WARNING: MDSplus module could not be loaded -- classes that use MDSplus for data access will not work.")
    _has_MDS = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.widgets as mplw
    import matplotlib.gridspec as mplgs
    import time
except Exception:
    print("WARNING: matplotlib modules could not be loaded -- plotting will not be available.")


class AttrDict(dict):
    """
    A dictionary with access via item, attribute, and call notations:
        >>> d = AttrDict()
        >>> d['Variable'] = 123
        >>> d['Variable']
        123
        >>> d.Variable
        123
        >>> d('Variable')
        123
    """
    def __init__(self, init={}):
        dict.__init__(self, init)

    def __getitem__(self, name):
        return super(AttrDict, self).__getitem__(name)

    def __setitem__(self, key, value):
        return super(AttrDict, self).__setitem__(key, value)

    __getattr__ = __getitem__
    __setattr__ = __setitem__
    __call__ = __getitem__


class PropertyAccessMixin(object):
    """
    Mixin to implement access of getter methods through a property-type
    interface without the need to apply a decorator to every property.
    
    For any getter obj.getSomething(), the call obj.Something will work.
    
    This is accomplished by overriding __getattribute__ such that if an
    attribute ATTR does not exist it then attempts to call self.getATTR(). If
    self.getATTR() does not exist, an AttributeError will be raised as usual.
    
    Also overrides __setattr__ such that it will raise an AttributeError when
    attempting to write an attribute ATTR for which there is already a method
    getATTR.
    """
    def __getattribute__(self, name):
        """
        Tries to get attribute as-written. If this fails, tries to call the
        method get[name] with no arguments. If this fails, raises
        AttributeError. This effectively generates a Python 'property' for
        each getter method.
        """
        try:
            return super(Equilibrium, self).__getattribute__(name)
        except AttributeError:
            try:
                return super(Equilibrium, self).__getattribute__('get'+name)()
            except AttributeError:
                raise AttributeError("%(class)s object has no attribute '%(n)s'"
                                     " or method 'get%(n)s'"
                                      % {'class': self.__class__.__name__,
                                         'n': name})

    def __setattr__(self, name, value):
        """
        Raises AttributeError if the object already has a method get[name], as
        creation of such an attribute would interfere with the automatic
        property generation in __getattribute__.
        """
        if hasattr(self, 'get'+name):
            raise AttributeError("%(class)s object already has getter method "
                                 "'get%(n)s', creating attribute '%(n)s' will"
                                 " conflict with automatic property generation."
                                 % {'class': self.__class__.__name__,
                                    'n': name})
        else:
            super(EFITTree, self).__setattr__(name, value)

"""
Dictionary to implement length unit conversions. The first key is the unit
you are converting FROM, the second the unit you are converting TO.
Supports:
    m, cm, mm, in, ft, yd, smoot, cubit, hand
"""
_length_conversion = {'m': {'m': 1.0,
                            'cm': 100.0,
                            'mm': 1000.0,
                            'in': 39.37,
                            'ft': 39.37 / 12.0,
                            'yd': 39.37 / (12.0 * 3.0),
                            'smoot': 39.37 / 67.0,
                            'cubit': 39.37 / 18.0,
                            'hand': 39.37 / 4.0},
                      'cm': {'m': 0.01,
                             'cm': 1.0,
                             'mm': 10.0,
                             'in': 39.37 / 100.0,
                             'ft': 39.37 / (100.0 * 12.0),
                             'yd': 39.37 / (100.0 * 12.0 * 3.0),
                             'smoot': 39.37 / (100.0 * 67.0),
                             'cubit': 39.37 / (100.0 * 18.0),
                             'hand': 39.37 / (100.0 * 4.0)},
                      'mm': {'m': 0.001,
                             'cm': 0.1,
                             'mm': 1.0,
                             'in': 39.37 / 1000.0,
                             'ft': 39.37 / (1000.0 * 12.0),
                             'yd': 39.37 / (1000.0 * 12.0 * 3.0),
                             'smoot': 39.37 / (1000.0 * 67.0),
                             'cubit': 39.37 / (1000.0 * 18.0),
                             'hand': 39.37 / (1000.0 * 4.0)},
                      'in': {'m': 1.0 / 39.37,
                             'cm': 100.0 / 39.37,
                             'mm': 1000.0 / 39.37,
                             'in': 1.0,
                             'ft': 1.0 / 12.0,
                             'yd': 1.0 / (12.0 * 3.0),
                             'smoot': 1.0 / 67.0,
                             'cubit': 1.0 / 18.0,
                             'hand': 1.0 / 4.0},
                      'ft': {'m': 12.0 / 39.37,
                             'cm': 12.0 * 100.0 / 39.37,
                             'mm': 12.0 * 1000.0 / 39.37,
                             'in': 12.0,
                             'ft': 1.0,
                             'yd': 1.0 / 3.0,
                             'smoot': 12.0 / 67.0,
                             'cubit': 12.0 / 18.0,
                             'hand': 12.0 / 4.0},
                      'yd': {'m': 3.0 * 12.0 / 39.37,
                             'cm': 3.0 * 12.0 * 100.0 / 39.37,
                             'mm': 3.0 * 12.0 * 1000.0 / 39.37,
                             'in': 3.0 * 12.0,
                             'ft': 3.0,
                             'yd': 1.0,
                             'smoot': 3.0 * 12.0 / 67.0,
                             'cubit': 3.0 * 12.0 / 18.0,
                             'hand': 3.0 * 12.0 / 4.0},
                      'smoot': {'m': 67.0 / 39.37,
                                'cm': 67.0 * 100.0 / 39.37,
                                'mm': 67.0 * 1000.0 / 39.37,
                                'in': 67.0,
                                'ft': 67.0 / 12.0,
                                'yd': 67.0 / (12.0 * 3.0),
                                'smoot': 1.0,
                                'cubit': 67.0 / 18.0,
                                'hand': 67.0 / 4.0},
                      'cubit': {'m': 18.0 / 39.37,
                                'cm': 18.0 * 100.0 / 39.37,
                                'mm': 18.0 * 1000.0 / 39.37,
                                'in': 18.0,
                                'ft': 18.0 / 12.0,
                                'yd': 18.0 / (12.0 * 3.0),
                                'smoot': 18.0 / 67.0,
                                'cubit': 1.0,
                                'hand': 18.0 / 4.0},
                      'hand': {'m': 4.0 / 39.37,
                               'cm': 4.0 * 100.0 / 39.37,
                               'mm': 4.0 * 1000.0 / 39.37,
                               'in': 4.0,
                               'ft': 4.0 / 12.0,
                               'yd': 4.0 / (12.0 * 3.0),
                               'smoot': 4.0 / 67.0,
                               'cubit': 4.0 / 18.0,
                               'hand': 1.0}}


class Equilibrium(object):
    """
    Abstract class of data handling object for magnetic reconstruction outputs.
    Defines the mapping routines and method fingerprints necessary.
    Each variable or set of variables is recovered with a corresponding
    getter method. Essential data for mapping are pulled on initialization
    (psirz grid, for example) to frontload timing overhead. Additional data
    are pulled at the first request and stored for subsequent usage.

    NOTE: this abstract class should not be used directly. Device- and code-
    specific subclasses are set up to account for inter-device/-code differences in
    data storage.
    """
    def __init__(self, length_unit='m', tspline=False):
        """
        Optional keyword length_unit[='m'] sets the base unit used for any
        quantity whose dimensions are length to any power. Valid options are:
        'm'         meters
        'cm'        centimeters
        'mm'        millimeters
        'in'        inches
        'ft'        feet
        'yd'        yards
        'smoot'     smoots
        'default'   whatever the default in the tree is (no conversion is
                        performed, units may be inconsistent)

        Optional keyword tspline[=False] sets the dimensionality of the
        interpolation to include variation parameters in time. This requires
        at least four complete equilibria at different times.  It is also
        assumed that they are functionally correlated, and that parameters do
        not vary out of their boundarys (derivative = 0 boundary condition)
        """
       # super(EqTree, self).__init__()
        if length_unit != 'default' and not (length_unit in _length_conversion):
            raise ValueError("Unit '%s' not a valid unit specifier!" % length_unit)
        else:
            self._length_unit = length_unit
        
        self._tricubic = bool(tspline) # forces this parameter to be a boolean regardless of input
        if self._tricubic:
            if not _has_trispline:
                raise ValueError("trispline module did NOT load, so argument tspline=True is invalid!")
            else:
                #variables that are purely time dependent require splines rather
                #than indexes for interpolation.
                self._psiOfLCFSSpline = {}
                self._psiOfPsi0Spline = {}
                self._MagRSpline = {}
            
        # These are indexes of splines, and become higher dimensional splines
        # with the setting of the tspline keyword

        self._psiOfRZSpline = {}
        self._phiNormSpline = {}
        self._volNormSpline = {}
        self._RmidSpline = {}
        
    
    def __str__(self):
        return 'This is an abstract class.  Please use machine-specific subclass.'
    
    ####################
    # Mapping routines #
    ####################
    
    def rz2psi(self, R, Z, t, return_t=False, make_grid=False, length_unit=1):
        """Converts the passed R, Z, t arrays to psi values.

        Uses scipy.interpolate.RectBivariateSpline to interpolate in terms of
        R and Z. Finds the nearest time slices to those given: nearest-neighbor
        interpolation in time.

        t can be a single value, in which case it is used as the time value for
        all of the (R, Z) points.

        R and Z can be single values, in which case they are used as the (R, Z)
        points for all of the t points.

        R and Z must have the same shape, unless they are being gridded
        (make_grid keyword, below).

        Set the make_grid keyword to True to first feed R and Z to
        scipy.meshgrid and evaluate over a grid. In this case it will likely be
        most useful to pass a single value for t.

        Set the return_t keyword to True to return a tuple of (psi, time_idxs),
        where time_idxs is the time indicies actually used in evaluating psi
        (nearest-neighbor interpolation).
        
        Set the length_unit keyword to specify what units you have given R and
        Z in. If length_unit is None, whatever the units specified when creating
        the instance are used. If length_unit="default" then it is assumed R
        and Z are in meters.


        Examples:
        All assume that EFITTree_instance is a valid instance of the appropriate
        extension of the EFITTree abstract class.

        Find single psi value at R=0.6m, Z=0.0m, t=0.26s:
        psi_val = EFITTree_instance.rz2psi(0.6, 0, 0.26)

        Find psi values at (R, Z) points (0.6m, 0m) and (0.8m, 0m) at the single
        time t=0.26s. Note that the Z vector must be fully specified, even if
        the values are all the same:
        psi_arr = EFITTree_instance.rz2psi([0.6, 0.8], [0, 0], 0.26)

        Find psi values at (R, Z) points (0.6m, 0m) at times t=[0.2s, 0.3s]:
        psi_arr = EFITTree_instance.rz2psi(0.6, 0, [0.2, 0.3])

        Find psi values at (R, Z, t) points (0.6m, 0m, 0.2s) and (0.5m, 0.2m, 0.3s):
        psi_arr = EFITTree_instance.rz2psi([0.6, 0.5], [0, 0.2], [0.2, 0.3])

        Find psi values on grid defined by 1D vector of radial positions R and
        1D vector of vertical positions Z at time t=0.2s:
        psi_mat = EFITTree_instance.rz2psi(R, Z, 0.2)"""
        
        # Check inputs and process into flat arrays with units of meters:
        (R,
         Z,
         t,
         time_idxs,
         original_shape,
         single_val,
         single_time) = self._processRZt(R, Z, t, make_grid=make_grid, length_unit=length_unit)

        # Optimized form for single t value case:
        if not self._tricubic:
            if single_time:
                out_vals = self._getFluxBiSpline(time_idxs[0]).ev(Z, R)
            else:
                out_vals = scipy.zeros(t.shape)
            # Need to loop over time_idxs
                for k in xrange(0, len(t)):
                    out_vals[k] = self._getFluxBiSpline(time_idxs[k]).ev(Z[k], R[k]) # IS THIS WORKING PROPERLY?
        else:
            out_vals = self._getFluxTriSpline().ev(Z,R,t)
        # Correct for current sign:
        out_vals = -1 * out_vals * self.getCurrentSign()

        # Restore correct shape:
        out_vals = scipy.reshape(out_vals, original_shape)

        # Unwrap back into single value to match input form, if necessary:
        if single_val:
            out = out_vals[0]
        else:
            out = out_vals

        if return_t:
            # will reshape time_idxs only if it is utilized (otherwise set to None)
            if not self._tricubic:
                time_idxs = scipy.reshape(time_idxs, original_shape)

            return (out, time_idxs)
        else:
            return out

    def rz2psinorm(self, R, Z, t, return_t=False, sqrt=False, make_grid=False, length_unit=1):
        """Calculates the normalized poloidal flux
        psi_norm = (psi - psi(0)) / (psi(a) - psi(0))
        at the given (R, Z, t).

        Set the sqrt keyword to True to return the square root of the normalized
        flux instead. Only the square root of positive psi_norm values is taken.
        Negative values are replaced with zeros, consistent with Steve Wolfe's
        IDL implementation efit_rz2rho.pro.

        Set the return_t keyword to True to return a tuple of (psinorm, time_idxs).

        Set the make_grid keyword to True to first feed R and Z to
        scipy.meshgrid to evaluate over a grid.
        
        Set the length_unit keyword to specify what units you have given R and
        Z in. If length_unit is None, whatever the units specified when creating
        the instance are used. If length_unit="default" then it is assumed R
        and Z are in meters.

        Behavior over various shapes of inputs is the same as for rz2psi, refer
        to rz2psi's docstring for more information and example calling patterns."""

        psi, time_idxs = self.rz2psi(R, Z, t, return_t=True, make_grid=make_grid, length_unit=length_unit)

        if not self._tricubic:
            psi_boundary = self.getFluxLCFS()[time_idxs]
            psi_0 = self.getFluxAxis()[time_idxs]

        else:
            # use spline to generate the psi at the core and at boundary.

            psi_boundary = scipy.interpolate.interp1d(self.getFluxLCFS(),self.getTimeBase()).ev(t)
            psi_0 = scipy.interpolate.interp1d(self.getFluxAxis(),self.getTimeBase()).ev(t)

        psi_norm = (psi - psi_0) / (psi_boundary - psi_0)

        if sqrt:
            scipy.place(psi_norm, psi_norm < 0, 0)
            out = scipy.sqrt(psi_norm)
        else:
            out = psi_norm

        # Unwrap single values to ensure least surprise:
        try:
            iter(psi)
        except TypeError:
            out = out[0]
            time_idxs = time_idxs[0]

        if return_t:
            return (out, time_idxs)
        else:
            return out

    def rz2phinorm(self, *args, **kwargs):
        """Calculates the normalized toroidal flux
        phi = integral(q(psi), dpsi)
        phi_norm = phi / phi(a),
        based on the IDL version efit_rz2rho.pro by Steve Wolfe.

        Uses cumulative trapezoidal integration to find the integral of q, then
        uses cubic spline interpolation to get from the uniform psi_norm grid
        to the desired point(s).

        Set the sqrt keyword to True to return the square root of the normalized
        flux instead. Only the square root of positive psi_norm values is taken.
        Negative values are replaced with zeros, consistent with Steve Wolfe's
        IDL implementation efit_rz2rho.pro.

        Set the return_t keyword to True to return a tuple of (phinorm, time_idxs).

        Set the make_grid keyword to True to first feed R and Z to
        scipy.meshgrid to evaluate over a grid.

        Behavior over various shapes of inputs is the same as for rz2psi, refer
        to rz2psi's docstring for more information and example calling patterns."""
        self._RZ2Quan(self._getPhiNormSpline, *args, **kwargs)

    def rz2volnorm(self, *args, **kwargs):
        """Calculates the normalized flux surface volume, based on the IDL
        version efit_rz2rho.pro by Steve Wolfe.

        Uses cubic spline interpolation to get from the uniform psi_norm grid
        to the desired point(s).

        Set the sqrt keyword to True to return the square root of the normalized
        flux instead. Only the square root of positive psi_norm values is taken.
        Negative values are replaced with zeros, consistent with Steve Wolfe's
        IDL implementation efit_rz2rho.pro.

        Set the return_t keyword to True to return a tuple of (volnorm, time_idxs).

        Set the make_grid keyword to True to first feed R and Z to
        scipy.meshgrid to evaluate over a grid.

        Behavior over various shapes of inputs is the same as for rz2psi, refer
        to rz2psi's docstring for more information and example calling patterns."""

        return self._RZ2Quan(self._getVolNormSpline, *args, **kwargs)

    def rz2rho(self, method, *args, **kwargs):
        """Convert the passed (R, Z, t) coordinates into one of several
        normalized coordinates. The following values for the method keyword are
        supported:
        psinorm     Normalized poloidal flux. See documentation for rz2psinorm.
        phinorm     Normalized toroidal flux. See documentation for rz2phinorm.
        volnorm     Normalized volume. See documentation for rz2volnorm."""

        if method == 'psinorm':
            return self.rz2psinorm(*args, **kwargs)
        elif method == 'phinorm':
            return self.rz2phinorm(*args, **kwargs)
        elif method == 'volnorm':
            return self.rz2volnorm(*args, **kwargs)
        else:
            raise ValueError("rz2rho: Unsupported normalized coordinate method '%s'!" % method)

    def rz2rmid(self, *args, **kwargs):
        """Maps the given points to the outboard midplane major radius, R_mid,
        based on the IDL version efit_rz2rmid.pro by Steve Wolfe.
        
        Uses cubic spline interpolation to get from the uniform psi_norm grid
        to the desired points.
        
        Set the return_t keyword to True to return a tuple of (Rmid, time_idxs).
        
        Set the make_grid keyword to True to first feed R and Z to
        scipy.meshgrid to evaluate over a grid.
        
        The length_unit specifies the units for BOTH the R, Z input and the
        R_mid output.
        
        Behavior over various shapes of inputs is the same as for rz2psi, refer
        to rz2psi's docstring for more information and example calling patterns."""
        
        # Steve Wolfe's version has an extra (linear) interpolation step for
        # small psi_norm. Should check to see if we need this still with the
        # scipy spline. So far looks fine...
        
        # Convert units from meters to desired target:
        try:
            length_unit = kwargs['length_unit']
        except KeyError:
            length_unit = 1
        
        unit_factor = self._getLengthConversionFactor('m', length_unit)
        
        return unit_factor * self._RZ2Quan(self._getRmidSpline, *args, **kwargs)

    def psinorm2rmid(self, psi_norm, t, return_t=False, rho=False, kind='cubic', length_unit=1):
        """Calculates the outboard R_mid location corresponding to the passed
        psi_norm (normalized poloidal flux) values.

        Set the return_t keyword to True to return a tuple of (Rmid, time_idxs).

        Set the rho keyword to true to return r/a instead of R_mid."""

        (psi_norm_proc,
         dum,
         t_proc,
         time_idxs,
         original_shape,
         single_val,
         single_time) = self._processRZt(psi_norm, psi_norm, t, make_grid=False, check_space=False)

        # Handling for single-value case:
        if single_val:
            psi_norm_proc = psi_norm
        
        # Convert units from meters to desired target:
        unit_factor = self._getLengthConversionFactor('m', length_unit)
        
        return unit_factor * self._psinorm2Quan(self._getRmidSpline,
                                                psi_norm_proc,
                                                time_idxs,
                                                psi_norm,
                                                t,
                                                return_t=return_t,
                                                rho=rho,
                                                kind=kind)

    def psinorm2volnorm(self, psi_norm, t, return_t=False, kind='cubic'):
        """Calculates the normalized volume corresponding to the passed
        psi_norm (normalized poloidal flux) values.

        Set the return_t keyword to True to return a tuple of (Rmid, time_idxs)."""

        (psi_norm_proc,
         dum,
         t_proc,
         time_idxs,
         original_shape,
         single_val,
         single_time) = self._processRZt(psi_norm, psi_norm, t, make_grid=False, check_space=False)

        # Handling for single-value case:
        if single_val:
            psi_norm_proc = psi_norm

        return self._psinorm2Quan(self._getVolNormSpline, psi_norm_proc, time_idxs, psi_norm, t, return_t=return_t, kind=kind)

    def psinorm2phinorm(self, psi_norm, t, return_t=False, kind='cubic'):
        """Calculates the normalized toroidal flux corresponding to the passed
        psi_norm (normalized poloidal flux) values.

        Set the return_t keyword to True to return a tuple of (Rmid, time_idxs)."""

        (psi_norm_proc,
         dum,
         t_proc,
         time_idxs,
         original_shape,
         single_val,
         single_time) = self._processRZt(psi_norm, psi_norm, t, make_grid=False, check_space=False)

        # Handling for single-value case:
        if single_val:
            psi_norm_proc = psi_norm

            
        return self._psinorm2Quan(self._getPhiNormSpline, psi_norm_proc, time_idxs, psi_norm, t, return_t=return_t, kind=kind)

    ###########################
    # Backend Mapping Drivers #
    ###########################

    def _psinorm2Quan(self, spline_func, psi_norm, time_idxs, x, t, return_t=False, sqrt=False, rho=False, kind='cubic'):
        """Utility function for computing a variety of quantities given psi_norm
        and the relevant time indices.

        Arguments:
        spline_func     Function which returns a 1d spline given a time index.
        psi_norm        Array of psi_norm values to evaluate at.
        time_idxs       Time indices array for each of the psi_norm values.
        x, t            Spatial and temporal arrays. (Needed for determining
                            output shape.)
        return_t=False  Boolean keyword to cause the function to return a tuple
                            of (Quan, time_idxs).
        sqrt=False      Boolean keyword to cause the square root of Quan to be
                            returned. Values with a negative Quan are set to
                            zero in this case.
        rho=False       Boolean keyword to cause the output of a routine that
                            returns R_mid to be converted to r/a."""

        # Handle single value case properly:
        try:
            iter(x)
        except TypeError:
            single_value = True
            psi_norm = scipy.array([psi_norm])
            time_idxs = scipy.array([time_idxs])
        else:
            single_value = False

        # Check for single time for speedy evaluation of these cases:
        try:
            iter(t)
        except TypeError:
            single_time = True
        else:
            single_time = False

        original_shape = psi_norm.shape
        psi_norm = scipy.reshape(psi_norm, -1)
        time_idxs = scipy.reshape(time_idxs, -1)

        if not self._tricubic:
            if single_time:
                quan_norm = spline_func(time_idxs[0], kind=kind)(psi_norm)
                if rho:
                    quan_norm = ((quan_norm - self.getMagR(length_unit='m')[time_idxs[0]])
                                 / (self.getRmidOut(length_unit='m')[time_idxs[0]]
                                    - self.getMagR(length_unit='m')[time_idxs[0]]))
            else:
                quan_norm = scipy.zeros(psi_norm.shape)
                for k in xrange(0, len(quan_norm)):
                    quan_norm[k] = spline_func(time_idxs[k], kind=kind)(psi_norm[k])
                    if rho:
                        quan_norm[k] = ((quan_norm[k] - self.getMagR(length_unit='m')[time_idxs[k]])
                                        / (self.getRmidOut(length_unit='m')[time_idxs[k]]
                                           - self.getMagR(length_unit='m')[time_idxs[k]]))
        else:
            quan_norm = spline_func(time_idxs)(psi_norm)
            if rho:
                #quan_norm((quan_norm - 
                print('no')

        # Restore original shape:
        quan_norm = scipy.reshape(quan_norm, original_shape)
        if sqrt:
            scipy.place(quan_norm, quan_norm < 0, 0)
            out = scipy.sqrt(quan_norm)
        else:
            out = quan_norm

        if single_value:
            out = out[0]
            time_idxs = time_idxs[0]

        if return_t:
            return (out, time_idxs)
        else:
            return out

    def _RZ2Quan(self, spline_func, R, Z, t, return_t=False, sqrt=False, make_grid=False, rho=False, kind='cubic', length_unit=1):
        """Utility function for computing a variety of things that are
        interpolated from something measured on a uniform normalized flux grid,
        in particular phi_norm, vol_norm and R_mid.

        Has the same fingerprint as the other mapping functions, with the
        addition of the first required parameter spline_func, which is the
        function that returns a 1d spline given a time index."""

        psi_norm, time_idxs = self.rz2psinorm(R, Z, t, sqrt=sqrt, return_t=True, make_grid=make_grid, length_unit=length_unit)

        return self._psinorm2Quan(spline_func, psi_norm, time_idxs, R, t, return_t=return_t, sqrt=sqrt, rho=rho, kind=kind)

    ####################
    # Helper Functions #
    ####################

    def _getLengthConversionFactor(self, start, end, default=None):
        """Gets the conversion factor to convert from units start to units end.
        
        If start is None, the starting unit is taken to be the unit specified
        when the instance was created.
        
        If start is an int, the starting unit is taken to be the unit specified
        when the instance was created raised to that power.
        
        If start is 'default', either explicitly or because of reverting to the
        instance-level unit, then the value passed in the keyword default is
        used. In this case, default must be a complete unit string (i.e., not
        None, not an int and not 'default').
        
        If default is None, an int or 'default', then the value given for start
        is used. (A circular definition is prevented for cases in which start
        is default by checking for this case during the handling of the case
        start=='default'.)
        
        If end is None, the target (ending) unit is taken to be the unit
        specified when the instance was created.
        
        If end is an int, the target unit is taken to be the unit specified
        when the instance was created raised to that power.
        
        If end is 'default', either explicitly or because of reverting to the
        instance-level unit, then the value passed in the keyword default is
        used.
        
        If end does not specify an exponent, it uses whatever the exponent on
        start is. This allows a user to ask for an area in units of m^2 by
        specifying length_unit='m', for instance. An error will still be
        raised if the user puts in a completely inconsistent specification
        such as length_unit='m^3' or length_unit='m^1'.
        
        Uses a regex to parse units of the form:
        'm'
        'm^2'
        'm2'
        Leading and trailing spaces are NOT allowed."""
        # Input handling:
        # Starting unit:
        if start is None:
            # If start is None, it means to use the instance's default unit (implied to the power of 1):
            start = self._length_unit
        elif isinstance(start, (int, long)):
            # If start is an integer type, this is used as the power applied to the instance's default unit:
            if self._length_unit != 'default':
                start = self._length_unit + '^' + str(start)
            else:
                # If the instance's default unit is 'default', this is handled next:
                start = self._length_unit
        if start == 'default':
            # If start is 'default', the thing passed to default is used, but only if it is a complete unit specification:
            if default is None or isinstance(default, (int, long)) or default == 'default':
                raise ValueError("You must specify a complete unit (i.e., "
                                 "non-None, non-integer and not 'default') "
                                 "when using 'default' for the starting unit.")
            else:
                start = default
        
        # Default unit:
        if default is None or isinstance(default, (int, long)) or default == 'default':
            # If start is 'default', these cases have already been caught above.
            default = start
        
        # Target (ending) unit:
        if end is None:
            # If end is None, it means to use the instance's default unit (implied to the power of 1):
            end = self._length_unit
        elif isinstance(end, (int, long)):
            # If end is an integer type, this is used as the power applied to the instance's default unit:
            if self._length_unit != 'default':
                end = self._length_unit + '^' + str(end)
            else:
                # If the instance's default unit is 'default', this is handled next:
                end = self._length_unit
        if end == 'default':
            # If end is 'default', the thing passed to default is used, which
            # defaults to start, which itself is not allowed to be 'default':
            end = default
        
        unit_regex = r'^([A-Za-z]+)\^?([0-9]*)$'
        
        # Need to explicitly cast because MDSplus returns its own classes and
        # re.split doesn't seem to handle the polymorphism properly:
        start = str(start)
        end = str(end)
        default = str(default)
        
        dum1, start_u, start_pow, dum2 = re.split(unit_regex, start)
        dum1, end_u, end_pow, dum2 = re.split(unit_regex, end)
        dum1, default_u, default_pow, dum2 = re.split(unit_regex, default)
        
        start_pow = 1.0 if start_pow == '' else float(start_pow)
        if end_pow == '':
            end_pow = start_pow
        else:
            end_pow = float(end_pow)
        default_pow = 1.0 if default_pow == '' else float(default_pow)
        
        if start_pow != end_pow or start_pow != default_pow:
            raise ValueError("Incompatible exponents between '%s', '%s' and '%s'!" % (start, end, default))
        try:
            return (_length_conversion[start_u][end_u])**start_pow
        except KeyError:
            raise ValueError("Unit '%s' is not a recognized length unit!" % end)

    def _processRZt(self, R, Z, t, make_grid=False, check_space=True, length_unit=1):
        """Input checker/processor. Takes R, Z and t. Appropriately packages
        into scipy arrays. Checks the validity of the R, Z ranges. If there is
        a single time value but multiple R, Z values, creates matching time
        vector. If there is a single R value but multiple t values, creates
        matching R and Z vectors. Finds list of nearest-neighbor time indices.
        
        Assumes R and Z are in meters!

        The make_grid keyword causes R and Z to be expanded with scipy.meshgrid.
        
        The check space keyword causes R and Z to be converted to meters then
        checked against the valid spatial grid.

        Returns a tuple in the following order:
        R               Flattened R array with out-of-range values replaced
                            with NaN.
        Z               Flattened Z array with out-of-range values replaced
                            with NaN.
        t               Flattened t array with out-of-range values replaced
                            with NaN.
        time_idxs       Flattened array of nearest-neighbor time indices.
        original_shape  Original shape tuple, used to return the arrays to
                            their starting form.
        single_val      Boolean indicating whether a single point is used. If
                            True, then the final step of the calling code
                            should unpack the result from the array.
        single_time     Boolean indicating whether a single time value is used.
                            If True, then certain simplifying steps can be made."""

        # Handle single-value form of R and Z:
        try:
            iter(R)
        except TypeError:
            single_val = True
            R = scipy.array([R], dtype=float)
            Z = scipy.array([Z], dtype=float)
        else:
            single_val = False
            # Cast into scipy.array so we can handle list inputs:
            R = scipy.array(R, dtype=float)
            Z = scipy.array(Z, dtype=float)

        # Make the grid, if called for:
        if make_grid:
            if R.ndim != 1 and Z.ndim != 1:
                raise ValueError('_processRZt: When using the make_grid keyword, the '
                                 'number of dimensions of R and Z must both be one!')
            else:
                R, Z = scipy.meshgrid(R, Z)

        if R.shape != Z.shape:
            raise ValueError('_processRZt: Shape of R and Z arrays must match!')

        # Check that R, Z points are fine:
        if check_space:
            # Convert units to meters:
            unit_factor = self._getLengthConversionFactor(length_unit, 'm', default='m')
            R = unit_factor * R
            Z = unit_factor * Z
            
            good_points, num_good = self._checkRZ(R, Z)

            if num_good < 1:
                raise ValueError('_processRZt: No valid points!')

            if not single_val:
                # Mask out the bad points here so we don't interfere with the
                # single-value case (which must be valid to have made it past the
                # test above):
                scipy.place(R, ~good_points, scipy.nan)
                scipy.place(Z, ~good_points, scipy.nan)

        # Handle single-value time cases:
        try:
            iter(t)
        except TypeError:
            single_time = True
            t = scipy.array([t])
        else:
            single_time = False
            t = scipy.array(t)
            # Handle case where there is a single R/Z but multiple t:
            if single_val:
                single_val = False
                R = scipy.ones(t.shape) * R
                Z = scipy.ones(t.shape) * Z

        if t.size > 1 and t.shape != R.shape:
            if make_grid:
                raise ValueError('_processRZt: shape of t does not match shape of R '
                                 'and Z. Recall that use of the make_grid '
                                 'keyword requires that t either be a single '
                                 'value, or that its shape matches that of '
                                 'scipy.meshgrid(R, Z).')
            else:
                raise ValueError('_processRZt: t must either be a single number, '
                                 'or must match the shape of R and Z!')

        # Handle non-vector array inputs: store the shape, then flatten the arrays.
        # Don't bother with a test/flag -- just use the shape vector at the end.
        original_shape = R.shape
        R = scipy.reshape(R, -1)
        Z = scipy.reshape(Z, -1)
        t = scipy.reshape(t, -1)

        # takes keyword to bypass for tricubic interpolation
        if not self._tricubic:

            # Set up times to use -- essentially use nearest-neighbor interpolation
            time_idxs = self._getNearestIdx(t, self.getTimeBase())
            # Check errors and warn if needed:
            t_errs = scipy.absolute(t - self.getTimeBase()[time_idxs])
            if (t_errs > scipy.mean(scipy.diff(self.getTimeBase())) / 3.0).any():
                print("Warning: _processRZt: Some time points are off by more than 1/3 "
                      "the EFIT point spacing. Using nearest-neighbor interpolation "
                      "between time points. You may want to run EFIT on the timebase "
                      "you need. Max error: %.3fs" % max(t_errs))

                # If a single time value is passed with multiple R, Z points, evaluate
                # them all at that time point:
            if single_time and not single_val:
                t = scipy.ones(R.shape) * t[0]
                time_idxs = scipy.ones(R.shape, dtype=int) * time_idxs[0]
        else:
            time_idxs = SP.array([None])

        return (R, Z, t, time_idxs, original_shape, single_val, single_time)

    def _checkRZ(self, R, Z):
        """Checks whether or not the passed arrays of (R, Z) are within the bounds
        of the reconstruction data. Returns the mask array of booleans indicating
        the goodness of each point at the corresponding index. Raises warnings if
        there are no good_points and if there are some values out of bounds.
        
        Assumes R and Z are in units of meters."""
        good_points = ((R <= self.getRGrid(length_unit='m')[-1]) &
                       (R >= self.getRGrid(length_unit='m')[0]) &
                       (Z <= self.getZGrid(length_unit='m')[-1]) &
                       (Z >= self.getZGrid(length_unit='m')[0]))
        # Gracefully handle single-value versus array inputs, returning in the
        # corresponding type.
        num_good = scipy.sum(good_points)
        test = scipy.array(R)
        if len(test.shape) > 0:
            num_pts = test.size
        else:
            num_good = good_points
            num_pts = 1
        if num_good == 0:
            print("Warning: _checkRZ: No valid (R, Z) points!")
        elif num_good != num_pts:
            print("Warning: _checkRZ: Some (R, Z) values out of bounds. "
                  "(%(bad)d bad out of %(tot)d)"
                    % {'bad': num_pts - num_good,
                       'tot': num_pts})

        return (good_points, num_good)

    def _getNearestIdx(self, v, a):
        """Returns the array of indices of the nearest value in a corresponding to
        each value in v."""
        # Gracefully handle single-value versus array inputs, returning in the
        # corresponding type.
        try:
            return scipy.array([(scipy.absolute(a - val)).argmin() for val in v])
        except TypeError:
            return (scipy.absolute(a - v)).argmin()

    def _getFluxBiSpline(self, idx):
        """Gets the spline corresponding to the given time index, generating
        as needed."""
        try:
            return self._psiOfRZSpline[idx]
        except KeyError:
            self._psiOfRZSpline[idx] = scipy.interpolate.RectBivariateSpline(self.getZGrid(length_unit='m'),
                                                                             self.getRGrid(length_unit='m'),
                                                                             self.getFluxGrid()[idx, :, :])
            return self._psiOfRZSpline[idx]

    def _getFluxTriSpline(self):
        """Gets the tricubic interpolating spline for the flux"""
        try:
            return self._psiOfRZSpline
        except KeyError:
            self_psiOfRZSpline = trispline.spline(self.getTimeBase(),
                                                  self.getRGrid(length_unit='m'),
                                                  self.getZGrid(length_unit='m'),
                                                  self.getFluxGrid())
            return self._psiOfRZSpline

    def _getPhiNormSpline(self, idx, kind='cubic'):
        """Returns the 1d cubic spline object corresponding to the passed time
        index idx, generating it if it does not already exist."""
        try:
            return self._phiNormSpline[idx][kind]
        except KeyError:
            phi_norm_meas = scipy.insert(scipy.integrate.cumtrapz(self.getQProfile()[:, idx]), 0, 0)
            phi_norm_meas = phi_norm_meas / phi_norm_meas[-1]

            spline = scipy.interpolate.interp1d(scipy.linspace(0, 1, len(phi_norm_meas)),
                                                phi_norm_meas,
                                                kind=kind,
                                                bounds_error=False)
            try:
                self._phiNormSpline[idx][kind] = spline
            except KeyError:
                self._phiNormSpline[idx] = {kind: spline}
            return self._phiNormSpline[idx][kind]

    def _getVolNormSpline(self, idx, kind='cubic'):
        """Returns the 1d cubic spline object corresponding to the passed time
        index idx, generating it if it does not already exist."""
        try:
            return self._volNormSpline[idx][kind]
        except KeyError:
            vol_norm_meas = self.getFluxVol()[:, idx]
            vol_norm_meas = vol_norm_meas / vol_norm_meas[-1]

            spline = scipy.interpolate.interp1d(scipy.linspace(0, 1, len(vol_norm_meas)),
                                                vol_norm_meas,
                                                kind=kind,
                                                bounds_error=False)
            try:
                self._volNormSpline[idx][kind] = spline
            except KeyError:
                self._volNormSpline[idx] = {kind: spline}
            return self._volNormSpline[idx][kind]

    def _getRmidSpline(self, idx, kind='cubic'):
        """Returns the 1d cubic spline object corresponding to the passed time
        index idx, generating it if it does not already exist.

        There are two approaches that come to mind:
            -- In Steve Wolfe's implementation of efit_rz2mid and efit_psi2rmid,
                he uses the EFIT output Rmid as a function of normalized flux
                (i.e., what is returned by self.getRmidPsi()) in the core, then
                expands the grid beyond this manually.
            -- A simpler approach would be to just compute the psi_norm(R_mid)
                grid directly from the radial grid.

        The latter approach is selected for simplicity.
        
        The units of R_mid are always meters, and are converted by the wrapper
        functions to whatever the user wants."""
        try:
            return self._RmidSpline[idx][kind]
        except KeyError:
            # New approach: create a fairly dense radial grid from the global
            # flux grid to avoid 1d interpolation problems in the core. The
            # bivariate spline seems to be a little more robust in this respect.
            resample_factor = 3
            R_grid = scipy.linspace(self.getMagR(length_unit='m')[idx],
                                    self.getRGrid(length_unit='m')[-1],
                                    resample_factor * len(self.getRGrid(length_unit='m')))

            psi_norm_on_grid = self.rz2psinorm(R_grid,
                                               self.getMagZ(length_unit='m')[idx] * scipy.ones(R_grid.shape),
                                               self.getTimeBase()[idx])

            spline = scipy.interpolate.interp1d(psi_norm_on_grid,
                                                R_grid,
                                                kind=kind,
                                                bounds_error=False)
            try:
                self._RmidSpline[idx][kind] = spline
            except KeyError:
                self._RmidSpline[idx] = {kind: spline}
            return self._RmidSpline[idx][kind]

    def getTreeInfo(self):
        #returns AttrDict of instance parameters (shot, EFIT tree, size and timebase info)
        raise NotImplementedError()

    def getTimeBase(self):
        #returns EFIT time base array (t)
        raise NotImplementedError()

    def getFluxGrid(self):
        #returns 3D grid of psi(r,z,t)
        raise NotImplementedError()

    def getRGrid(self):
        #returns vector of R-values for psirz grid (r)
        raise NotImplementedError()

    def getZGrid(self):
        #returns vector of Z-values for psirz grid (z)
        raise NotImplementedError()

    def getFluxAxis(self):
        #returns psi at magnetic axis as simagx(t)
        raise NotImplementedError()

    def getFluxLCFS(self):
        #returns psi at separatrix, sibdry(t)
        raise NotImplementedError()

    def getRLCFS(self):
        #returns R-positions mapping LCFS, rbbbs(t,n)
        raise NotImplementedError()

    def getZLCFS(self):
        #returns Z-positions mapping LCFS, zbbbs(t,n)
        raise NotImplementedError()

    def getFluxVol(self):
        #returns volume contained within a flux surface as function of psi, volp(psi,t)
        raise NotImplementedError()

    def getVolLCFS(self):
        #returns plasma volume in LCFS, vout(t)
        raise NotImplementedError()

    def getRmidPsi(self):
        #returns max major radius of flux surface, rpres(t,psi)
        raise NotImplementedError()

    def getFluxPres(self):
        #returns EFIT-calculated pressure p(psi,t)
        raise NotImplementedError()

    def getElongation(self):
        #returns LCFS elongation, kappa(t)
        raise NotImplementedError()

    def getUpperTriangularity(self):
        #returns LCFS upper triangularity, delta_u(t)
        raise NotImplementedError()

    def getLowerTriangularity(self):
        #returns LCFS lower triangularity, delta_l(t)
        raise NotImplementedError()

    def getShaping(self):
        #returns dimensionless shaping parameters for plasma
        #AttrDict containing {LCFS elongation, LCFS upper/lower triangularity)
        raise NotImplementedError()

    def getMagR(self):
        #returns magnetic-axis major radius, rmagx(t)
        raise NotImplementedError()

    def getMagZ(self):
        #returns magnetic-axis Z, zmagx(t)
        raise NotImplementedError()

    def getAreaLCFS(self):
        #returns LCFS surface area, areao(t)
        raise NotImplementedError()

    def getAOut(self):
        #returns outboard-midplane minor radius
        raise NotImplementedError()

    def getRmidOut(self):
        #returns outboard-midplane major radius
        raise NotImplementedError()

    def getGeometry(self):
        #returns dimensional geometry parameters for plasma
        #AttrDict containing {mag axis r,z, LCFS area, volume, outboard midplane major radius}
        raise NotImplementedError()

    def getQProfile(self):
        #returns safety factor profile q(psi,t):
        raise NotImplementedError()

    def getQ0(self):
        #returns q-value on magnetic axis, q0(t)
        raise NotImplementedError()

    def getQ95(self):
        #returns q at 95% flux, psib(t)
        raise NotImplementedError()

    def getQLCFS(self):
        #returns q on LCFS, qout(t)
        raise NotImplementedError()

    def getQ1Surf(self):
        #returns outboard-midplane minor radius of q=1 surface, aaq1(t)
        raise NotImplementedError()
    
    def getQ2Surf(self):
        #returns outboard-midplane minor radius of q=2 surface, aaq2(t)
        raise NotImplementedError()

    def getQ3Surf(self):
        #returns outboard-midplane minor radius of q=3 surface, aaq3(t)
        raise NotImplementedError()

    def getQs(self):
        #returns specific q-profile values
        #AttrDict containing {q0, q95, q(LCFS), minor radius of q=1,2,3 surfaces}
        raise NotImplementedError()

    def getBtVac(self):
        #returns vacuum on-axis toroidal field btaxv(t)
        raise NotImplementedError()

    def getBtPla(self):
        #returns plasma on-axis toroidal field btaxp(t)
        raise NotImplementedError()

    def getBpAvg(self):
        #returns avg poloidal field, bpolav(t)
        raise NotImplementedError() 

    def getFields(self):
        #returns magnetic-field measurements from EFIT
        #dict containing {Btor on magnetic axis (plasma and vacuum), avg Bpol)
        raise NotImplementedError()

    def getIpCalc(self):
        #returns EFIT-calculated plasma current
        raise NotImplementedError()

    def getIpMeas(self):
        #returns measured plasma current
        raise NotImplementedError()

    def getJp(self):
        #returns (r,z,t) grid of EFIT-calculated current density
        raise NotImplementedError()

    def getBetaT(self):
        #returns calculated toroidal beta, betat(t)
        raise NotImplementedError()

    def getBetaP(self):
        #returns calculated avg poloidal beta, betap(t)
        raise NotImplementedError()

    def getLi(self):
        #returns calculated internal inductance of plasma, ali(t)
        raise NotImplementedError()

    def getBetas(self):
        #returns calculated beta and inductive values
        #AttrDict of {betat,betap,li}
        raise NotImplementedError()

    def getDiamagFlux(self):
        #returns diamagnetic flux, diamag(t)
        raise NotImplementedError()

    def getDiamagBetaT(self):
        #returns diamagnetic-loop toroidal beta, betatd(t)
        raise NotImplementedError()

    def getDiamagBetaP(self):
        #returns diamagnetic-loop poloidal beta, betapd(t)
        raise NotImplementedError()

    def getDiamagTauE(self):
        #returns diamagnetic-loop energy confinement time, taudia(t)
        raise NotImplementedError()

    def getDiamagWp(self):
        #returns diamagnetic-loop plasma stored energy, wplasmd(t)
        raise NotImplementedError()

    def getDiamag(self):
        #returns diamagnetic measurements of plasma parameters
        #AttrDict of {diamag flux, betat,betap from diamag coils, tau_E from diamag, diamag stored energy)
        raise NotImplementedError()

    def getWMHD(self):
        #returns EFIT-calculated MHD stored energy wplasm(t)
        raise NotImplementedError()

    def getTauMHD(self):
        #returns EFIT-calculated MHD energy confinement time taumhd(s)
        raise NotImplementedError()

    def getPinj(self):
        #returns EFIT-calculated injected power, pbinj(t)
        raise NotImplementedError()

    def getWbdot(self):
        #returns EFIT-calculated d/dt of magnetic stored energy, wbdot(t)
        raise NotImplementedError()

    def getWpdot(self):
        #returns EFIT-calculated d/dt of plasma stored energy, wpdot(t)
        raise NotImplementedError()

    def getEnergy(self):
        #returns stored-energy parameters
        #dict of {stored energy, MHD tau_E, injected power, d/dt of magnetic, plasma stored energy)
        raise NotImplementedError()

    def getParam(self,path):
        #backup function - takes parameter name for EFIT variable, returns that variable
        #acts as wrapper for EFIT tree access from within object
        raise NotImplementedError()

    def getMachineCrossSection(self):
        raise NotImplementedError("function to return machine cross-section not implemented for this class yet!")

    def plotFlux(self):
        """
        streamlined plotting of flux contours directly from psi grid.
        """
        
        plt.ion()
        
        try:
            psiRZ = self.getFluxGrid()
            rGrid = self.getRGrid(length_unit='m')
            zGrid = self.getZGrid(length_unit='m')
            t = self.getTimeBase()

            RLCFS = self.getRLCFS(length_unit='m')
            ZLCFS = self.getZLCFS(length_unit='m')
            try:
                x, y = self.getMachineCrossSection()
            except NotImplementedError:
                print('No machine cross-section implemented!')
                x = []
                y = []
        except ValueError:
            raise AttributeError('cannot plot EFIT flux map.')

        #event handler for arrow key events in plot windows.  Pass slider object
        #to update as masked argument using lambda function
        #lambda evt: arrow_respond(my_slider,evt)
        def arrowRespond(slider,event):
            if event.key == 'right':
                slider.set_val(min(slider.val+1, slider.valmax))
            if event.key == 'left':
                slider.set_val(max(slider.val-1, slider.valmin))

        #make time-slice window
        fluxPlot = plt.figure(figsize=(6,11))
        gs = mplgs.GridSpec(2,1,height_ratios=[30,1])
        psi = fluxPlot.add_subplot(gs[0,0])
        psi.set_aspect('equal')
        timeSliderSub = fluxPlot.add_subplot(gs[1,0])
        title = fluxPlot.suptitle('')

        def updateTime(val):
            start = time.time()

            psi.clear()
            t_idx = int(timeSlider.val)

            title.set_text('EFIT Reconstruction, $t = %(t).2f$ s' % {'t':t[t_idx]})
            psi.set_xlabel('$R$ [m]')
            psi.set_ylabel('$Z$ [m]')
            machine = psi.plot(x,y,'k')
            mask = scipy.where(RLCFS[:,t_idx] > 0.0)
            RLCFSframe = RLCFS[mask[0],t_idx]
            ZLCFSframe = ZLCFS[mask[0],t_idx]
            LCFS = psi.plot(RLCFSframe,ZLCFSframe,'r',linewidth=3)
            fillcont = psi.contourf(rGrid,zGrid,psiRZ[t_idx],50)
            cont = psi.contour(rGrid,zGrid,psiRZ[t_idx],50,colors='k',linestyles='solid')
            fluxPlot.canvas.draw()

            end = time.time()
            print 'elapsed time: '+str(end-start)

        timeSlider = mplw.Slider(timeSliderSub,'t index',0,len(t)-1,valinit=0,valfmt="%d")
        timeSlider.on_changed(updateTime)
        updateTime(0)

        fluxPlot.canvas.mpl_connect('key_press_event', lambda evt: arrowRespond(timeSlider, evt))



class EFITTree(Equilibrium):
    """
    Inherits Equilibrium class. EFIT-specific data handling class for machines using
    standard EFIT tag names/tree structure with MDSplus. Constructor and/or data loading may
    need overriding in a machine-specific implementation.
    Pulls EFIT data from selected MDS tree and shot, stores as object attributes.
    Each EFIT variable or set of variables is recovered with a corresponding getter method.
    Essential data for EFIT mapping are pulled on initialization (e.g. psirz grid).
    Additional data are pulled at the first request and stored for subsequent usage.
    """
    def __init__(self, shot, tree, root, length_unit='m', tspline=False):
        """
        Intializes EFITTree object. Pulls data from MDS tree for storage in
        instance attributes. Core attributes are populated from the MDS tree
        on initialization. Additional attributes are initialized as None,
        filled on the first request to the object.

        INPUTS:
        shot:   shot index
        tree:   MDSplus tree to open to fetch EFIT data.
        root:   Root path for EFIT data in MDSplus tree.
        """

        if not _has_MDS:
            print("ERROR: MDSplus module did not load properly. Exception is below:")
            raise _e_MDS

        super(EFITTree, self).__init__(length_unit=length_unit, tspline=False)
        
        self._shot = shot
        self._tree = tree
        self._root = root

        self._MDSTree = MDSplus.Tree(self._tree, self._shot)
        
        self._defaultUnits = {}
        
        #initialize None for non-essential data

        #flux-surface pressure
        self._fluxPres = None                                                #pressure on flux surface (psi,t)

        #fields
        self._btaxp = None                                                   #Bt on-axis, with plasma (t)
        self._btaxv = None                                                   #Bt on-axis, vacuum (t)
        self._bpolav = None                                                  #avg poloidal field (t)

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
        self._fluxVol = None                                                 #volume within flux surface (psi,t)
        self._volLCFS = None                                                 #volume within LCFS (t)
        self._qpsi = None                                                    #q profile (psi,t)
        self._RmidPsi = None                                                 #max major radius of flux surface (t,psi)
        
        # Call the get functions to preload the data. Add any other calls you
        # want to preload here.
        self.getTimeBase()
        self.getFluxGrid()
        self.getFluxLCFS()
        self.getFluxAxis()
        self.getFluxVol()
        self.getVolLCFS()
        self.getQProfile()
        self.getRmidPsi()  # loads _psiRZ, _rGrid and _zGrid at once.
        
    def __str__(self):
        """
        string formatting for EFITTree class.
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

    def getTreeInfo(self):
        """
        returns AttrDict of shot information
        outputs:
        AttrDict containing
            shot:   C-Mod shot index (long)
            tree:   EFIT tree (string)
            nr:     size of R-axis for spatial grid
            nz:     size of Z-axis for spatial grid
            nt:     size of timebase for flux grid
        """
        try:
            nt = len(self._time)
            nr = len(self._rGrid)
            nz = len(self._zGrid)
        except TypeError:
            nt, nr, nz = 0, 0, 0
            print 'tree has failed data load.'

        return AttrDict({'shot': self._shot, 'tree': self._tree, 'nr': nr, 'nz': nz, 'nt': nt})

    def getTimeBase(self):
        """
        returns EFIT time base vector
        """
        if self._time is None:
            try:
                timeNode = self._MDSTree.getNode(self._root+'a_eqdsk:time')
                self._time = timeNode.data()
                self._defaultUnits['_time'] = timeNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._time.copy()

    def getFluxGrid(self):
        """
        returns EFIT flux grid, [t,r,z]
        """
        if self._psiRZ is None:
            try:
                psinode = self._MDSTree.getNode(self._root+'g_eqdsk:psirz')
                self._psiRZ = psinode.data()
                self._rGrid = psinode.dim_of(0).data()
                self._zGrid = psinode.dim_of(1).data()
                self._defaultUnits['_psiRZ'] = psinode.units
                self._defaultUnits['_rGrid'] = psinode.dim_of(0).units
                self._defaultUnits['_zGrid'] = psinode.dim_of(1).units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiRZ.copy()

    def getRGrid(self, length_unit=1):
        """
        returns EFIT R-axis [r]
        """
        if self._rGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'], length_unit)
        return unit_factor * self._rGrid.copy()

    def getZGrid(self, length_unit=1):
        """
        returns EFIT Z-axis [z]
        """
        if self._zGrid is None:
            raise ValueError('data retrieval failed.')
        
        # Default units should be 'm'
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zGrid'], length_unit)
        return unit_factor * self._zGrid.copy()

    def getFluxAxis(self):
        """
        returns psi on magnetic axis [t]
        """
        if self._psiAxis is None:
            try:
                psiAxisNode = self._MDSTree.getNode(self._root+'a_eqdsk:simagx')
                self._psiAxis = psiAxisNode.data()
                self._defaultUnits['_psiAxis'] = psiAxisNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiAxis.copy()

    def getFluxLCFS(self):
        """
        returns psi at separatrix [t]
        """
        if self._psiLCFS is None:
            try:
                psiLCFSNode = self._MDSTree.getNode(self._root+'a_eqdsk:sibdry')
                self._psiLCFS = psiLCFSNode.data()
                self._defaultUnits['_psiLCFS'] = psiLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._psiLCFS.copy()

    def getFluxVol(self, length_unit=3):
        """
        returns volume within flux surface [psi,t]
        """
        if self._fluxVol is None:
            try:
                fluxVolNode = self._MDSTree.getNode(self._root+'fitout:volp')
                self._fluxVol = fluxVolNode.data()
                # Units aren't properly stored in the tree for this one!
                if fluxVolNode.units != ' ':
                    self._defaultUnits['_fluxVol'] = fluxVolNode.units
                else:
                    self._defaultUnits['_fluxVol'] = 'm^3'
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units are m^3, but aren't stored in the tree!
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_fluxVol'], length_unit)
        return unit_factor * self._fluxVol.copy()

    def getVolLCFS(self, length_unit=3):
        """
        returns volume within LCFS [t]
        """
        if self._volLCFS is None:
            try:
                volLCFSNode = self._MDSTree.getNode(self._root+'a_eqdsk:vout')
                self._volLCFS = volLCFSNode.data()
                self._defaultUnits['_volLCFS'] = volLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Default units should be 'cm^3':
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'], length_unit)
        return unit_factor * self._volLCFS.copy()

    def getRmidPsi(self, length_unit=1):
        """
        returns maximum major radius of each flux surface [t,psi]
        """
        if self._RmidPsi is None:
            try:
                RmidPsiNode = self._MDSTree.getNode(self._root+'fitout:rpres')
                self._RmidPsi = RmidPsiNode.data()
                # Units aren't properly stored in the tree for this one!
                if RmidPsiNode.units != ' ':
                    self._defaultUnits['_RmidPsi'] = RmidPsiNode.units
                else:
                    self._defaultUnits['_RmidPsi'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidPsi'], length_unit)
        return unit_factor * self._RmidPsi.copy()

    def getRLCFS(self, length_unit=1):
        """
        returns R-values of LCFS position [t,n]
        """
        if self._RLCFS is None:
            try:
                RLCFSNode = self._MDSTree.getNode(self._root+'g_eqdsk:rbbbs')
                self._RLCFS = RLCFSNode.data()
                self._defaultUnits['_RLCFS'] = RLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RLCFS'], length_unit)
        return unit_factor * self._RLCFS.copy()

    def getZLCFS(self, length_unit=1):
        """
        returns Z-values of LCFS position [t,n]
        """
        if self._ZLCFS is None:
            try:
                ZLCFSNode = self._MDSTree.getNode(self._root+'g_eqdsk:zbbbs')
                self._ZLCFS = ZLCFSNode.data()
                self._defaultUnits['_ZLCFS'] = ZLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_ZLCFS'], length_unit)
        return unit_factor * self._ZLCFS.copy()

    def getFluxPres(self):
        """
        returns pressure at flux surface [psi,t]
        """
        if self._fluxPres is None:
            try:
                fluxPresNode = self._MDSTree.getNode(self._root+'g_eqdsk:pres')
                self._fluxPres = fluxPresNode.data()
                self._defaultUnits['_fluxPres'] = fluxPresNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._fluxPres.copy()

    def getElongation(self):
        """
        returns LCFS elongation [t]
        """
        if self._kappa is None:
            try:
                kappaNode = self._MDSTree.getNode(self._root+'a_eqdsk:eout')
                self._kappa = kappaNode.data()
                self._defaultUnits['_kappa'] = kappaNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._kappa.copy()

    def getUpperTriangularity(self):
        """
        returns LCFS upper triangularity [t]
        """
        if self._dupper is None:
            try:
                dupperNode = self._MDSTree.getNode(self._root+'a_eqdsk:doutu')
                self._dupper = dupperNode.data()
                self._defaultUnits['_dupper'] = dupperNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dupper.copy()

    def getLowerTriangularity(self):
        """
        returns LCFS lower triangularity [t]
        """
        if self._dlower is None:
            try:
                dlowerNode = self._MDSTree.getNode(self._root+'a_eqdsk:doutl')
                self._dlower = dlowerNode.data()
                self._defaultUnits['_dlower']  = dlowerNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._dlower.copy()

    def getShaping(self):
        """
        pulls LCFS elongation and upper/lower triangularity
        returns AttrDict containing {kappa, delta_u, delta_l}
        """
        try:
            kap = self.getElongation()
            du = self.getUpperTriangularity()
            dl = self.getLowerTriangularity()
            return AttrDict({'kappa': kap, 'delta_u': du, 'delta_l': dl})
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getMagR(self, length_unit=1):
        """
        returns magnetic-axis major radius [t]
        """
        if self._rmag is None:
            try:
                rmagNode = self._MDSTree.getNode(self._root+'a_eqdsk:rmagx')
                self._rmag = rmagNode.data()
                self._defaultUnits['_rmag'] = rmagNode.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'], length_unit)
        return unit_factor * self._rmag.copy()

    def getMagZ(self, length_unit=1):
        """
        returns magnetic-axis Z [t]
        """
        if self._zmag is None:
            try:
                zmagNode = self._MDSTree.getNode(self._root+'a_eqdsk:zmagx')
                self._zmag = zmagNode.data()
                self._defaultUnits['_zmag'] = zmagNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zmag'], length_unit)
        return unit_factor * self._zmag.copy()

    def getAreaLCFS(self, length_unit=2):
        """
        returns LCFS cross-sectional area [t]
        """
        if self._areaLCFS is None:
            try:
                areaLCFSNode = self._MDSTree.getNode(self._root+'a_eqdsk:areao')
                self._areaLCFS = areaLCFSNode.data()
                self._defaultUnits['_areaLCFS'] = areaLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        # Units should be cm^2:
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_areaLCFS'], length_unit)
        return unit_factor * self._areaLCFS.copy()

    def getAOut(self, length_unit=1):
        """
        returns outboard-midplane minor radius at LCFS [t]
        """
        if self._aLCFS is None:
            try:
                aLCFSNode = self._MDSTree.getNode(self._root+'a_eqdsk:aout')
                self._aLCFS = aLCFSNode.data()
                self._defaultUnits['_aLCFS'] = aLCFSNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_aLCFS'], length_unit)
        return unit_factor * self._aLCFS.copy()

    def getRmidOut(self, length_unit=1):
        """
        returns outboard-midplane major radius [t]
        """
        if self._RmidLCFS is None:
            try:
                RmidLCFSNode = self._MDSTree.getNode(self._root+'a_eqdsk:rmidout')
                self._RmidLCFS = RmidLCFSNode.data()
                # The units aren't properly stored in the tree for this one!
                # Should be meters.
                if RmidLCFSNode.units != ' ':
                    self._defaultUnits['_RmidLCFS'] = RmidLCFSNode.units
                else:
                    self._defaultUnits['_RmidLCFS'] = 'm'
            except TreeException:
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidLCFS'], length_unit)
        return unit_factor * self._RmidLCFS.copy()

    def getGeometry(self, length_unit=None):
        """
        pulls dimensional geometry parameters
        returns AttrDict containing {magnetic-axis R,Z, LCFS area, outboard-midplane LCFS a,R}
        """
        try:
            Rmag = self.getMagR(length_unit=(length_unit if length_unit is not None else 1))
            Zmag = self.getMagZ(length_unit=(length_unit if length_unit is not None else 1))
            AreaLCFS = self.getAreaLCFS(length_unit=(length_unit if length_unit is not None else 2))
            aOut = self.getAOut(length_unit=(length_unit if length_unit is not None else 1))
            RmidOut = self.getRmidOut(length_unit=(length_unit if length_unit is not None else 1))
            return AttrDict({'Rmag': Rmag, 'Zmag': Zmag, 'AreaLCFS': AreaLCFS, 'aOut': aOut, 'RmidOut': RmidOut})
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getQProfile(self):
        """
        returns safety factor q [psi,t]
        """
        if self._qpsi is None:
            try:
                qpsiNode = self._MDSTree.getNode(self._root+'g_eqdsk:qpsi')
                self._qpsi = qpsiNode.data()
                self._defaultUnits['_qpsi'] = qpsiNode.units
            except TreeException:
                raise ValueError('data retrieval failed.')
        return self._qpsi.copy()

    def getQ0(self):
        """
        returns q on magnetic axis [t]
        """
        if self._q0 is None:
            try:
                q0Node = self._MDSTree.getNode(self._root+'a_eqdsk:qqmagx')
                self._q0 = q0Node.data()
                self._defaultUnits['_q0'] = q0Node.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q0.copy()

    def getQ95(self):
        """
        returns q at 95% flux surface [t]
        """
        if self._q95 is None:
            try:
                q95Node = self._MDSTree.getNode(self._root+'a_eqdsk:qpsib')
                self._q95 = q95Node.data()
                self._defaultUnits['_q95'] = q95Node.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._q95.copy()

    def getQLCFS(self):
        """
        returns q on LCFS [t]
        """
        if self._qLCFS is None:
            try:
                qLCFSNode = self._MDSTree.getNode(self._root+'a_eqdsk:qout')
                self._qLCFS = qLCFSNode.data()
                self._defaultUnits['_qLCFS'] = qLCFSNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._qLCFS.copy()

    def getQ1Surf(self, length_unit=1):
        """
        returns outboard-midplane minor radius of q=1 surface [t]
        """
        if self._rq1 is None:
            try:
                rq1Node = self._MDSTree.getNode(self._root+'a_eqdsk:aaq1')
                self._rq1 = rq1Node.data()
                self._defaultUnits['_rq1'] = rq1Node.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq1'], length_unit)
        return unit_factor * self._rq1.copy()

    def getQ2Surf(self, length_unit=1):
        """
        returns outboard-midplane minor radius of q=2 surface [t]
        """
        if self._rq2 is None:
            try:
                rq2Node = self._MDSTree.getNode(self._root+'a_eqdsk:aaq2')
                self._rq2 = rq2Node.data()
                self._defaultUnits['_rq2'] = rq2Node.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq2'], length_unit)
        return unit_factor * self._rq2.copy()

    def getQ3Surf(self, length_unit=1):
        """
        returns outboard-midplane minor radius of q=3 surface [t]
        """
        if self._rq3 is None:
            try:
                rq3Node = self._MDSTree.getNode(self._root+'a_eqdsk:aaq3')
                self._rq3 = rq3Node.data()
                self._defaultUnits['_rq3'] = rq3Node.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq3'], length_unit)
        return unit_factor * self._rq3.copy()

    def getQs(self, length_unit=1):
        """
        pulls q values
        returns AttrDict containing {q0,q95,qLCFS,rq1,rq2,rq3}
        """
        try:
            q0 = self.getQ0()
            q95 = self.getQ95()
            qLCFS = self.getQLCFS()
            rq1 = self.getQ1Surf(length_unit=length_unit)
            rq2 = self.getQ2Surf(length_unit=length_unit)
            rq3 = self.getQ3Surf(length_unit=length_unit)
            return AttrDict({'q0': q0, 'q95': q95, 'qLCFS': qLCFS, 'rq1': rq1, 'rq2': rq2, 'rq3': rq3})
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getBtVac(self):
        """
        returns on-axis vacuum toroidal field [t]
        """
        if self._btaxv is None:
            try:
                btaxvNode = self._MDSTree.getNode(self._root+'a_eqdsk:btaxv')
                self._btaxv = btaxvNode.data()
                self._defaultUnits['_btaxv'] = btaxvNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxv.copy()

    def getBtPla(self):
        """
        returns on-axis plasma toroidal field [t]
        """
        if self._btaxp is None:
            try:
                btaxpNode = self._MDSTree.getNode(self._root+'a_eqdsk:btaxp')
                self._btaxp = btaxpNode.data()
                self._defaultUnits['_btaxp'] = btaxpNode.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._btaxp.copy()

    def getBpAvg(self):
        """
        returns average poloidal field [t]
        """
        if self._bpolav is None:
            try:
                bpolavNode = self._MDSTree.getNode(self._root+'a_eqdsk:bpolav')
                self._bpolav = bpolavNode.data()
                self._defaultUnits['_bpolav'] = bpolavNode.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._bpolav.copy()

    def getFields(self):
        """
        pulls vacuum and plasma toroidal field, avg poloidal field
        returns AttrDict containing {btaxv,btaxp,bpolav}
        """
        try:
            btaxv = self.getBtVac()
            btaxp = self.getBtPla()
            bpolav = self.getBpAvg()
            return AttrDict({'BtVac': btaxv, 'BtPla': btaxp, 'BpAvg': bpolav})
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getIpCalc(self):
        """
        returns EFIT-calculated plasma current [t]
        """
        if self._IpCalc is None:
            try:
                IpCalcNode = self._MDSTree.getNode(self._root+'a_eqdsk:cpasma')
                self._IpCalc = IpCalcNode.data()
                self._defaultUnits['_IpCalc'] = IpCalcNode.units
            except (TreeException,AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpCalc.copy()

    def getIpMeas(self):
        """
        returns magnetics-measured plasma current [t]
        """
        if self._IpMeas is None:
            try:
                IpMeasNode = self._MDSTree.getNode(self._root+'a_eqdsk:pasmat')
                self._IpMeas = IpMeasNode.data()
                self._defaultUnits['_IpMeas'] = IpMeasNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._IpMeas.copy()

    def getJp(self):
        """
        returns EFIT-calculated plasma current density Jp on flux grid [t,r,z]
        """
        if self._Jp is None:
            try:
                JpNode = self._MDSTree.getNode(self._root+'g_eqdsk:pcurrt')
                self._Jp = JpNode.data()
                # Units come in as 'a': am I missing something about the
                # definition of this quantity?
                self._defaultUnits['_Jp'] = JpNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Jp.copy()

    def getBetaT(self):
        """
        returns EFIT-calculated toroidal beta [t]
        """
        if self._betat is None:
            try:
                betatNode = self._MDSTree.getNode(self._root+'a_eqdsk:betat')
                self._betat = betatNode.data()
                self._defaultUnits['_betat'] = betatNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betat.copy()

    def getBetaP(self):
        """
        returns EFIT-calculated poloidal beta [t]
        """
        if self._betap is None:
            try:
                betapNode = self._MDSTree.getNode(self._root+'a_eqdsk:betap')
                self._betap = betapNode.data()
                self._defaultUnits['_betap'] = betapNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betap.copy()

    def getLi(self):
        """
        returns EFIT-calculated internal inductance [t]
        """
        if self._Li is None:
            try:
                LiNode = self._MDSTree.getNode(self._root+'a_eqdsk:ali')
                self._Li = LiNode.data()
                self._defaultUnits['_Li'] = LiNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Li.copy()

    def getBetas(self):
        """
        pulls calculated betap, betat, internal inductance
        returns AttrDict containing {betat,betap,Li}
        """
        try:
            betat = self.getBetaT()
            betap = self.getBetaP()
            Li = self.getLi()
            return AttrDict({'betat': betat, 'betap': betap, 'Li': Li})
        except ValueError:
                raise ValueError('data retrieval failed.')

    def getDiamagFlux(self):
        """
        returns measured diamagnetic-loop flux [t]
        """
        if self._diamag is None:
            try:
                diamagNode = self._MDSTree.getNode(self._root+'a_eqdsk:diamag')
                self._diamag = diamagNode.data()
                self._defaultUnits['_diamag'] = diamagNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._diamag.copy()

    def getDiamagBetaT(self):
        """
        returns diamagnetic-loop toroidal beta [t]
        """
        if self._betatd is None:
            try:
                betatdNode = self._MDSTree.getNode(self._root+'a_eqdsk:betatd')
                self._betatd = betatdNode.data()
                self._defaultUnits['_betatd'] = betatdNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betatd.copy()

    def getDiamagBetaP(self):
        """
        returns diamagnetic-loop avg poloidal beta [t]
        """
        if self._betapd is None:
            try:
                betapdNode = self._MDSTree.getNode(self._root+'a_eqdsk:betapd')
                self._betapd = betapdNode.data()
                self._defaultUnits['_betapd'] = betapdNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._betapd.copy()

    def getDiamagTauE(self):
        """
        returns diamagnetic-loop energy confinement time [t]
        """
        if self._tauDiamag is None:
            try:
                tauDiamagNode = self._MDSTree.getNode(self._root+'a_eqdsk:taudia')
                self._tauDiamag = tauDiamagNode.data()
                self._defaultUnits['_tauDiamag'] = tauDiamagNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._tauDiamag.copy()

    def getDiamagWp(self):
        """
        returns diamagnetic-loop plasma stored energy [t]
        """
        if self._WDiamag is None:
            try:
                WDiamagNode = self._MDSTree.getNode(self._root+'a_eqdsk:wplasmd')
                self._WDiamag = WDiamagNode.data()
                self._defaultUnits['_WDiamag'] = WDiamagNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WDiamag.copy()

    def getDiamag(self):
        """
        pulls diamagnetic flux measurements, toroidal and poloidal beta, energy confinement time and stored energy
        returns AttrDict containing {diamag. flux, betatd, betapd, tauDiamag, WDiamag}
        """
        try:
            dFlux = self.getDiamagFlux()
            betatd = self.getDiamagBetaT()
            betapd = self.getDiamagBetaP()
            dTau = self.getDiamagTauE()
            dWp = self.getDiamagWp()
            return AttrDict({'diaFlux': dFlux, 'diaBetat': betatd, 'diaBetap': betapd, 'diaTauE': dTau, 'diaWp': dWp})
        except ValueError:
                raise ValueError('data retrieval failed.')

    def getWMHD(self):
        """
        returns EFIT-calculated MHD stored energy [t]
        """
        if self._WMHD is None:
            try:
                WMHDNode = self._MDSTree.getNode(self._root+'a_eqdsk:wplasm')
                self._WMHD = WMHDNode.data()
                self._defaultUnits['_WMHD'] = WMHDNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._WMHD.copy()

    def getTauMHD(self):
        """
        returns EFIT-calculated MHD energy confinement time [t]
        """
        if self._tauMHD is None:
            try:
                tauMHDNode = self._MDSTree.getNode(self._root+'a_eqdsk:taumhd')
                self._tauMHD = tauMHDNode.data()
                self._defaultUnits['_tauMHD'] = tauMHDNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._tauMHD.copy()

    def getPinj(self):
        """
        returns EFIT-calculated injected power [t]
        """
        if self._Pinj is None:
            try:
                PinjNode = self._MDSTree.getNode(self._root+'a_eqdsk:pbinj')
                self._Pinj = PinjNode.data()
                self._defaultUnits['_Pinj'] = PinjNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Pinj.copy()

    def getWbdot(self):
        """
        returns EFIT-calculated d/dt of magnetic stored energy [t]
        """
        if self._Wbdot is None:
            try:
                WbdotNode = self._MDSTree.getNode(self._root+'a_eqdsk:wbdot')
                self._Wbdot = WbdotNode.data()
                self._defaultUnits['_Wbdot'] = WbdotNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Wbdot.copy()

    def getWpdot(self):
        """
        returns EFIT-calculated d/dt of plasma stored energy [t]
        """
        if self._Wpdot is None:
            try:
                WpdotNode = self._MDSTree.getNode(self._root+'a_eqdsk:wpdot')
                self._Wpdot = WpdotNode.data()
                self._defaultUnits['_Wpdot'] = WpdotNode.units
            except (TreeException, AttributeError):
                raise ValueError('data retrieval failed.')
        return self._Wpdot.copy()

    def getEnergy(self):
        """
        pulls EFIT-calculated energy parameters - stored energy, tau_E, injected power, d/dt of magnetic and plasma stored energy
        returns AttrDict containing {WMHD,tauMHD,Pinj,Wbdot,Wpdot}
        """
        try:
            WMHD = self.getWMHD()
            tauMHD = self.getTauMHD()
            Pinj = self.getPinj()
            Wbdot = self.getWbdot()
            Wpdot = self.getWpdot()
            return AttrDict({'WMHD': WMHD, 'tauMHD': tauMHD, 'Pinj': Pinj, 'Wbdot': Wbdot, 'Wpdot': Wpdot})
        except ValueError:
            raise ValueError('data retrieval failed.')

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's
        IDL implementation efit_rz2psi.pro."""
        if self._currentSign is None:
            self._currentSign = 1 if scipy.mean(self.getIpMeas()) > 1e5 else -1
        return self._currentSign

    def getParam(self, path):
        """
        backup function - path to parameter as input, returns desired variable
        acts as wrapper for MDS call
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


class CModEFITTree(EFITTree):
    """
    Inherits EFITTree class. machine-specific data handling class for Alcator C-Mod.
    Pulls EFIT data from selected MDS tree and shot, stores as object attributes.
    Each EFIT variable or set of variables is recovered with a corresponding getter method.
    Essential data for EFIT mapping are pulled on initialization (e.g. psirz grid).
    Additional data are pulled at the first request and stored for subsequent usage.
    """

    def __init__(self, shot, tree='ANALYSIS', length_unit='m', tspline=False):
        """
        Intializes C-Mod version of EFITTree object.  Pulls data from MDS tree for storage
        in instance attributes.  Core attributes are populated from the MDS tree on initialization.
        Additional attributes are initialized as None, filled on the first request to the object.

        INPUTS:
        shot:   C-Mod shot index (long)
        tree:   optional input for EFIT tree, defaults to 'ANALYSIS' (i.e.,
                    EFIT data are under \\analysis::top.efit.results).
                    For any string TREE (such as 'EFIT20') other than 'ANALYSIS',
                    data are taken from \\TREE::top.results.
        """
        if tree.upper() == 'ANALYSIS':
            root = '\\analysis::top.efit.results.'
        else:
            root = '\\'+tree+'::top.results.'

        super(CModEFITTree, self).__init__(shot, tree, root, length_unit=length_unit, tspline=tspline)
    
    def getMachineCrossSection(self):
        """
        Pulls C-Mod cross-section data from tree, converts to plottable
        vector format for use in other plotting routines

        INPUTS:
        shot:   C-Mod shot index (used for tree access) (long)

        OUTPUTS:

        """
        #pull cross-section from tree
        try:
            ccT = MDSplus.Tree('analysis', self._shot)
            path = '\\analysis::top.limiters.tiles:'
            xvctr = ccT.getNode(path+'XTILE').data()
            yvctr = ccT.getNode(path+'YTILE').data()
            nvctr = ccT.getNode(path+'NSEG').data()
            lvctr = ccT.getNode(path+'PTS_PER_SEG').data()
        except MDSplus._treeshr.TreeException:
            raise ValueError('data load failed.')

        #xvctr, yvctr stored as [nvctr,npts] ndarray.  Each of [nvctr] rows
        #represents the x- or y- coordinate of a line segment; lvctr stores the length
        #of each line segment row.  x/yvctr rows padded out to npts = max(lvctr) with
        #uniform zeros.  To rapidly plot this, we want to flatten xvctr,yvctr down to
        #a single x,y vector set.  We'll use Nones to separate line segments (as these
        #break the continuous plotting joints).
        x = []
        y = []
        for i in range(nvctr):
            length = lvctr[i]
            xseg = xvctr[i,0:length]
            yseg = yvctr[i,0:length]
            x.extend(xseg)
            y.extend(yseg)
            if i != nvctr-1:
                x.append(None)
                y.append(None)

        x = scipy.array(x)
        y = scipy.array(y)
        return (x, y)


class CModEFITTreeProp(CModEFITTree, PropertyAccessMixin):
    """CModEFITTree with the PropertyAccessMixin added to enable property-style
    access. This is good for interactive use, but may drag the performance down."""
    pass
