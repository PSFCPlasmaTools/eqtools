# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
#    This file is part of eqtools.
#
#    EqTools is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    EqTools is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with EqTools.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright 2013 Ian C. Faust
""" This module provides interface to the tricubic spline interpolator. It also
contains an enhanced bivariate spline which generates bounds errors.
"""


import scipy 
import scipy.interpolate
try:
    import _tricub
except:
    # Won't be able to use actual trispline, but still can use other routines.
    pass


class Spline():
    """Tricubic interpolating spline with forced edge derivative equal zero
    conditions.  It assumes a cartesian grid.  The ordering of f[z,y,x] is
    extremely important for the proper evaluation of the spline.  It assumes
    that f is in C order.
    
    Create a new Spline instance.

    Args:
        x (1-dimensional float array): Values of the positions of the 1st
            Dimension of f. Must be monotonic without duplicates.
        y (1-dimensional float array): Values of the positions of the 2nd
            dimension of f. Must be monotonic without duplicates.
        z (1-dimensional float array): Values of the positions of the 3rd
            dimension of f. Must be monotonic without duplicates.
        f (3-dimensional float array): f[x,y,z]. NaN and Inf will hamper
            performance and affect interpolation in 4x4x4 space about its value.
    
    Keyword Args:
        regular (Boolean): If the grid is known to be regular, forces 
            matrix-based fast evaluation of interpolation.
        fast (Boolean): Outdated input to test the indexing performance of the
            c code vs internal python handling.
    
    Raises:
        ValueError: If any of the dimensions do not match specified f dim
        ValueError: If x,y, or z are not monotonic
        
    Examples:
        All assume that `x`, `y`, `z`, and `f` are valid instances of the appropriate
        numpy arrays which take independent variables x,y,z and create numpy array
        f. `x1`, `y1`, and `z1` are numpy arrays which data f is to be interpolated.
            
        Generate a Trispline instance map with data x, y, z and f::
            
            map = Spline(x, y, z, f)
    
        Evaluate Trispline instance map at x1, y1, z1::
            
            output = map.ev(x1, y1, z1)
    
    """
    def __init__(self, x, y, z, f, boundary = 'natural', dx=0, dy=0, dz=0, bounds_error=True, fill_value=scipy.nan):
        #if dx != 0 or dy != 0 or dz != 0:
        #    raise NotImplementedError(
        #        "Trispline derivatives are not implemented, do not use tricubic "
        #        "interpolation if you need to compute magnetic fields!"
        #    )

        
        self._x = scipy.array(x,dtype=float)
        self._y = scipy.array(y,dtype=float)
        self._z = scipy.array(z,dtype=float)

        self._xlim = scipy.array((x.min(), x.max()))
        self._ylim = scipy.array((y.min(), y.max()))
        self._zlim = scipy.array((z.min(), z.max()))

        self._dx = scipy.array(dx, dtype=int)
        self._dy = scipy.array(dy, dtype=int)
        self._dz = scipy.array(dz, dtype=int)
        
        self.bounds_error = bounds_error
        self.fill_value = fill_value

        if f.shape != (self._x.size,self._y.size,self._z.size):
            raise ValueError("dimensions do not match f")
            
        if _tricub.ismonotonic(self._x) and _tricub.ismonotonic(self._y) and _tricub.ismonotonic(self._z):
            self._x = scipy.insert(self._x,0,2*self._x[0]-self._x[1])
            self._x = scipy.append(self._x,2*self._x[-1]-self._x[-2])
            self._y = scipy.insert(self._y,0,2*self._y[0]-self._y[1])
            self._y = scipy.append(self._y,2*self._y[-1]-self._y[-2])
            self._z = scipy.insert(self._z,0,2*self._z[0]-self._z[1])
            self._z = scipy.append(self._z,2*self._z[-1]-self._z[-2])

        
        self._f = scipy.zeros(scipy.array(f.shape)+(2,2,2))
        self._f[1:-1,1:-1,1:-1] = scipy.array(f) # place f in center, so that it is padded by unfilled values on all sides
        
        if boundary == 'clamped':
            # faces
            self._f[(0,-1),1:-1,1:-1] = f[(0,-1),:,:] 
            self._f[1:-1,(0,-1),1:-1] = f[:,(0,-1),:]
            self._f[1:-1,1:-1,(0,-1)] = f[:,:,(0,-1)]
            #verticies
            self._f[(0,0,-1,-1),(0,-1,0,-1),1:-1] = f[(0,0,-1,-1),(0,-1,0,-1),:] 
            self._f[(0,0,-1,-1),1:-1,(0,-1,0,-1)] = f[(0,0,-1,-1),:,(0,-1,0,-1)]
            self._f[1:-1,(0,0,-1,-1),(0,-1,0,-1)] = f[:,(0,0,-1,-1),(0,-1,0,-1)]
            #corners
            self._f[(0,0,0,0,-1,-1,-1,-1),(0,0,-1,-1,0,0,-1,-1),(0,-1,0,-1,0,-1,0,-1)] = f[(0,0,0,0,-1,-1,-1,-1),(0,0,-1,-1,0,0,-1,-1),(0,-1,0,-1,0,-1,0,-1)]
        elif boundary == 'natural':
            # faces
            self._f[(0,-1),1:-1,1:-1] = 2*f[(0,-1),:,:] - f[(1,-2),:,:]
            self._f[1:-1,(0,-1),1:-1] = 2*f[:,(0,-1),:] - f[:,(1,-2),:]
            self._f[1:-1,1:-1,(0,-1)] = 2*f[:,:,(0,-1)] - f[:,:,(1,-2)]
            #verticies
            self._f[(0,0,-1,-1),(0,-1,0,-1),1:-1] = 4*f[(0,0,-1,-1),(0,-1,0,-1),:] - f[(1,1,-2,-2),(0,-1,0,-1),:] - f[(0,0,-1,-1),(1,-2,1,-2),:] - f[(1,1,-2,-2),(1,-2,1,-2),:]
            self._f[(0,0,-1,-1),1:-1,(0,-1,0,-1)] = 4*f[(0,0,-1,-1),:,(0,-1,0,-1)] - f[(1,1,-2,-2),:,(0,-1,0,-1)] - f[(0,0,-1,-1),:,(1,-2,1,-2)] - f[(1,1,-2,-2),:,(1,-2,1,-2)]
            self._f[1:-1,(0,0,-1,-1),(0,-1,0,-1)] = 4*f[:,(0,0,-1,-1),(0,-1,0,-1)] - f[:,(1,1,-2,-2),(0,-1,0,-1)] - f[:,(0,0,-1,-1),(1,-2,1,-2)] - f[:,(1,1,-2,-2),(1,-2,1,-2)]
            #corners
            self._f[(0,0,0,0,-1,-1,-1,-1),(0,0,-1,-1,0,0,-1,-1),(0,-1,0,-1,0,-1,0,-1)] = 8*f[(0,0,0,0,-1,-1,-1,-1),(0,0,-1,-1,0,0,-1,-1),(0,-1,0,-1,0,-1,0,-1)] -f[(1,1,1,1,-2,-2,-2,-2),(0,0,-1,-1,0,0,-1,-1),(0,-1,0,-1,0,-1,0,-1)] -f[(0,0,0,0,-1,-1,-1,-1),(1,1,-2,-2,1,1,-2,-2),(0,-1,0,-1,0,-1,0,-1)] -f[(0,0,0,0,-1,-1,-1,-1),(0,0,-1,-1,0,0,-1,-1),(1,-2,1,-2,1,-2,1,-2)] -f[(1,1,1,1,-2,-2,-2,-2),(1,1,-2,-2,1,1,-2,-2),(0,-1,0,-1,0,-1,0,-1)] -f[(0,0,0,0,-1,-1,-1,-1),(1,1,-2,-2,1,1,-2,-2),(1,-2,1,-2,1,-2,1,-2)] -f[(1,1,1,1,-2,-2,-2,-2),(0,0,-1,-1,0,0,-1,-1),(1,-2,1,-2,1,-2,1,-2)] -f[(1,1,1,1,-2,-2,-2,-2),(1,1,-2,-2,1,1,-2,-2),(1,-2,1,-2,1,-2,1,-2)]

        self._regular = False
        if _tricub.isregular(self._x) and _tricub.isregular(self._y) and _tricub.isregular(self._z):
            self._regular = True

            
    def _check_bounds(self, x_new, y_new, z_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Args:
            x_new (float array):
            
            y_new (float array):

        Returns:
            out_of_bounds (Boolean array): The mask on x_new and y_new of
            values that are NOT of bounds.
        """
        below_bounds_x = x_new < self._xlim[0]
        above_bounds_x = x_new > self._xlim[1]

        below_bounds_y = y_new < self._ylim[0]
        above_bounds_y = y_new > self._ylim[1]
      
        below_bounds_z = z_new < self._zlim[0]
        above_bounds_z = z_new > self._zlim[1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds_x.any():
            raise ValueError("A value in x is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds_x.any():
            raise ValueError("A value in x is above the interpolation "
                "range.")
        if self.bounds_error and below_bounds_y.any():
            raise ValueError("A value in y is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds_y.any():
            raise ValueError("A value in y is above the interpolation "
                "range.")
        if self.bounds_error and below_bounds_z.any():
            raise ValueError("A value in z is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds_z.any():
            raise ValueError("A value in z is above the interpolation "
                "range.")

        out_of_bounds = scipy.logical_not(scipy.logical_or(scipy.logical_or(scipy.logical_or(below_bounds_x, above_bounds_x),
                                                                            scipy.logical_or(below_bounds_y, above_bounds_y)),
                                                           scipy.logical_or(below_bounds_z, above_bounds_z)))
        return out_of_bounds


            
    def ev(self, xi, yi, zi, dx=0, dy=0, dz=0):
        """evaluates tricubic spline at point (xi,yi,zi) which is f[xi,yi,zi].

        Args:
            xi (scalar float or 1-dimensional float): Position in x dimension.
               This is the first dimension of 3d-valued grid.
            yi (scalar float or 1-dimensional float): Position in y dimension.
               This is the second dimension of 3d-valued grid.
            zi (scalar float or 1-dimensional float): Position in z dimension. 
               This is the third dimension of 3d-valued grid.

        Returns:
            val (array or scalar float): The interpolated value at (xi,yi,zi).
            
        Raises:
            ValueError: If any of the dimensions exceed the evaluation boundary
                of the grid
        
        """
        x = scipy.atleast_1d(xi)
        y = scipy.atleast_1d(yi)
        z = scipy.atleast_1d(zi) # This will not modify x1,y1,z1.

        val = self.fill_value*scipy.ones(x.shape)
        idx = self._check_bounds(x, y, z)

        if dx == 0:
            dx = self._dx

        if dy == 0:
            dy = self._dy

        if dz == 0:
            dz = self._dz
        
        if z[idx].size != 0:
            if self._regular:
                if dx or dy or dz:
                    val[idx] = _tricub.reg_ev_full(z[idx], y[idx], x[idx], self._f, self._z, self._y, self._x, dz, dy, dx) 
                else:
                    val[idx] = _tricub.reg_ev(z[idx], y[idx], x[idx], self._f, self._z, self._y, self._x)  

            else:
                if dx or dy or dz:
                    val[idx] = _tricub.nonreg_ev_full(z[idx], y[idx], x[idx], self._f, self._z, self._y, self._x, dz, dy, dx)
                else:
                    val[idx] = _tricub.nonreg_ev(z[idx], y[idx], x[idx], self._f, self._z, self._y, self._x)

        return val


class RectBivariateSpline(scipy.interpolate.RectBivariateSpline):
    """the lack of a graceful bounds error causes the fortran to fail hard. 
    This masks scipy.interpolate.RectBivariateSpline with a proper bound
    checker and value filler such that it will not fail in use for EqTools
 
    Can be used for both smoothing and interpolating data.

    Args:
        x (1-dimensional float array):
            1-D array of coordinates in monotonically increasing order.
        y (1-dimensional float array):
            1-D array of coordinates in monotonically increasing order.
        z (2-dimensional float array):
            2-D array of data with shape (x.size,y.size).

    Keyword Args:
        bbox (1-dimensional float): Sequence of length 4 specifying the
            boundary of the rectangular approximation domain.  By default,
            ``bbox=[min(x,tx),max(x,tx), min(y,ty),max(y,ty)]``.
        kx (integer): Degrees of the bivariate spline. Default is 3.
        ky (integer): Degrees of the bivariate spline. Default is 3.
        s (float): Positive smoothing factor defined for estimation condition,
            ``sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s``
            Default is ``s=0``, which is for interpolation.
    """

    def __init__(self, x, y, z, bbox=[None] *4, kx=3, ky=3, s=0, bounds_error=True, fill_value=scipy.nan):

        super(RectBivariateSpline, self).__init__( x, y, z, bbox=bbox, kx=kx, ky=ky, s=s)
        self._xlim = scipy.array((x.min(), x.max()))
        self._ylim = scipy.array((y.min(), y.max()))
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def _check_bounds(self, x_new, y_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Args:
            x_new (float array):
            
            y_new (float array):

        Returns:
            out_of_bounds (Boolean array): The mask on x_new and y_new of
            values that are NOT of bounds.
        """
        below_bounds_x = x_new < self._xlim[0]
        above_bounds_x = x_new > self._xlim[1]

        below_bounds_y = y_new < self._ylim[0]
        above_bounds_y = y_new > self._ylim[1]

        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and below_bounds_x.any():
            raise ValueError("A value in x is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds_x.any():
            raise ValueError("A value in x is above the interpolation "
                "range.")
        if self.bounds_error and below_bounds_y.any():
            raise ValueError("A value in y is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds_y.any():
            raise ValueError("A value in y is above the interpolation "
                "range.")

        out_of_bounds = scipy.logical_not(scipy.logical_or(scipy.logical_or(below_bounds_x, above_bounds_x),
                                                           scipy.logical_or(below_bounds_y, above_bounds_y)))
        return out_of_bounds


    def ev(self, xi, yi):
        """Evaluate the rectBiVariateSpline at (xi,yi).  (x,y)values are
           checked for being in the bounds of the interpolated data.

        Args:
            xi (float array): input x dimensional values 
            yi (float array): input x dimensional values 

        Returns:
            val (float array): evaluated spline at points (x[i], y[i]), i=0,...,len(x)-1
        """
        idx = self._check_bounds(xi, yi)
        # print(idx)
        zi = self.fill_value*scipy.ones(xi.shape)
        zi[idx] = super(RectBivariateSpline, self).ev(scipy.atleast_1d(xi)[idx],
                                                      scipy.atleast_1d(yi)[idx])
        return zi

class BivariateInterpolator(object):
    """This class provides a wrapper for `scipy.interpolate.CloughTocher2DInterpolator`.
    
    This is necessary because `scipy.interpolate.SmoothBivariateSpline` cannot
    be made to interpolate, and gives inaccurate answers near the boundaries.
    """
    def __init__(self, x, y, z):
        self._ct_interp = scipy.interpolate.CloughTocher2DInterpolator(
            scipy.hstack((scipy.atleast_2d(x).T, scipy.atleast_2d(y).T)),
            z
        )
    
    def ev(self, xi, yi):
        return self._ct_interp(
            scipy.hstack((scipy.atleast_2d(xi).T, scipy.atleast_2d(yi).T))
        )

class UnivariateInterpolator(scipy.interpolate.InterpolatedUnivariateSpline):
    """Interpolated spline class which overcomes the shortcomings of interp1d
    (inaccurate near edges) and InterpolatedUnivariateSpline (can't set NaN
    where it extrapolates).
    """
    def __init__(self, *args, **kwargs):
        self.min_val = kwargs.pop('minval', None)
        self.max_val = kwargs.pop('maxval', None)
        if kwargs.pop('enforce_y', True):
            if self.min_val is None:
                self.min_val = min(args[1])
            if self.max_val is None:
                self.max_val = max(args[1])
        super(UnivariateInterpolator, self).__init__(*args, **kwargs)
    
    def __call__(self, x, *args, **kwargs):
        x = scipy.asarray(x, dtype=float)
        out = super(UnivariateInterpolator, self).__call__(x, *args, **kwargs)
        if self.min_val is not None:
            out[out < self.min_val] = self.min_val
        if self.max_val is not None:
            out[out > self.max_val] = self.max_val
        out[(x < self.get_knots().min()) | (x > self.get_knots().max())] = scipy.nan
        return out
