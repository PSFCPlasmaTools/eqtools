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
import _tricub


class Spline():
    """Tricubic interpolating spline with forced edge derivative equal zero
    conditions.  It assumes a cartesian grid.  The ordering of f[z,y,x] is
    extremely important for the proper evaluation of the spline.  It assumes
    that f is in C order.
    
    Create a new Spline instance.

    Args:
        z (1-dimensional float array): Values of the positions of the 1st
            Dimension of f. Must be monotonic without duplicates.
        y (1-dimensional float array): Values of the positions of the 2nd
            dimension of f. Must be monotonic without duplicates.
        x (1-dimensional float array): Values of the positions of the 3rd
            dimension of f. Must be monotonic without duplicates.
        f (3-dimensional float array): f[z,y,x]. NaN and Inf will hamper
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
            
            map = Spline(z, y, x, f)
    
        Evaluate Trispline instance map at x1, y1, z1::
            
            output = map.ev(z1, y1, x1)
    
    """
    def __init__(self, z, y, x, f, regular=True, fast=False, dx=0, dy=0, dz=0):
        if dx != 0 or dy != 0 or dz != 0:
            raise NotImplementedError(
                "Trispline derivatives are not implemented, do not use tricubic "
                "interpolation if you need to compute magnetic fields!"
            )
        self._f = scipy.zeros(scipy.array(f.shape)+(2,2,2)) #pad the f array so as to force the Neumann Boundary Condition
        self._f[1:-1,1:-1,1:-1] = scipy.array(f) # place f in center, so that it is padded by unfilled values on all sides
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

        if len(x) == self._f.shape[2]-2:
            self._x = scipy.array(x,dtype=float)
            if not _tricub.ismonotonic(self._x):
                raise ValueError("x is not monotonic")
            # add pad values to x for evaluation
            self._x = scipy.insert(self._x,0,2*self._x[0]-self._x[1])
            self._x = scipy.append(self._x,2*self._x[-1]-self._x[-2])
        else:
            raise ValueError("dimension of x does not match that of f ")

        if len(y) == self._f.shape[1]-2:
            self._y = scipy.array(y,dtype=float)
            if not _tricub.ismonotonic(self._y):
                raise ValueError("y is not monotonic")
            # add pad values for y for evaluation
            self._y = scipy.insert(self._y,0,2*self._y[0]-self._y[1])
            self._y = scipy.append(self._y,2*self._y[-1]-self._y[-2])
        else:
            raise ValueError("dimension of y does not match that of f ")
        
        if len(z) == self._f.shape[0]-2:
            self._z = scipy.array(z,dtype=float)
            if not _tricub.ismonotonic(self._z):
                raise ValueError("z is not monotonic")
            # add pad values for z for evaluation
            self._z = scipy.insert(self._z,0,2*self._z[0]-self._z[1])
            self._z = scipy.append(self._z,2*self._z[-1]-self._z[-2])
        else:
            raise ValueError("dimension of z does not match that of f ")

        self._regular = regular
        self._fast = fast
       # if not regular:
       #     for i in x,y,z:
       #         regular = regular or bool(_tricub.isregular(i))
       #     self._regular = regular

    def ev(self, z1, y1, x1):
        """evaluates tricubic spline at point (x1,y1,z1) which is f[z1,y1,x1].

        Args:
            z1 (scalar float or 1-dimensional float): Position in z dimension.
               This is the first dimension of 3d-valued grid.
            y1 (scalar float or 1-dimensional float): Position in y dimension.
               This is the second dimension of 3d-valued grid.
            x1 (scalar float or 1-dimensional float): Position in x dimension. 
               This is the third dimension of 3d-valued grid.

        Returns:
            val (array or scalar float): The interpolated value at (x1,y1,z1).
            
        Raises:
            ValueError: If any of the dimensions exceed the evaluation boundary
                of the grid

        """
        x = scipy.atleast_1d(x1)
        y = scipy.atleast_1d(y1)
        z = scipy.atleast_1d(z1) # This will not modify x1,y1,z1.
        val = scipy.nan*scipy.zeros(x.shape)

        if scipy.any(x < self._x[1]) or scipy.any(x > self._x[-2]):
            raise ValueError('x value exceeds bounds of interpolation grid ')
        if scipy.any(y < self._y[1]) or scipy.any(y > self._y[-2]):
            raise ValueError('y value exceeds bounds of interpolation grid ')
        if scipy.any(z < self._z[1]) or scipy.any(z > self._z[-2]):
            raise ValueError('z value exceeds bounds of interpolation grid ')

        xinp = scipy.array(scipy.where(scipy.isfinite(x)))
        yinp = scipy.array(scipy.where(scipy.isfinite(y)))
        zinp = scipy.array(scipy.where(scipy.isfinite(z)))
        inp = scipy.intersect1d(scipy.intersect1d(xinp, yinp), zinp)
        if inp.size != 0:
            
            if self._fast:
                ix = scipy.clip(scipy.digitize(x[inp],self._x),2,self._x.size - 2) - 1
                iy = scipy.clip(scipy.digitize(y[inp],self._y),2,self._y.size - 2) - 1
                iz = scipy.clip(scipy.digitize(z[inp],self._z),2,self._z.size - 2) - 1
                pos = ix - 1 + self._f.shape[1]*((iy - 1) + self._f.shape[2]*(iz - 1))
                indx = scipy.argsort(pos) # I believe this is much faster...

                if self._regular:
                    dx =  (x[inp]-self._x[ix])/(self._x[ix+1]-self._x[ix])
                    dy =  (y[inp]-self._y[iy])/(self._y[iy+1]-self._y[iy])
                    dz =  (z[inp]-self._z[iz])/(self._z[iz+1]-self._z[iz])
                    val[inp] = _tricub.reg_eval(dx,dy,dz,self._f,pos,indx)

                else:
                    val[inp] = _tricub.nonreg_eval(x[inp],y[inp],z[inp],self._f,self._x,self._y,self._z,pos,indx,ix,iy,iz)
                    
            else:
                
                if self._regular:
                    val[inp] = _tricub.reg_ev(x[inp], y[inp], z[inp], self._f, self._x, self._y, self._z)  
                else:
                    val[inp] = _tricub.nonreg_ev(x[inp], y[inp], z[inp], self._f, self._x, self._y, self._z)

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
