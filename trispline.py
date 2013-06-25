import scipy as SP
import _tricubic

"""
    This file is part of the EqTools package.

    EqTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqTools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2013 Ian C. Faust
"""

class spline():


    def __init__(self,x,y,z,f):
        self._f = SP.zeros(SP.array(f.shape)+(2,2,2)) #pad the f array so as to force the Neumann Boundary Condition
        self._f[1:-1,1:-1,1:-1] = SP.array(f) # place f in center, so that it is padded by unfilled values on all sides
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

        self._x = SP.array(x)
        self._y = SP.array(y)
        self._z = SP.array(z)

    def ev(self,x1,y1,z1):
        x = SP.array((x1,))
        y = SP.array((y1,))
        z = SP.array((z1,)) # This will not modify x1,y1,z1.
        val = SP.nan*SP.zeros(x.shape)
        print(z)
        if SP.any(x < self._x[0]) or SP.any(x > self._x[-1]):
            raise ValueError('x value exceeds bounds of interpolation grid ')
        if SP.any(y < self._y[0]) or SP.any(y > self._y[-1]):
            raise ValueError('y value exceeds bounds of interpolation grid ')
        if SP.any(z < self._z[0]) or SP.any(z > self._z[-1]):
            raise ValueError('z value exceeds bounds of interpolation grid ')
        xinp = SP.array(SP.where(SP.isfinite(x)))
        yinp = SP.array(SP.where(SP.isfinite(y)))
        zinp = SP.array(SP.where(SP.isfinite(z)))
        inp = SP.intersect1d(SP.intersect1d(xinp,yinp),zinp)

        if inp.size != 0:
            ix = SP.digitize(x[inp],self._x) - 1
            iy = SP.digitize(y[inp],self._y) - 1
            iz = SP.digitize(z[inp],self._z) - 1
            pos = ix + self._f.shape[1]*(iy + self._f.shape[2]*iz)
            indx = SP.argsort(pos) #each voxel is described uniquely, and this is passed to speed evaluation.
            dx =  (x[inp]-self._x[ix])/(self._x[ix+1]-self._x[ix])
            dy =  (y[inp]-self._y[iy])/(self._y[iy+1]-self._y[iy])
            dz =  (z[inp]-self._z[iz])/(self._z[iz+1]-self._z[iz])
            val[inp] = _tricubic.ev(dx,dy,dz,self._f,pos,indx)  

 
        return(val)
