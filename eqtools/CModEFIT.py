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

import scipy

from .EFIT import EFITTree
from .core import PropertyAccessMixin, ModuleWarning

import warnings

try:
    import MDSplus
    from MDSplus._treeshr import TreeException
    _has_MDS = True
except Exception as _e_MDS:
    if isinstance(_e_MDS, ImportError):
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work.",
                      ModuleWarning)
    else:
        warnings.warn("MDSplus module could not be loaded -- classes that use "
                      "MDSplus for data access will not work. Exception raised "
                      "was of type %s, message was '%s'."
                      % (_e_MDS.__class__, _e_MDS.message),
                      ModuleWarning)
    _has_MDS = False

class CModEFITTree(EFITTree):
    """
    Inherits EFITTree class. machine-specific data handling class for Alcator C-Mod.
    Pulls EFIT data from selected MDS tree and shot, stores as object attributes.
    Each EFIT variable or set of variables is recovered with a corresponding getter method.
    Essential data for EFIT mapping are pulled on initialization (e.g. psirz grid).
    Additional data are pulled at the first request and stored for subsequent usage.
    """

    def __init__(self, shot, tree='ANALYSIS', length_unit='m', tspline=False, fast=False):
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

        super(CModEFITTree, self).__init__(shot, tree, root, length_unit=length_unit, tspline=tspline, fast=fast)
    
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
