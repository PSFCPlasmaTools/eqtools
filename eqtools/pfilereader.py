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

"""
This module contains the :py:class:`PFileReader` class, a lightweight data
handler for p-file (radial profile) datasets.

Classes:
    PFileReader: 
        Data-storage class for p-file data.  Reads 
        data from ASCII p-file, storing as copy-safe object 
        attributes.
"""

import numpy as np
import csv
import re
from collections import namedtuple


class PFileReader(object):
    """Class to read ASCII p-file (profile data storage) into lightweight, 
    user-friendly data structure.  

    P-files store data blocks containing the following: a header with parameter
    name, parameter units, x-axis units, and number of data points, followed by 
    values of axis x, parameter y, and derivative dy/dx.  Each parameter block 
    is read into a namedtuple storing 

    ======== ==============
    'name'   parameter name
    'npts'   array size
    'x'      abscissa array
    'y'      data array
    'dydx'   data gradient
    'xunits' abscissa units
    'units'  data units
    ======== ==============

    with each namedtuple stored as an attribute of the PFileReader  instance.  
    This gracefully handles variable formats of p-files (differing versions of 
    p-files will have different parameters stored).  Data blocks are accessed 
    as attributes in a copy-safe manner.
    
    Creates instance of PFileReader.

    Args:
        pfile (String): Path to ASCII p-file to be loaded.

    Keyword Args:
        verbose (Boolean): Option to print message on object creation 
            listing available data parameters. Defaults to True.

    Examples:
        Load p-file data located at `file_path`, while suppressing terminal
        output of stored parameters::

            pfr = eqtools.PFileReader(file_path,verbose=False)

        Recover electron density data (for example)::

            ne_data = pfr.ne

        Recover abscissa and electron density data (for example)::

            ne = pfr.ne.y
            abscis = pfr.ne.x

        Available parameters in pfr may be listed via the overridden __str__
        command.
    """
    def __init__(self,pfile,verbose=True):
        self._pfile = pfile
        self._params = []

        with open(pfile,'r') as readfile:
            dia = csv.excel()
            dia.skipinitialspace = True
            reader = csv.reader(readfile,dia,delimiter=' ')

            # define data structure as named tuple for storing parameter values
            data = namedtuple('DataStruct',
                             ['name','npts','units','xunits','x','y','dydx'])

            # iterate through lines of file, checking for a header line; 
            # at each header, read the next npts lines of data into 
            # appropriate arrays.
            # continue until no headerline is found (throws StopIteration).  
            # Populate list of params with available variables.
            while True:
                try:
                    headerline = next(reader)
                except StopIteration:
                    break

                npts = int(headerline[0])               # size of abscissa, data arrays
                abscis = headerline[1]                  # string name of abscissa variable (e.g. 'psinorm')
                var = re.split('[\(\)]',headerline[2])
                param = var[0]                          # string name of parameter (e.g. 'ne')
                units = var[1]                          # string name of units (e.g. '10^20/m^3')

                # read npts next lines, populate arrays
                x = []
                val = []
                gradval = []
                for j in range(npts):
                    dataline = next(reader)
                    x.append(float(dataline[0]))
                    val.append(float(dataline[1]))
                    gradval.append(float(dataline[2]))
                x = np.array(x)
                val = np.array(val)
                gradval = np.array(gradval)

                # collate into storage structure
                vars(self)['_'+param] = data(name=param,
                                             npts=npts,
                                             units=units,
                                             xunits=abscis,
                                             x=x,
                                             y=val,
                                             dydx=gradval)
                self._params.append(param)

        if verbose:
            print('P-file data loaded from '+self._pfile)
            print('Available parameters:')
            for par in self._params:
                un = vars(self)['_'+par].units
                xun = vars(self)['_'+par].xunits
                print(str(par).ljust(8)+str(xun).ljust(12)+str(un))

    def __str__(self):
        """overrides default string method for useful output.
        """
        mes = 'P-file data from '+self._pfile+' containing parameters:\n'
        for par in self._params:
            un = vars(self)['_'+par].units
            xun = vars(self)['_'+par].xunits
            mes += str(par).ljust(8)+str(xun).ljust(12)+str(un)+'\n'
        return mes

    def __getattribute__(self, name):
        """Copy-safe attribute retrieval method overriding default 
        object.__getattribute__.

        Tries to retrieve attribute as-written (first check for default object 
        attributes).  If that fails, looks for pseudo-private attributes, marked
        by preceding underscore, to retrieve data blocks.  If this fails, 
        raise AttributeError.

        Args:
            name (String): Name (without leading underscore for data variables)
            of attribute.

        Raises:
            AttributeError: if no attribute can be found.
        """
        try:
            return super(PFileReader,self).__getattribute__(name)
        except AttributeError:
            try:
                attr = super(PFileReader,self).__getattribute__('_'+name)
                if type(attr) is list:
                    return attr[:]
                else:
                    return attr
            except AttributeError:
                raise AttributeError('No attribute "%s" found' % name)

    def __setattr__(self, name, value):
        """Copy-safe attribute setting method overriding default 
        `object.__setattr__`.

        Raises error if object already has attribute `_{name}` for input name,
        as such an attribute would interfere with automatic property generation 
        in :py:meth:`__getattribute__`.

        Args:
            name (String): Attribute name.

        Raises:
            AttributeError: if attempting to create attribute with protected
                pseudo-private name.
        """
        if hasattr(self, '_'+name):
            raise AttributeError("PFileReader object already has data attribute"
                                 " '_%(n)s', creating attribute '%(n)s' will"
                                 " conflict with automatic property generation."
                                 % {'n': name})
        else:
            super(PFileReader, self).__setattr__(name, value)