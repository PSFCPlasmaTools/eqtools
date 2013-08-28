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
This module contains the PFileReader class, a lightweight data
handler for p-file (radial profile) datasets.

Classes:
    PFileReader: Data-storage class for p-file data.  Reads 
    data from ASCII p-file, storing as copy-safe object attributes.
"""

import numpy as np
import csv
import re
from collections import namedtuple


class PFileReader(object):
    """
    Class to read ASCII p-file (profile data storage) into lightweight, user-friendly data structure.  

    P-files store data blocks containing the following: a header with parameter
    name, parameter units, x-axis units, and number of data points, followed by values of
    axis x, parameter y, and derivative dy/dx.  Each parameter block is read into a
    namedtuple storing ['name','npts','units','xunits','x','y','dydx'], with each namedtuple
    stored as an attribute of the PFileReader instance.  This gracefully handles variable
    formats of p-files (differing versions of p-files will have different parameters stored).
    Data blocks are accessed as attributes in a copy-safe manner.
    """
    def __init__(self,pfile,verbose=True):
        """
        initialize data-storage object and read data from supplied p-file.

        INPUTS:
        pfile:      (str) path to p-file
        verbose:    (bool, def True) print available parameters on load
        """
        """
        Creates 
        """
        self._pfile = pfile
        self._params = []

        with open(pfile,'r') as readfile:
            dia = csv.excel()
            dia.skipinitialspace = True
            reader = csv.reader(readfile,dia,delimiter=' ')

            # define data structure as named tuple for storing parameter values
            data = namedtuple('DataStruct',['name','npts','units','xunits','x','y','dydx'])

            # iterate through lines of file, checking for a header line; 
            # at each header, read the next npts lines of data into appropriate arrays.
            # continue until no headerline is found (throws StopIteration).  Populate list
            # of params with available variables.
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
                vars(self)['_'+param] = data(name=param,npts=npts,units=units,xunits=abscis,x=x,y=val,dydx=gradval)
                self._params.append(param)

        print('P-file data loaded.')
        if verbose:
            print('Available parameters:')
            for par in self._params:
                un = vars(self)['_'+par].units
                xun = vars(self)['_'+par].xunits
                print(str(par).ljust(8)+str(xun).ljust(12)+str(un))

    def __str__(self):
        mes = 'P-file data from '+self._pfile+' containing parameters:\n'
        for par in self._params:
            un = vars(self)['_'+par].units
            xun = vars(self)['_'+par].xunits
            mes += str(par).ljust(8)+str(xun).ljust(12)+str(un)+'\n'
        return mes

    def __getattribute__(self, name):
        """
        Tries to get attribute as written.  If this fails, trys to call the attribute
        with preceding underscore, marking a pseudo-private variable.  If this exists,
        returns a copy-safe value.  If this fails, raises AttributeError.  Generates a
        copy-safe version of each data attribute.
        """
        try:
            return super(PFileReader,self).__getattribute__(name)
        except AttributeError:
            try:
                attr = super(PFileReader,self).__getattribute__('_'+name)
                if type(attr) is namedtuple:
                    return attr.copy()
                elif type(attr) is list:
                    return attr[:]
                else:
                    return attr
            except AttributeError:
                raise AttributeError('No attribute "%s" found' % name)

    def __setattr__(self, name, value):
        """
        Raises AttributeError if the object already has a method get[name], as
        creation of such an attribute would interfere with the automatic
        property generation in __getattribute__.
        """
        if hasattr(self, '_'+name):
            raise AttributeError("PFileReader object already has data attribute "
                                 "'_%(n)s', creating attribute '%(n)s' will"
                                 " conflict with automatic property generation."
                                 % {'n': name})
        else:
            super(PFileReader, self).__setattr__(name, value)

            

            












            
