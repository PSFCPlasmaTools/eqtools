#inherited class of Equilibrium to handle eqdsk files

from EqTools import *
import scipy
import os
import glob

class EQDSKReader(Equilibrium):
    """
    Inherits Equilibrium class.  EQDSK-specific data handling class using standard
    EFIT tag names.  Pulls EFIT data from a- and g-files associated with the given shot
    and time window, stores as object attributes.  Each EFIT variable or set of variables
    is recovered with a corresponding getter method.
    """
    def __init__(self,shot,time,filename=None,length_unit='m'):
        """
        Initializes EQDSKReader object.  Pulls data from g- and a-files for given
        shot, time slice.  By default, attempts to parse shot, time inputs into file
        name, and searches directory for appropriate files.  Optionally, the user may
        instead directly input a file path for a-file, g-file.

        INPUTS:
        shot:       shot index
        time:       time slice in ms
        filename:   (optional, default None) if set, ignores shot,time inputs
        """
        #instantiate superclass, forcing time splining to false (eqdsk only contains single time slice)
        super(EQDSKReader,self).__init__(length_unit=length_unit,tspline=False)

        #if filename input is not set, attempt to parse shot, time inputs into file path
        if filename is None:
            self._shot = shot

            #parse time slice (ms) to five digits to conform with filename conventions
            if len(str(time)) < 5:
                timestring = '0'*(5-len(str(time))) + str(time)
            elif len(str(time)) == 5:
                timestring = str(time)
            
            name = str(shot)+'.'+timestring

            #check current directory for filenames containing putative eqdsk name
            currfiles = glob.glob('*'+name+'*')
