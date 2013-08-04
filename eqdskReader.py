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
    def __init__(self,shot,time,gfilename=None,afilename=None,length_unit='m'):
        """
        Initializes EQDSKReader object.  Pulls data from g- and a-files for given
        shot, time slice.  By default, attempts to parse shot, time inputs into file
        name, and searches directory for appropriate files.  Optionally, the user may
        instead directly input a file path for a-file, g-file.

        INPUTS:
        shot:       shot index
        time:       time slice in ms
        gfilename:  (optional, default None) if set, ignores shot,time inputs and pulls g-file by name
        afilename:  (optional, default None) if set, ignores shot,time inputs and pulls a-file by name
        """
        #instantiate superclass, forcing time splining to false (eqdsk only contains single time slice)
        super(EQDSKReader,self).__init__(length_unit=length_unit,tspline=False)

        #parse shot and time inputs into standard naming convention
        if len(str(time)) < 5:
            timestring = '0'*(5-len(str(time))) + str(time)
        elif len(str(time)) > 5:
            timestring = str(time)[-5:]
            print('Time window string greater than 5 digits.  Masking to last 5 digits.  If this violates the selected EQ files, \
                  please use explicit filename inputs.')
        else:   #exactly five digits
            timestring = str(time)

        name = str(shot)+'.'+timestring

        #if explicit filename for g-file is not set, check current directory for files matching name
        if gfilename is None:
            gcurrfiles = glob.glob('g'+name+'*')
            if len(gcurrfiles) == 1:
                gfilename = gcurrfiles[0]
            elif len(gcurrfiles) > 1:
                raise ValueError('multiple valid g-files detected in directory.  Please select a file with explicit \
                                  input or clean directory.')
            else:   #no files found
                raise ValueError('no valid g-files detected in directory.  Please select a file with explicit input or \
                                  ensure file is in directory.')

        #and likewise for a-file name
        if afilename is None:
            acurrfiles = glob.glob('a'+name+'*')
            if len(acurrfiles) == 1:
                afilename = acurrfiles[0]
            elif len(acurrfiles) > 1:
                raise ValueError('multiple valid a-files detected in directory.  Please select a file with explicit \
                                  input or clean directory.')
            else:   #no files found
                raise ValueError('no valid a-files detected in directory.  Please select a file with explicit input or \
                                  ensure file in in directory.')
        






