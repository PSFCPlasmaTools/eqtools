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

"""This module contains the EqdskReader class, which creates Equilibrium class
functionality for equilibria stored in eqdsk files from EFIT(a- and g-files).

Classes:
    EqdskReader: 
        Class inheriting Equilibrium reading g- and a-files for
        equilibrium data.
"""

import scipy
import glob
import re
import csv
import warnings
from collections import namedtuple
from .core import Equilibrium, ModuleWarning, inPolygon
from .afilereader import AFileReader

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    _has_plt = True
except Exception:
    _has_plt = False
    warnings.warn("WARNING: matplotlib modules could not be loaded -- plotting "
                  "will not be available.",
                  ModuleWarning)


class EqdskReader(Equilibrium):
    """Equilibrium subclass working from eqdsk ASCII-file equilibria.

    Inherits mapping and structural data from Equilibrium, populates equilibrium
    and profile data from g- and a-files for a selected shot and time window.
    
    Create instance of EqdskReader.

    Generates object and reads data from selected g-file (either manually set or
    autodetected based on user shot and time selection), storing as object
    attributes for usage in Equilibrium mapping methods.

    Calling structure - user may call class with shot and time (ms) values, set 
    by keywords (or positional placement allows calling without explicit keyword
    syntax).  EqdskReader then attempts to construct filenames from the 
    shot/time, of the form 'g[shot].[time]' and 'a[shot].[time]'.  Alternately, 
    the user may skip this input and explicitly set paths to the g- and/or 
    a-files, using the gfile and afile keyword arguments.  If both types of 
    calls are set, the explicit g-file and a-file paths override the 
    auto-generated filenames from the shot and time.

    Keyword Args:
        shot (Integer): Shot index.
        time (Integer): Time index (typically ms).  Shot and Time used to 
            autogenerate filenames.
        gfile (String): Manually selects ASCII file for equilibrium read.
        afile (String): Manually selects ASCII file for time-history read.
        length_unit (String): Flag setting length unit for equilibrium scales.
            Defaults to 'm' for lengths in meters.
        verbose (Boolean): When set to False, suppresses terminal outputs during
            CSV read.  Defaults to True (prints terminal output).

    Raises:
        IOError: if both name/shot and explicit filenames are not set.
        ValueError: if the g-file cannot be found, or if multiple valid 
            g/a-files are found.

    Examples:
        Instantiate EqdskReader for a given `shot` and `time` -- will search current
        working directory for files of the form g[shot].[time] and 
        a[shot].[time], suppressing terminal outputs::

            edr = eqtools.EqdskReader(shot,time,verbose=False)

        or::

            edr = eqtools.EqdskReader(shot=shot,time=time,verbose=False)

        Instantiate EqdskReader with explicit file paths `gfile_path` and 
        `afile_path`::

            edr = eqtools.EqdskReader(gfile=gfile_path,afile=afile_path)
    """
    def __init__(self,
                 shot=None,
                 time=None,
                 gfile=None,
                 afile=None,
                 length_unit='m',
                 verbose=True):
        # instantiate superclass, forcing time splining to false 
        # (eqdsk only contains single time slice)
        super(EqdskReader,self).__init__(length_unit=length_unit,tspline=False,monotonic=False)
        self._verbose = bool(verbose)

        # dict to store default units of length-scale parameters, 
        # used by core._getLengthConversionFactor
        self._defaultUnits = {}

        # parse shot and time inputs into standard naming convention
        if shot is not None and time is not None:
            if len(str(time)) < 5:
                timestring = '0'*(5-len(str(time))) + str(time)
            elif len(str(time)) > 5:
                timestring = str(time)[-5:]
                warnings.warn("Time window string greater than 5 digits.  "
                              "Masking to last 5 digits. "
                              "If this does not match the selected EQ files, "
                              "please use explicit filename inputs.",
                              RuntimeWarning)
            else:   #exactly five digits
                timestring = str(time)
            name = str(shot)+'.'+timestring
        else:
            name = None

        if name is None and gfile is None:
            raise IOError('must specify shot/time or filenames.')

        # if explicit filename for g-file is not set, check current directory 
        # for files matching name
        # if multiple valid files or no files are found, trigger ValueError
        if gfile is None:   #attempt to generate filename
            if verbose:
                print('Searching directory for file g%s.' % name)
            gcurrfiles = glob.glob('g'+name+'*')
            if len(gcurrfiles) == 1:
                self._gfilename = gcurrfiles[0]
                if verbose:
                    print('File found: '+self._gfilename)
            elif len(gcurrfiles) > 1:
                raise ValueError("Multiple valid g-files detected in directory."
                                 " Please select a file with explicit"
                                 " input or clean directory.")
            else:   # no files found
                raise ValueError("No valid g-files detected in directory. "
                                  "Please select a file with explicit input or "
                                  "ensure file is in directory.")
        else:   # check that given file is in directory
            gcurrfiles = glob.glob(gfile)
            if len(gcurrfiles) < 1:
                raise ValueError("No g-file with the given name detected in "
                                 "directory.  Please ensure the file is in the "
                                 "active directory or that you have supplied "
                                 "the correct name.")
            else:
                self._gfilename = gfile

        # and likewise for a-file name.  However, we can operate at reduced 
        # capacity without the a-file.  If no file with explicitly-input name 
        # is found, or multiple valid files (with no explicit input) are found, 
        # raise ValueError.  Otherwise (no autogenerated files found) set 
        # hasafile flag false and nonfatally warn user.
        if afile is None:
            if name is not None:
                if verbose:
                    print('Searching directory for file a%s.' % name)
                acurrfiles = glob.glob('a'+name+'*')
                if len(acurrfiles) == 1:
                    self._afilename = acurrfiles[0]
                    if verbose:
                        print('File found: '+self._afilename)
                elif len(acurrfiles) > 1:
                    raise ValueError("Multiple valid a-files detected in "
                                  "directory.  Please select a file with "
                                  "explicit input or clean directory.")
                else:   # no files found
                    warnings.warn("No valid a-files detected in directory. "
                                  "Please select a file with explicit input or "
                                  "ensure file in in directory.",
                                  RuntimeWarning)
                    self._afilename = None
            else:   # name and afile both are not specified
                self._afilename = None
        else:   # check that given file is in directory
            acurrfiles = glob.glob(afile)
            if len(acurrfiles) < 1:
                raise ValueError("No a-file with the given name detected in "
                                 "directory.  Please ensure the file is in the "
                                 "active directory or that you have supplied "
                                 "the correct name.")
            else:
                self._afilename = afile

        # now we start reading the g-file
        with open(self._gfilename,'r') as gfile:
            reader = csv.reader(gfile)  # skip the CSV delimiter, let split or regexs handle parsing.
                                        # use csv package for error handling.
            # read the header line, containing grid size, mfit size, and type data
            line = next(reader)[0].split()
            try:
                self._date = line[1]                            # (str) date of g-file generation, MM/DD/YYYY
            except ValueError:
                self._date = None

            try:
                self._shot = int(re.split('\D',line[-5])[-1])   # (int) shot index
            except ValueError:
                self._shot = None

            try:
                timestring = line[-4]                       # (str) time index, with units (e.g. '875ms')
                print(timestring)
            except ValueError:
                timestring = None

            #imfit = int(line[-3])                           # not sure what this is supposed to be...
            nw = int(line[-2])                              # width of flux grid (dim(R))
            nh = int(line[-1])                              # height of flux grid (dim(Z))

            print(nw,nh)
            #extract time, units from timestring
            try:
                time = re.findall('\d+',timestring)[0]
                tunits = timestring.split(time)[1]
                timeConvertDict = {'ms':1./1000.,'s':1.}
                self._time = scipy.array([float(time)*timeConvertDict[tunits]]) # returns time in seconds as array
    
            except KeyError:
                tunits = None
                self._time = None
            except IndexError:
                tunits = None
                self._time = None

            self._defaultUnits['_time'] = 's'
            
            # next line - construction values for RZ grid
            line = next(reader)[0]
            line = re.findall('-?\d\.\d*[eE][-+]\d*',line)     # regex magic!
            xdim = float(line[0])     # width of R-axis in grid
            zdim = float(line[1])     # height of Z-axis in grid
            self._RCentr = scipy.array(float(line[2]))    # rcentr for Bcentr
            self._defaultUnits['_RCentr'] = 'm'
            rgrid0 = float(line[3])   # start point of R grid
            zmid = float(line[4])     # midpoint of Z grid

            # construct EFIT grid
            self._rGrid = scipy.linspace(rgrid0,rgrid0 + xdim,nw)
            self._zGrid = scipy.linspace(zmid - zdim/2.0,zmid + zdim/2.0,nh)
            #drefit = (self._rGrid[-1] - self._rGrid[0])/(nw-1)
            #dzefit = (self._zGrid[-1] - self._zGrid[0])/(nh-1)
            self._defaultUnits['_rGrid'] = 'm'
            self._defaultUnits['_zGrid'] = 'm'

            # read R,Z of magnetic axis, psi at magnetic axis and LCFS, and bzero
            line = next(reader)[0]
            line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
            self._rmag = scipy.array([float(line[0])])
            self._zmag = scipy.array([float(line[1])])
            self._defaultUnits['_rmag'] = 'm'
            self._defaultUnits['_zmag'] = 'm'
            self._psiAxis = scipy.array([float(line[2])])
            self._psiLCFS = scipy.array([float(line[3])])
            self._BCentr = scipy.array([float(line[4])])
            self._defaultUnits['_psiAxis'] = 'Wb/rad'
            self._defaultUnits['_psiLCFS'] = 'Wb/rad'

            # read EFIT-calculated plasma current, psi at magnetic axis (duplicate), 
            # dummy, R of magnetic axis (duplicate), dummy
            line = next(reader)[0]
            line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
            self._IpCalc = scipy.array([float(line[0])])
            self._defaultUnits['_IpCalc'] = 'A'

            # read Z of magnetic axis (duplicate), dummy, psi at LCFS (duplicate), dummy, dummy
            line = next(reader)[0]
            # don't actually need anything from this line

            # start reading fpol, next nw inputs
            nrows = nw/5
            if nw % 5 != 0:     # catch truncated rows
                nrows += 1

            self._fpol = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    self._fpol.append(float(val))
            self._fpol = scipy.array(self._fpol).reshape((1,nw))
            self._defaultUnits['_fpol'] = 'T m'

            # and likewise for pressure
            self._fluxPres = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    self._fluxPres.append(float(val))
            self._fluxPres = scipy.array(self._fluxPres).reshape((1,nw))
            self._defaultUnits['_fluxPres'] = 'Pa'

            # geqdsk written as negative for positive plasma current
            # ffprim, pprime input with correct EFIT sign
            self._ffprim = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    self._ffprim.append(float(val))
            self._ffprim = scipy.array(self._ffprim).reshape((1,nw))
            self._defaultUnits['_ffprim'] = 'T^2 m'

            self._pprime = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    self._pprime.append(float(val))
            self._pprime = scipy.array(self._pprime).reshape((1,nw))
            self._defaultUnits['_pprime'] = 'J/m^2'

            # read the 2d [nw,nh] array for psiRZ
            # start by reading nw x nh points into 1D array,
            # then repack in column order into final array
            npts = nw*nh
            nrows = npts/5
            if npts % 5 != 0:
                nrows += 1

            psis = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    psis.append(float(val))
            self._psiRZ = scipy.array(psis).reshape((1,nh,nw),order='C')
            self._defaultUnits['_psiRZ'] = 'Wb/rad'

            # read q(psi) profile, nw points (same basis as fpol, pres, etc.)
            nrows = nw/5
            if nw % 5 != 0:
                nrows += 1

            self._qpsi = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    self._qpsi.append(float(val))
            self._qpsi = scipy.array(self._qpsi).reshape((1,nw))

            # read nbbbs, limitr
            line = next(reader)[0].split()
            nbbbs = int(line[0])
            limitr = int(line[1])

            # next data reads as 2 x nbbbs array, then broken into
            # rbbbs, zbbbs (R,Z locations of LCFS)
            npts = 2*nbbbs
            nrows = npts/5
            if npts % 5 != 0:
                nrows += 1
            bbbs = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    bbbs.append(float(val))
            bbbs = scipy.array(bbbs).reshape((2,nbbbs),order='F')
            self._RLCFS = bbbs[0].reshape((1,nbbbs))
            self._ZLCFS = bbbs[1].reshape((1,nbbbs))
            self._defaultUnits['_RLCFS'] = 'm'
            self._defaultUnits['_ZLCFS'] = 'm'

            # next data reads as 2 x limitr array, then broken into
            # xlim, ylim (locations of limiter)(?)
            npts = 2*limitr
            nrows = npts/5
            if npts % 5 != 0:
                nrows += 1
            lim = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d\.\d*[eE][-+]\d*',line)
                for val in line:
                    lim.append(float(val))
            lim = scipy.array(lim).reshape((2,limitr),order='F')
            self._xlim = lim[0,:]
            self._ylim = lim[1,:]

            # this is the extent of the original g-file read.
            # attempt to continue read for newer g-files; exception
            # handler sets relevant parameters to None for older g-files
            try:
                # read kvtor, rvtor, nmass
                line = next(reader)[0].split()
                kvtor = int(line[0])
                #rvtor = float(line[1])
                nmass = int(line[2])

                # read kvtor data if present
                if kvtor > 0:
                    nrows = nw/5
                    if nw % 5 != 0:
                        nrows += 1
                    self._presw = []
                    for i in range(nrows):
                        line = next(reader)[0]
                        line = re.findall('-?\d.\d*[eE][-+]\d*',line)
                        for val in line:
                            self._presw.append(float(val))
                    self._presw = scipy.array(self._presw).reshape((nw,1))
                    self._preswp = []
                    for i in range(nrows):
                        line = next(reader)[0]
                        line = re.findall('-?\d.\d*[eE][-+]\d*',line)
                        for val in line:
                            self._preswp.append(float(val))
                    self._preswp = scipy.array(self._preswp).reshape((1,nw))
                else:
                    self._presw = scipy.atleast_2d(scipy.array([0]))
                    self._preswp = scipy.atleast_2d(scipy.array([0]))

                # read ion mass density if present
                if nmass > 0:
                    nrows = nw/5
                    if nw % 5 != 0:
                        nrows += 1
                    self._dmion = []
                    for i in range(nrows):
                        line = next(reader)[0]
                        line = re.findall('-?\d.\d*[eE][-+]\d*',line)
                        for val in line:
                            self._dmion.append(float(val))
                    self._dmion = scipy.array(self._dmion).reshape((1,nw))
                else:
                    self._dmion = scipy.atleast_2d(scipy.array([0]))

                # read rhovn
                nrows = nw/5
                if nw % 5 != 0:
                    nrows += 1
                self._rhovn = []
                for i in range(nrows):
                    line = next(reader)[0]
                    line = re.findall('-?\d.\d*[eE][-+]\d*',line)
                    for val in line:
                        self._rhovn.append(float(val))
                self._rhovn = scipy.array(self._rhovn).reshape((1,nw))

                # read keecur; if >0 read workk
                line = gfile.readline.split()
                keecur = int(line[0])
                if keecur > 0:
                    self._workk = []
                    for i in range(nrows):
                        line = next(reader)[0]
                        line = re.findall('-?\d.\d*[eE][-+]\d*',line)
                        for val in line:
                            self._workk.append(float(val))
                    self._workk = scipy.array(self._workk).reshape((1,nw))
                else:
                    self._workk = scipy.atleast_2d(scipy.array([0]))
            except:
                self._presw = scipy.atleast_2d(scipy.array([0]))
                self._preswp = scipy.atleast_2d(scipy.array([0]))
                self._rhovn = scipy.atleast_2d(scipy.array([0]))
                self._dmion = scipy.atleast_2d(scipy.array([0]))
                self._workk = scipy.atleast_2d(scipy.array([0]))

            # read through to end of file to get footer line
            try:
                r = ''
                for row in reader:
                    r = row[0]
                self._efittype = r.split()[-1]
            except:
                self._efittype = None
            

        # toroidal current density on (r,z,t) grid typically not
        # written to g-files.  Override getter method and initialize
        # to none.
        self._Jp = None
        # initialize current direction, used by mapping routines.
        self._currentSign = None

        # initialize data stored in a-file
        # fields
        self._btaxp = None
        self._btaxv = None
        self._bpolav = None
        self._defaultUnits['_btaxp'] = 'T'
        self._defaultUnits['_btaxv'] = 'T'
        self._defaultUnits['_bpolav'] = 'T'

        # currents
        self._IpMeas = None
        self._defaultUnits['_IpMeas'] = 'A'

        # safety factor parameters
        self._q0 = None
        self._q95 = None
        self._qLCFS = None
        self._rq1 = None
        self._rq2 = None
        self._rq3 = None
        self._defaultUnits['_rq1'] = 'cm'
        self._defaultUnits['_rq2'] = 'cm'
        self._defaultUnits['_rq3'] = 'cm'

        # shaping parameters
        self._kappa = None
        self._dupper = None
        self._dlower = None

        # dimensional geometry parameters
        self._aLCFS = None
        self._areaLCFS = None
        self._RmidLCFS = None
        self._defaultUnits['_aLCFS'] = 'cm'
        self._defaultUnits['_areaLCFS'] = 'cm^2'
        self._defaultUnits['_RmidLCFS'] = 'm'

        # calc. normalized pressure values
        self._betat = None
        self._betap = None
        self._Li = None

        # diamagnetic measurements
        self._diamag = None
        self._betatd = None
        self._betapd = None
        self._WDiamag = None
        self._tauDiamag = None
        self._defaultUnits['_diamag'] = 'Vs'
        self._defaultUnits['_WDiamag'] = 'J'
        self._defaultUnits['_tauDiamag'] = 'ms'

        # calculated energy
        self._WMHD = None
        self._tauMHD = None
        self._Pinj = None
        self._Wbdot = None
        self._Wpdot = None
        self._defaultUnits['_WMHD'] = 'J'
        self._defaultUnits['_tauMHD'] = 'ms'
        self._defaultUnits['_Pinj'] = 'W'
        self._defaultUnits['_Wbdot'] = 'W'
        self._defaultUnits['_Wpdot'] = 'W'

        # fitting parameters
        self._volLCFS = None
        self._fluxVol = None
        self._RmidPsi = None
        self._defaultUnits['_volLCFS'] = 'cm^3'
        self._defaultUnits['_fluxVol'] = 'm^3'
        self._defaultUnits['_RmidPsi'] = 'm'

        # attempt to populate these parameters from a-file
        if self._afilename is not None:
            try:
                self.readAFile(self._afilename)
            except IOError:
                if verbose:
                    print('a-file data not loaded.')
        else:
            if verbose:
                print('a-file data not loaded.')
                    
    def __str__(self):
        """Overrides default __str__ method with more useful information.
        """
        if self._efittype is None:
            eq = 'equilibrium'
        else:
            eq = self._efittype+' equilibrium'
        return 'G-file '+eq+' from '+str(self._gfilename)
        
    def getInfo(self):
        """returns namedtuple of equilibrium information
        
        Returns:
            namedtuple containing
                
                ========   ==============================================
                shot       shot index
                time       time point of g-file
                nr         size of R-axis of spatial grid
                nz         size of Z-axis of spatial grid
                efittype   EFIT calculation type (magnetic, kinetic, MSE)
                ========   ==============================================
        """
        data = namedtuple('Info',['shot','time','nr','nz','efittype'])
        try:
            nr = len(self._rGrid)
            nz = len(self._zGrid)
            shot = self._shot
            time = self._time
            efittype = self._efittype
        except TypeError:
            nr,nz,shot,time=0
            efittype=None
            print 'failed to load data from g-file.'
        return data(shot=shot,time=time,nr=nr,nz=nz,efittype=efittype)

    def readAFile(self,afile):
        """Reads a-file (scalar time-history data) to pull additional 
        equilibrium data not found in g-file, populates remaining data 
        (initialized as None) in object.

        Args:
            afile (String): Path to ASCII a-file.

        Raises:
            IOError: If afile is not found.
        """
        try:
            afr = AFileReader(afile)

            # fields
            self._btaxp = scipy.array([afr.btaxp])
            self._btaxv = scipy.array([afr.btaxv])
            self._bpolav = scipy.array([afr.bpolav])

            # currents
            self._IpMeas = scipy.array([afr.pasmat])

            # safety factor parameters
            self._q0 = scipy.array([afr.qqmin])
            self._q95 = scipy.array([afr.qpsib])
            self._qLCFS = scipy.array([afr.qout])
            self._rq1 = scipy.array([afr.aaq1])
            self._rq2 = scipy.array([afr.aaq2])
            self._rq3 = scipy.array([afr.aaq3])

            # shaping parameters
            self._kappa = scipy.array([afr.eout])
            self._dupper = scipy.array([afr.doutu])
            self._dlower = scipy.array([afr.doutl])

            # dimensional geometry parameters
            self._aLCFS = scipy.array([afr.aout])
            self._areaLCFS = scipy.array([afr.areao])
            self._RmidLCFS = scipy.array([afr.rmidout])

            # calc. normalized pressure values
            self._betat = scipy.array([afr.betat])
            self._betap = scipy.array([afr.betap])
            self._Li = scipy.array([afr.ali])

            # diamagnetic measurements
            self._diamag = scipy.array([afr.diamag])
            self._betatd = scipy.array([afr.betatd])
            self._betapd = scipy.array([afr.betapd])
            self._WDiamag = scipy.array([afr.wplasmd])
            self._tauDiamag = scipy.array([afr.taudia])

            # calculated energy
            self._WMHD = scipy.array([afr.wplasm])
            self._tauMHD = scipy.array([afr.taumhd])
            self._Pinj = scipy.array([afr.pbinj])
            self._Wbdot = scipy.array([afr.wbdot])
            self._Wpdot = scipy.array([afr.wpdot])

            # fitting parameters
            self._volLCFS = scipy.array([afr.vout])
            self._fluxVol = None    # not written in g- or a-file; disable volnorm mapping routine
            self._RmidPsi = None    # not written in g- or a-file, not used by fitting parameters

        except IOError:
            raise IOError('no file "%s" found.' % afile)

    ####################################################
    # wrappers for mapping routines handling time call #
    ####################################################

    def rz2psi(self,R,Z,*args,**kwargs):
        """Calculates the non-normalized poloidal flux at the given (`R`, `Z`). 
        Wrapper for 
        :py:meth:`Equilibrium.rz2psi <eqtools.core.Equilibrium.rz2psi>` masking 
        out timebase dependence.

        Args:
            R (Array-like or scalar float): Values of the radial coordinate to 
                map to poloidal flux.  If `R` and `Z` are both scalar, then a 
                scalar `psi` is returned.  `R` and `Z` must have the same shape 
                unless the `make_grid` keyword is set.  If `make_grid` is True, 
                `R` must have shape (`len_R`,).
            Z (Array-like or scalar float): Values of the vertical coordinate to 
                map to poloidal flux.  If `R` and `Z` are both scalar, then a 
                scalar `psi` is returned.  `R` and `Z` must have the same shape 
                unless the `make_grid` keyword is set.  If `make_grid` is True, 
                `Z` must have shape (`len_Z`,).

        All keyword arguments are passed to the parent 
        :py:meth:`Equilibrium.rz2psi <eqtools.core.Equilibrium.rz2psi>`.  
        Remaining arguments in \*args are ignored.

        Returns:
            psi (Array-like or scalar float): non-normalized poloidal flux.  If 
            all input arguments are scalar, then `psi` is scalar.  IF `R` and `Z` 
            have the same shape, then `psi` has this shape as well.  If `make_grid` 
            is True, then `psi` has the shape (`len_R`, `len_Z`). 

        Examples:
            All assume that Eq_instance is a valid instance EqdskReader:

            Find single psi value at R=0.6m, Z=0.0m::
            
                psi_val = Eq_instance.rz2psi(0.6, 0)

            Find psi values at (R, Z) points (0.6m, 0m) and (0.8m, 0m).
            Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                psi_arr = Eq_instance.rz2psi([0.6, 0.8], [0, 0])

            Find psi values on grid defined by 1D vector of radial positions
            R and 1D vector of vertical positions Z::
            
                psi_mat = Eq_instance.rz2psi(R, Z, make_grid=True)
        """
        t = self.getTimeBase()
        return super(EqdskReader,self).rz2psi(R,Z,t,**kwargs)

    def rz2psinorm(self,R,Z,*args,**kwargs):
        r"""Calculates the normalized poloidal flux at the given (R,Z).
        Wrapper for 
        :py:meth:`Equilibrium.rz2psinorm <eqtools.core.Equilibrium.rz2psinorm>` 
        masking out timebase dependence.

        Uses the definition:

        .. math::

            \texttt{psi\_norm} = \frac{\psi - \psi(0)}{\psi(a) - \psi(0)}

        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to normalized poloidal flux.  Must have the same shape as 
                `Z` unless the `make_grid` keyword is set. If the `make_grid`
                keyword is True, `R` must have shape (`len_R`,).
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to normalized poloidal flux.  Must have the same shape as 
                `R` unless the `make_grid` keyword is set. If the `make_grid`
                keyword is True, `Z` must have shape (`len_Z`,).

        All keyword arguments are passed to the parent 
        :py:meth:`Equilibrium.rz2psinorm <eqtools.core.Equilibrium.rz2psinorm>`.  
        Remaining arguments in \*args are ignored.

        Returns:
            psinorm (Array-like or scalar float): non-normalized poloidal flux.  If 
            all input arguments are scalar, then `psinorm` is scalar.  IF `R` and `Z` 
            have the same shape, then `psinorm` has this shape as well.  If `make_grid` 
            is True, then `psinorm` has the shape (`len_R`, `len_Z`). 

        Examples:
            All assume that Eq_instance is a valid instance of EqdskReader:

            Find single psinorm value at R=0.6m, Z=0.0m::
            
                psi_val = Eq_instance.rz2psinorm(0.6, 0)

            Find psinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m).
            Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                psi_arr = Eq_instance.rz2psinorm([0.6, 0.8], [0, 0])

            Find psinorm values on grid defined by 1D vector of radial positions
            R and 1D vector of vertical positions Z::
            
                psi_mat = Eq_instance.rz2psinorm(R, Z, make_grid=True)
        """
        t = self.getTimeBase()[0]
        return super(EqdskReader,self).rz2psinorm(R,Z,t,**kwargs)

    def rz2phinorm(self,R,Z,*args,**kwargs):
        r"""Calculates normalized toroidal flux at a given (R,Z), using

        .. math::

            \texttt{phi} &= \int q(\psi)\,d\psi\\
            \texttt{phi\_norm} &= \frac{\phi}{\phi(a)}
        
        Wrapper for 
        :py:meth:`Equilibrium.rz2phinorm <eqtools.core.Equilibrium.rz2phinorm>` 
        masking out timebase dependence.

        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to normalized toroidal flux. Must have the same shape as `Z` 
                unless the `make_grid` keyword is set. If the `make_grid` 
                keyword is True, R must have shape (`len_R`,).
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to normalized toroidal flux. Must have the same shape as `R` 
                unless the `make_grid` keyword is set. If the `make_grid` 
                keyword is True, Z must have shape (`len_Z`,).

        All keyword arguments are passed to the parent 
        :py:meth:`Equilibrium.rz2phinorm <eqtools.core.Equilibrium.rz2phinorm>`.  
        Remaining arguments in \*args are ignored.

        Returns:
            phinorm (Array-like or scalar float): non-normalized poloidal flux.  If 
            all input arguments are scalar, then `phinorm` is scalar.  IF `R` and `Z` 
            have the same shape, then `phinorm` has this shape as well.  If `make_grid` 
            is True, then `phinorm` has the shape (`len_R`, `len_Z`). 

        Examples:
            All assume that Eq_instance is a valid instance of EqdskReader.

            Find single phinorm value at R=0.6m, Z=0.0m::
            
                phi_val = Eq_instance.rz2phinorm(0.6, 0)
        
            Find phinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m).
            Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                phi_arr = Eq_instance.rz2phinorm([0.6, 0.8], [0, 0])

            Find phinorm values on grid defined by 1D vector of radial positions 
            R and 1D vector of vertical positions Z::
            
                phi_mat = Eq_instance.rz2phinorm(R, Z, make_grid=True)
        """
        t = self.getTimeBase()[0]
        return super(EqdskReader,self).rz2phinorm(R,Z,t,**kwargs)

    def rz2volnorm(self,*args,**kwargs):
        """Calculates the normalized flux surface volume.
        
        Not implemented for EqdskReader, as necessary parameter
        is not read from a/g-files.

        Raises:
            NotImplementedError: in all cases.
        """
        raise NotImplementedError('Cannot calculate volnorm from g-file equilibria.')

    def rz2rho(self,method,R,Z,t=False,sqrt=False,make_grid=False,k=3,
               length_unit=1):
        """Convert the passed (R, Z) coordinates into one of several 
        normalized coordinates.  Wrapper for 
        :py:meth:`Equilibrium.rz2rho <eqtools.core.Equilibrium.rz2rho>` masking 
        timebase dependence.
        
        Args:
            method (String): Indicates which normalized coordinates to use.
                Valid options are:
                    
                    =======     ========================
                    psinorm     Normalized poloidal flux
                    phinorm     Normalized toroidal flux
                    volnorm     Normalized volume
                    =======     ========================
                    
            R (Array-like or scalar float): Values of the radial coordinate to
                map to normalized coordinate. Must have the same shape as `Z` 
                unless the make_grid keyword is set. If the make_grid keyword
                is True, `R` must have shape (`len_R`,).
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to normalized coordinate. Must have the same shape as `R` 
                unless the make_grid keyword is set. If the make_grid keyword 
                is True, `Z` must have shape (`len_Z`,).
        
        Keyword Args:
            t (indeterminant): Provides duck typing for inclusion of t values. 
                Passed t values either as an Arg or Kwarg are neglected.
            sqrt (Boolean): Set to True to return the square root of normalized
                coordinate. Only the square root of positive values is taken.
                Negative values are replaced with zeros, consistent with Steve
                Wolfe's IDL implementation efit_rz2rho.pro. Default is False
                (return normalized coordinate itself).
            make_grid (Boolean): Set to True to pass R and Z through meshgrid
                before evaluating. If this is set to True, R and Z must each
                only have a single dimension, but can have different lengths.
                Default is False (do not form meshgrid).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            length_unit (String or 1): Length unit that R and Z are being given
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
            
        Returns:
            rho (Array-like or scalar float): If all of the input arguments are
            scalar, then a scalar is returned. Otherwise, a scipy Array
            instance is returned. If R and Z both have the same shape then
            rho has this shape as well. If the make_grid keyword was True
            then rho has shape (len(Z), len(R)).
        
        Raises:
            ValueError: If method is not one of the supported values.
        
        Examples:
            All assume that Eq_instance is a valid instance of the appropriate
            extension of the Equilibrium abstract class.

            Find single psinorm value at R=0.6m, Z=0.0m::
            
                psi_val = Eq_instance.rz2rho('psinorm', 0.6, 0)

            Find psinorm values at (R, Z) points (0.6m, 0m) and (0.8m, 0m).
            Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                psi_arr = Eq_instance.rz2rho('psinorm', [0.6, 0.8], [0, 0])

            Find psinorm values on grid defined by 1D vector of radial positions R
            and 1D vector of vertical positions Z::
            
                psi_mat = Eq_instance.rz2rho('psinorm', R, Z, make_grid=True)
        """
        t = self.getTimeBase()[0]
        if method == 'psinorm':
            kwargs = {'return_t':False,'sqrt':sqrt,'make_grid':make_grid,'length_unit':length_unit}
        else:
            kwargs = {'return_t':False,'sqrt':sqrt,'make_grid':make_grid,'rho':False,'k':k,'length_unit':length_unit}
        if method == 'volnorm':
            raise ValueError('Cannot calculate volnorm from g-file equilibria.')
        else:
            return super(EqdskReader,self).rz2rho(method,R,Z,t,**kwargs)

    def rz2rmid(self,R,Z,t=False,sqrt=False,make_grid=False,rho=False,
                k=3,length_unit=1):
        """Maps the given points to the outboard midplane major radius, R_mid.
        Wrapper for 
        :py:meth:`Equilibrium.rz2rmid <eqtools.core.Equilibrium.rz2rmid>` 
        masking timebase dependence.
        
        Based on the IDL version efit_rz2rmid.pro by Steve Wolfe.
        
        Args:
            R (Array-like or scalar float): Values of the radial coordinate to
                map to midplane radius. Must have the same shape as `Z` unless 
                the make_grid keyword is set. If the make_grid keyword is True,
                `R` must have shape (`len_R`,).
            Z (Array-like or scalar float): Values of the vertical coordinate to
                map to midplane radius. Must have the same shape as `R` unless 
                the make_grid keyword is set. If the make_grid keyword is True, 
                `Z` must have shape (`len_Z`,).
        
        Keyword Args:
            t (indeterminant): Provides duck typing for inclusion of t values. 
                Passed t values either as an Arg or Kwarg are neglected.
            sqrt (Boolean): Set to True to return the square root of midplane
                radius. Only the square root of positive values is taken.
                Negative values are replaced with zeros, consistent with Steve
                Wolfe's IDL implementation efit_rz2rho.pro. Default is False
                (return R_mid itself).
            make_grid (Boolean): Set to True to pass `R` and `Z` through 
                meshgrid before evaluating. If this is set to True, `R` and `Z` 
                must each only have a single dimension, but can have different 
                lengths.  Default is False (do not form meshgrid).
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of `R_mid`. Default is False (return major radius, 
                R_mid).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            length_unit (String or 1): Length unit that R and Z are being given
                in AND that R_mid is returned in. If a string is given, it
                must be a valid unit specifier:
                
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
                value is 1 (R and Z given in meters, R_mid returned in meters).
            
        Returns:
            R_mid (Array or scalar float): If all of the input arguments are
            scalar, then a scalar is returned. Otherwise, a scipy Array
            instance is returned. If `R` and `Z` both have the same shape 
            then `R_mid` has this shape as well. If the make_grid keyword 
            was True then `R_mid` has shape (`len(Z)`, `len(R)`).
        
        Examples:
            All assume that Eq_instance is a valid instance of the appropriate
            extension of the Equilibrium abstract class.

            Find single R_mid value at R=0.6m, Z=0.0m::
            
                R_mid_val = Eq_instance.rz2rmid(0.6, 0)

            Find R_mid values at (R, Z) points (0.6m, 0m) and (0.8m, 0m).
            Note that the Z vector must be fully specified,
            even if the values are all the same::
            
                R_mid_arr = Eq_instance.rz2rmid([0.6, 0.8], [0, 0])

            Find R_mid values on grid defined by 1D vector of radial positions R
            and 1D vector of vertical positions Z::
            
                R_mid_mat = Eq_instance.rz2rmid(R, Z, make_grid=True)
        """
        t = self.getTimeBase()[0]
        kwargs = {'return_t':False,'sqrt':sqrt,'make_grid':make_grid,'rho':rho,'k':k,'length_unit':length_unit}
        return super(EqdskReader,self).rz2rmid(R,Z,t,**kwargs)

    def psinorm2rmid(self,psi_norm,t=False,rho=False,k=3,length_unit=1):
        """Calculates the outboard R_mid location corresponding to the passed 
        psi_norm (normalized poloidal flux) values.
        
        Args:
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to midplane radius.

        Keyword Args:
            t (indeterminant): Provides duck typing for inclusion of t values. 
                Passed `t` values either as an Arg or Kwarg are neglected.
            rho (Boolean): Set to True to return r/a (normalized minor radius)
                instead of `R_mid`. Default is False (return major radius, 
                `R_mid`).
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            length_unit (String or 1): Length unit that `R_mid` is returned in. 
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
                
                If `length_unit` is 1 or None, meters are assumed. The default
                value is 1 (`R_mid` returned in meters).
            
        Returns:
            R_mid (Array-like or scalar float): If all of the input arguments 
            are scalar, then a scalar is returned. Otherwise, a scipy Array
            instance is returned.
        
        Examples:
            All assume that Eq_instance is a valid instance of the appropriate
            extension of the Equilibrium abstract class.

            Find single R_mid value for psinorm=0.7::
            
                R_mid_val = Eq_instance.psinorm2rmid(0.7)

            Find R_mid values at psi_norm values of 0.5 and 0.7.
            Note that the Z vector must be fully specified, even if the
            values are all the same::
            
                R_mid_arr = Eq_instance.psinorm2rmid([0.5, 0.7])
        """
        t = self.getTimeBase()[0]
        kwargs = {'return_t':False,'rho':rho,'k':k,'length_unit':length_unit}
        return super(EqdskReader,self).psinorm2rmid(psi_norm,t,**kwargs)

    def psinorm2volnorm(self,*args,**kwargs):
        """Calculates the outboard R_mid location corresponding to psi_norm 
        (normalized poloidal flux) values.
        
        Not implemented for EqdskReader, as necessary parameter is not read 
        from a/g-files.

        Raises:
            NotImplementedError: in all cases.            
        """
        raise NotImplementedError('Cannot calculate volnorm from g-file equilibria.')

    def psinorm2phinorm(self,psi_norm,t=False,k=3):
        """Calculates the normalized toroidal flux corresponding to the passed 
        psi_norm (normalized poloidal flux) values.
        
        Args:
            psi_norm (Array-like or scalar float): Values of the normalized
                poloidal flux to map to normalized toroidal flux.
        
        Keyword Args:
            t (indeterminant): Provides duck typing for inclusion of t values. 
                Passed `t` values either as an Arg or Kwarg are neglected.
            k (positive int): The degree of polynomial spline interpolation to
                use in converting coordinates.
            
        Returns:
            phinorm (Array-like or scalar float): If all of the input arguments 
            are scalar, then a scalar is returned. Otherwise, a scipy Array
            instance is returned.
        
        Examples:
            All assume that Eq_instance is a valid instance of the appropriate
            extension of the Equilibrium abstract class.

            Find single phinorm value for psinorm=0.7::
            
                phinorm_val = Eq_instance.psinorm2phinorm(0.7)

            Find phinorm values at psi_norm values of 0.5 and 0.7.
            Note that the Z vector must be fully specified, even if the
            values are all the same::
            
                phinorm_arr = Eq_instance.psinorm2phinorm([0.5, 0.7])
        """
        t = self.getTimeBase()[0]
        kwargs = {'return_t':False,'k':3}
        return super(EqdskReader,self).psinorm2phinorm(psi_norm,t,**kwargs)

    #################
    # data handlers #
    #################

    def getTimeBase(self):
        """Returns EFIT time point.

        Returns:
            time (Array): 1-element, 1D array of time in s.  Returns array for
            consistency with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        return self._time.copy()

    def getCurrentSign(self):
        """Returns the sign of the current, based on the check in Steve Wolfe's 
        IDL implementation efit_rz2psi.pro.

        Returns:
            currentSign (Int): 1 for positive current, -1 for reversed.
        """
        if self._currentSign is None:
            self._currentSign = 1 if scipy.mean(self.getIpCalc()) > 1e5 else -1
        return self._currentSign

    def getFluxGrid(self):
        """Returns EFIT flux grid.

        Returns:
            psiRZ (Array): [1,r,z] Array of flux values.  Includes 1-element
            time axis for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` implementations 
            with time variation.
        """
        return self._psiRZ.copy()

    def getRGrid(self,length_unit=1):
        """Returns EFIT R-axis.

        Returns:
            R (Array): [r] array of R-axis values for RZ grid.
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rGrid'],length_unit)
        return unit_factor * self._rGrid.copy()

    def getZGrid(self,length_unit=1):
        """Returns EFIT Z-axis.

        Returns:
            Z (Array): [z] array of Z-axis values for RZ grid.
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zGrid'],length_unit)
        return unit_factor * self._zGrid.copy()

    def getFluxAxis(self):
        """Returns psi on magnetic axis.

        Returns:
            psi0 (Array): [1] array of psi on magnetic axis.  Returns array for
            consistency with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        # scale by current sign for consistency with sign of psiRZ.
        return -1. * self.getCurrentSign() * scipy.array(self._psiAxis)

    def getFluxLCFS(self):
        """Returns psi at separatrix.

        Returns:
            psia (Array): [1] array of psi at separatrix.  Returns array for
            consistency with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        # scale by current sign for consistency with sign of psiRZ.
        return -1 * self.getCurrentSign() * scipy.array(self._psiLCFS)

    def getRLCFS(self,length_unit=1):
        """Returns array of R-values of LCFS.

        Returns:
            RLCFS (Array): [1,n] array of R values describing LCFS.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` implementations 
            with time variation.
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RLCFS'],length_unit)
        return unit_factor * self._RLCFS.copy()

    def getZLCFS(self,length_unit=1):
        """Returns array of Z-values of LCFS.

        Returns:
            ZLCFS (Array): [1,n] array of Z values describing LCFS.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` implementations 
            with time variation.
        """
        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_ZLCFS'],length_unit)
        return unit_factor * self._ZLCFS.copy()
        
    def remapLCFS(self,mask=False):
        """Overwrites RLCFS, ZLCFS values pulled from EFIT with 
        explicitly-calculated contour of psinorm=1 surface.

        Keyword Args:
            mask (Boolean): Set True to mask LCFS path to limiter outline 
                (using inPolygon).  Set False to draw full contour of 
                psi = psiLCFS.  Defaults to False.
        """
        if not _has_plt:
            raise NotImplementedError("Requires matplotlib.pyplot for contour calculation.")
            
        try:
            Rlim,Zlim = self.getMachineCrossSection()
        except:
            raise ValueError("Limiter outline in self.getMachineCrossSection must be available.")

        plt.ioff()
            
        psiRZ = self.getFluxGrid()
        R = self.getRGrid()
        Z = self.getZGrid()
        psiLCFS = -1.0 * self.getCurrentSign() * self.getFluxLCFS()
        
        fig = plt.figure()  # generate a dummy plotting window to dump contour into; will be deleted later
        cs = plt.contour(R,Z,psiRZ[0],psiLCFS)   # calculates psi= psiLCFS contour

        paths = cs.collections[0].get_paths()
        RLCFS = []
        ZLCFS = []
        for path in paths:
            v = path.vertices
            RLCFS.extend(v[:,0])
            ZLCFS.extend(v[:,1])
            RLCFS.append(scipy.nan)
            ZLCFS.append(scipy.nan)
        RLCFS = scipy.array(RLCFS)
        ZLCFS = scipy.array(ZLCFS)
        
        # generate masking array
        if mask:
            maskarr = scipy.array([False for i in range(len(RLCFS))])
            for i,x in enumerate(RLCFS):
                y = ZLCFS[i]
                maskarr[i] = inPolygon(Rlim,Zlim,x,y)
                
            RLCFS = RLCFS[maskarr]
            ZLCFS = ZLCFS[maskarr]

        npts = len(RLCFS)
        self._RLCFS = RLCFS.reshape((1,npts))
        self._ZLCFS = ZLCFS.reshape((1,npts))
        
        # cleanup
        plt.ion()
        plt.clf()
        plt.close(fig)
        plt.ioff()

    def getFluxVol(self):
        """Returns volume contained within a flux surface as a function of psi.

        Not implemented in :py:class:`EqdskReader`, as required data is not 
        stored in g/a-files.

        Raises:
            NotImplementedError: in all cases.
        """
        raise NotImplementedError()

    def getVolLCFS(self,length_unit=3):
        """Returns volume with LCFS.

        Returns:
            Vol (Array): [1] array of plasma volume.  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._volLCFS is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_volLCFS'],length_unit)
            return unit_factor * self._volLCFS.copy()

    def getRmidPsi(self):
        """Returns outboard-midplane major radius of flux surfaces.
        
        Data not read from a/g-files, not implemented for :py:class:`EqdskReader`.

        Raises:
            NotImplementedError: in all cases.
        """
        raise NotImplementedError('RmidPsi not read from a/g-files.')

    def getF(self):
         """returns F=RB_{\Phi}(\Psi), calculated for grad-shafranov solutions  
         [psi,t]

        Returns:
            F (Array): [1,n] array of F(\psi).  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """       
         return self._fpol.copy()
    
    def getFluxPres(self):
        """Returns pressure on flux surface p(psi).

        Returns:
            p (Array): [1,n] array of pressure.  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        return self._fluxPres.copy()

    def getFFPrime(self):
        """returns FF' function used for grad-shafranov solutions.

        Returns:
            FF (Array): [1,n] array of FF'(\psi).  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        return self._ffprim.copy()

    def getPPrime(self): 
        """returns plasma pressure gradient as a function of psi.

        Returns: 
            pp (Array): [1,n] array of pp'(\psi).  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        return self._pprime.copy()

    def getElongation(self):
        """Returns elongation of LCFS.

        Returns:
            kappa (Array): [1] array of plasma elongation.  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._kappa is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._kappa.copy()

    def getUpperTriangularity(self):
        """Returns upper triangularity of LCFS.

        Returns:
            delta (Array): [1] array of plasma upper triangularity.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._dupper is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._dupper.copy()

    def getLowerTriangularity(self):
        """Returns lower triangularity of LCFS.

        Returns:
            delta (Array): [1] array of plasma lower triangularity.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._dlower is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._dlower.copy()

    def getShaping(self):
        """Pulls LCFS elongation, upper/lower triangularity.
        
        Returns:
            namedtuple containing [kappa,delta_u,delta_l].

        Raises:
            ValueError: if a-file data is not read.
        """
        try:
            kap = self.getElongation()
            du = self.getUpperTriangularity()
            dl = self.getLowerTriangularity()
            data = namedtuple('Shaping',['kappa','delta_u','delta_l'])
            return data(kappa=kap,delta_u=du,delta_l=dl)
        except ValueError:
            raise ValueError('must read a-file for this data.') 

    def getMagR(self,length_unit=1):
        """Returns major radius of magnetic axis.

        Keyword Args:
            length_unit (String or 1): length unit R is specified in.  Defaults
                to 1 (default unit of rmagx, typically m).

        Returns:
            magR (Array): [1] array of major radius of magnetic axis.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._rmag is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rmag'],length_unit)
            return unit_factor * self._rmag.copy()

    def getMagZ(self,length_unit=1):
        """Returns Z of magnetic axis.

        Keyword Args:
            length_unit (String or 1): length unit Z is specified in.  Defaults
                to 1 (default unit of zmagx, typically m).

        Returns:
            magZ (Array): [1] array of Z of magnetic axis.  Returns array for
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._zmag is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_zmag'],length_unit)
            return unit_factor * self._zmag.copy()

    def getAreaLCFS(self,length_unit=2):
        """Returns surface area of LCFS.

        Keyword Args:
            length_unit (String or 2): unit area is specified in.  Defaults to 2
                (default unit, typically m^2).

        Returns:
            AreaLCFS (Array): [1] array of surface area of LCFS.  Returns array 
            for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._areaLCFS is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_areaLCFS'],length_unit)
            return unit_factor * self._areaLCFS.copy()

    def getAOut(self,length_unit=1):
        """Returns outboard-midplane minor radius of LCFS.

        Keyword Args:
            length_unit (String or 1): unit radius is specified in.  Defaults 
                to 1 (default unit, typically m).

        Returns:
            AOut (Array): [1] array of outboard-midplane minor radius at LCFS.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._aLCFS is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_aLCFS'],length_unit)
            return unit_factor * self._aLCFS.copy()

    def getRmidOut(self,length_unit=1):
        """Returns outboard-midplane major radius of LCFS.

        Keyword Args:
            length_unit (String or 1): unit radius is specified in.  Defaults to 
                1 (default unit, typically m).

        Returns:
            Rmid (Array): [1] array of outboard-midplane major radius at LCFS.  
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._RmidLCFS is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RmidLCFS'],length_unit)
            return unit_factor * self._RmidLCFS.copy()

    def getGeometry(self,length_unit=None):
        """Pulls dimensional geometry parameters.

        Keyword Args:
            length_unit (String): length unit parameters are specified in.  
                Defaults to None, using default units for individual getter 
                methods for constituent parameters.

        Returns:
            namedtuple containing [Rmag,Zmag,AreaLCFS,aOut,RmidOut]

        Raises:
            ValueError: if a-file data is not read.
        """
        try:
            Rmag = self.getMagR(length_unit=(length_unit if length_unit is not None else 1))
            Zmag = self.getMagZ(length_unit=(length_unit if length_unit is not None else 1))
            AreaLCFS = self.getAreaLCFS(length_unit=(length_unit if length_unit is not None else 2))
            aOut = self.getAOut(length_unit=(length_unit if length_unit is not None else 1))
            RmidOut = self.getRmidOut(length_unit=(length_unit if length_unit is not None else 1))
            data = namedtuple('Geometry',['Rmag','Zmag','AreaLCFS','aOut','RmidOut'])
            return data(Rmag=Rmag,Zmag=Zmag,AreaLCFS=AreaLCFS,aOut=aOut,RmidOut=RmidOut)
        except ValueError:
            raise ValueError('must read a-file for this data.')

    def getQProfile(self):
        """Returns safety factor q(psi).

        Returns:
            qpsi (Array): [1,n] array of q(psi).
        """
        return self._qpsi.copy()

    def getQ0(self):
        """Returns safety factor q on-axis, q0.

        Returns:
            q0 (Array): [1] array of q(psi=0).  Returns array for consistency 
            with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._q0 is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._q0.copy()

    def getQ95(self):
        """Returns safety factor q at 95% flux surface.

        Returns:
            q95 (Array): [1] array of q(psi=0.95).  Returns array for consistency 
            with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._q95 is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._q95.copy()

    def getQLCFS(self):
        """Returns safety factor q at LCFS (interpolated).

        Returns:
            qLCFS (Array): [1] array of q* (interpolated).  Returns array for 
            consistency with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not loaded.
        """
        if self._qLCFS is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._qLCFS.copy()

    def getQ1Surf(self,length_unit=1):
        """Returns outboard-midplane minor radius of q=1 surface.

        Keyword Args:
            length_unit (String or 1): unit of minor radius.  Defaults to 1
                (default unit, typically m)

        Returns:
            qr1 (Array): [1] array of minor radius of q=1 surface.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._rq1 is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq1'],length_unit)
            return unit_factor * self._rq1.copy()
    
    def getQ2Surf(self,length_unit=1):
        """Returns outboard-midplane minor radius of q=2 surface.

        Keyword Args:
            length_unit (String or 1): unit of minor radius.  Defaults to 1
                (default unit, typically m)

        Returns:
            qr2 (Array): [1] array of minor radius of q=2 surface.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._rq2 is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq2'],length_unit)
            return unit_factor * self._rq2.copy()

    def getQ3Surf(self,length_unit=1):
        """Returns outboard-midplane minor radius of q=3 surface.

        Keyword Args:
            length_unit (String or 1): unit of minor radius.  Defaults to 1
                (default unit, typically m)

        Returns:
            qr3 (Array): [1] array of minor radius of q=3 surface.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._rq3 is None:
            raise ValueError('must read a-file for this data.')
        else:
            unit_factor = self._getLengthConversionFactor(self._defaultUnits['_rq3'],length_unit)
            return unit_factor * self._rq3.copy()

    def getQs(self,length_unit=1):
        """Pulls q-profile data.

        Keyword Args:
            length_unit (String or 1): unit of minor radius.  Defaults to 1
                (default unit, typically m)
        
        Returns:
            namedtuple containing [q0,q95,qLCFS,rq1,rq2,rq3]

        Raises:
            ValueError: if a-file data is not read.
        """
        try:
            q0 = self.getQ0()
            q95 = self.getQ95()
            qLCFS = self.getQLCFS()
            rq1 = self.getQ1Surf(length_unit=length_unit)
            rq2 = self.getQ2Surf(length_unit=length_unit)
            rq3 = self.getQ3Surf(length_unit=length_unit)
            data = namedtuple('Qs',['q0','q95','qLCFS','rq1','rq2','rq3'])
            return data(q0=q0,q95=q95,qLCFS=qLCFS,rq1=rq1,rq2=rq2,rq3=rq3)
        except ValueError:
            raise ValueError('must read a-file for this data.')

    def getBtVac(self):
        """Returns vacuum toroidal field on-axis.

        Returns:
            BtVac (Array): [1] array of vacuum toroidal field.  Returns array 
            for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._btaxv is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._btaxv.copy()

    def getBtPla(self):
        """Returns plasma toroidal field on-axis.

        Returns:
            BtPla (Array): [1] array of toroidal field including plasma effects.
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._btaxp is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._btaxp.copy()

    def getBpAvg(self):
        """Returns average poloidal field.

        Returns:
            BpAvg (Array): [1] array of average poloidal field.  Returns array 
            for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._bpolav is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._bpolav.copy()

    def getFields(self):
        """Pulls vacuum and plasma toroidal field, poloidal field data.
        
        Returns:
            namedtuple containing [BtVac,BtPla,BpAvg]

        Raises:
            ValueError: if a-file data is not read.
        """
        try:
            btaxv = self.getBtVac()
            btaxp = self.getBtPla()
            bpolav = self.getBpAvg()
            data = namedtuple('Fields',['BtVac','BtPla','BpAvg'])
            return data(BtVac=btaxv,BtPla=btaxp,BpAvg=bpolav)
        except ValueError:
            raise ValueError('must read a-file for this data.')

    def getIpCalc(self):
        """Returns EFIT-calculated plasma current.

        Returns:
            IpCalc (Array): [1] array of EFIT-reconstructed plasma current.
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.
        """
        return self._IpCalc.copy()

    def getIpMeas(self):
        """Returns measured plasma current.

        Returns:
            IpMeas (Array): [1] array of measured plasma current.  Returns 
            array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._IpMeas is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._IpMeas.copy()

    def getJp(self):
        """Returns (r,z) grid of toroidal plasma current density.
        
        Data not read from g-file, not implemented for :py:class:`EqdskReader`.

        Raises:
            NotImplementedError: In all cases.
        """
        raise NotImplementedError('Jp not read from g-file.')

    def getBetaT(self):
        """Returns EFIT-calculated toroidal beta.

        Returns:
            BetaT (Array): [1] array of average toroidal beta.  Returns array 
            for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._betat is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._betat.copy()

    def getBetaP(self):
        """Returns EFIT-calculated poloidal beta.

        Returns:
            BetaP (Array): [1] array of average poloidal beta.  Returns array 
            for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read
        """
        if self._betap is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._betap.copy()

    def getLi(self):
        """Returns internal inductance of plasma.

        Returns:
            Li (Array): [1] array of internal inductance.  Returns array for 
            consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._Li is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._Li.copy()

    def getBetas(self):
        """Pulls EFIT-calculated betas and internal inductance.
        
        Returns:
            namedtuple containing [betat,betap,Li]

        Raises:
            ValueError: if a-file data is not read.
        """
        try:
            betat = self.getBetaT()
            betap = self.getBetaP()
            Li = self.getLi()
            data = namedtuple('Betas',['betat','betap','Li'])
            return data(betat=betat,betap=betap,Li=Li)
        except ValueError:
            raise ValueError('must read a-file for this data.')
            
    def getDiamagFlux(self):
        """Returns diamagnetic flux.

        Returns:
            Flux (Array): [1] array of measured diamagnetic flux.  Returns array 
            for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._diamag is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._diamag.copy()

    def getDiamagBetaT(self):
        """Returns diamagnetic-loop measured toroidal beta.

        Returns:
            BetaT (Array): [1] array of measured diamagnetic toroidal beta.   
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._betatd is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._betatd.copy()

    def getDiamagBetaP(self):
        """Returns diamagnetic-loop measured poloidal beta.

        Returns:
            BetaP (Array): [1] array of measured diamagnetic poloidal beta.  
            Returns array for consistency with 
           :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._betapd is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._betapd.copy()

    def getDiamagTauE(self):
        """Returns diamagnetic-loop energy confinement time.

        Returns:
            TauE (Array): [1] array of measured energy confinement time.  
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._tauDiamag is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._tauDiamag.copy()

    def getDiamagWp(self):
        """Returns diamagnetic-loop measured stored energy.

        Returns:
            Wp (Array): [1] array of diamagnetic stored energy.  
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._WDiamag is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._WDiamag.copy()

    def getDiamag(self):
        """Pulls diamagnetic flux, diamag. measured toroidal and poloidal beta, 
        stored energy, and energy confinement time.
        
        Returns:
            namedtuple containing [diaFlux,diaBetat,diaBetap,diaTauE,diaWp]

        Raises:
            ValueError: if a-file data is not read
        """
        try:
            dFlux = self.getDiamagFlux()
            betatd = self.getDiamagBetaT()
            betapd = self.getDiamagBetaP()
            dTau = self.getDiamagTauE()
            dWp = self.getDiamagWp()
            data = namedtuple('Diamag',['diaFlux','diaBetat','diaBetap','diaTauE','diaWp'])
            return data(diaFlux=dFlux,diaBetat=betatd,diaBetap=betapd,diaTauE=dTau,diaWp=dWp)
        except ValueError:
                raise ValueError('must read a-file for this data.')

    def getWMHD(self):
        """Returns EFIT-calculated stored energy.

        Returns:
            WMHD (Array): [1] array of EFIT-reconstructed stored energy.  
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._WMHD is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._WMHD.copy()

    def getTauMHD(self):
        """Returns EFIT-calculated energy confinement time.

        Returns:
            tauMHD (Array): [1] array of EFIT-reconstructed energy confinement
            time.  Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._tauMHD is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._tauMHD.copy()

    def getPinj(self):
        """Returns EFIT injected power.

        Returns:
            Pinj (Array): [1] array of EFIT-reconstructed injected power.  
            Returns array for consistency with 
            :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._Pinj is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._Pinj.copy()

    def getWbdot(self):
        """Returns EFIT d/dt of magnetic stored energy

        Returns:
            dWdt (Array): [1] array of d(Wb)/dt.  Returns array for consistency 
            with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._Wbdot is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._Wbdot.copy()

    def getWpdot(self):
        """Returns EFIT d/dt of plasma stored energy.

        Returns:
            dWdt (Array): [1] array of d(Wp)/dt.  Returns array for consistency 
            with :py:class:`Equilibrium <eqtools.core.Equilibrium>` 
            implementations with time variation.

        Raises:
            ValueError: if a-file data is not read.
        """
        if self._Wpdot is None:
            raise ValueError('must read a-file for this data.')
        else:
            return self._Wpdot.copy()

    def getBCentr(self):
        """returns Vacuum toroidal magnetic field in Tesla at Rcentr

        Returns:
            B_cent (Array): [nt] array of B_t at center [T]

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """

        return self._BCentr.copy()

    def getRCentr(self, length_unit=1):
        """returns radius where Bcentr evaluated

        Returns:
            R: Radial position where Bcent calculated [m]

        Raises:
            ValueError: if module cannot retrieve data from MDS tree.
        """

        unit_factor = self._getLengthConversionFactor(self._defaultUnits['_RCentr'], length_unit)
        return unit_factor * self._RCentr.copy()

    def getEnergy(self):
        """Pulls EFIT stored energy, energy confinement time, injected power, 
        and d/dt of magnetic and plasma stored energy.
        
        Returns:
            namedtuple containing [WMHD,tauMHD,Pinj,Wbdot,Wpdot]

        Raises:
            ValueError: if a-file data is not read.
        """
        try:
            WMHD = self.getWMHD()
            tauMHD = self.getTauMHD()
            Pinj = self.getPinj()
            Wbdot = self.getWbdot()
            Wpdot = self.getWpdot()
            data = namedtuple('Energy',['WMHD','tauMHD','Pinj','Wbdot','Wpdot'])
            return data(WMHD=WMHD,tauMHD=tauMHD,Pinj=Pinj,Wbdot=Wbdot,Wpdot=Wpdot)
        except ValueError:
            raise ValueError('must read a-file for this data.')

    def getParam(self,name):
        """Backup function, applying a direct path input for tree-like data 
        storage access for parameters not typically found in Equilbrium object.  
        Directly calls attributes read from g/a-files in copy-safe manner.

        Args:
            name (String): Parameter name for value stored in EqdskReader 
                instance.

        Returns:
            param (Array-like or scalar float): value stored as attribute in
            :py:class:`EqdskReader`.

        Raises:
            AttributeError: raised if no attribute is found.
        """
        try:
            return super(EqdskReader,self).__getattribute__(name)
        except AttributeError:
            try:
                attr = self.__getattribute__('_'+name)
                if type(attr) is scipy.array:
                    return attr.copy()
                else:
                    return attr
            except AttributeError:
                raise AttributeError('No attribute "_%s" found' % name)
        
    def getMachineCrossSection(self):
        """Method to pull machine cross-section from data storage, convert to 
        standard format for plotting routine.

        Returns:
            (`R_limiter`, `Z_limiter`)

            * **R_limiter** (`Array`) - [n] array of x-values for machine cross-section.
            * **Z_limiter** (`Array`) - [n] array of y-values for machine cross-section.
        """
        return (self._xlim,self._ylim)
        
    def getMachineCrossSectionFull(self):
        """Returns vectorization of machine cross-section.
        
        Absent additional data (not found in eqdsks) simply returns 
        self.getMachineCrossSection().
        """
        return self.getMachineCrossSection()

    def gfile(self, time=None, nw=None, nh=None, shot=None, name=None, 
              tunit='ms', title='EQTOOLS', nbbbs=100):
        """Generates an EFIT gfile with gfile naming convention
                  
        Keyword Args:
            time (scalar float): Time of equilibrium to
                generate the gfile from. This will use the specified
                spline functionality to do so. Allows for it to be 
                unspecified for single-time-frame equilibria.
            nw (scalar integer): Number of points in R.
                R is the major radius, and describes the 'width' of the 
                gfile.
            nh (scalar integer): Number of points in Z. In cylindrical
                coordinates Z is the height, and nh describes the 'height' 
                of the gfile.
            shot (scalar integer): The shot numer of the equilibrium.
                Used to help generate the gfile name if unspecified.
            name (String): Name of the gfile.  If unspecified, will follow
                standard gfile naming convention (g+shot.time) under current
                python operating directory.  This allows for it to be saved
                in other directories, etc.
            tunit (String): Specified unit for tin. It can only be 'ms' for
                milliseconds or 's' for seconds.
            title (String): Title of the gfile on the first line. Name cannot
                exceed 10 digits. This is so that the style of the first line
                is preserved.
            nbbbs (scalar integer): Number of points to define the plasma 
                seperatrix within the gfile.  The points are defined equally
                spaced in angle about the plasma center.  This will cause the 
                x-point to be poorly defined.

        Raises:
            ValueError: If title is longer than 10 characters.
        
        Examples:
            All assume that `Eq_instance` is a valid instance of the appropriate
            extension of the :py:class:`Equilibrium` abstract class (example
            shot number of 1001).
            
            Generate a gfile (time at t=.26s) output of g1001.26::
            
                Eq_instance.gfile()
            
        """
        if time is None:
            time = self.getTimeBase()

        super(EqdskReader,self).gfile(time,
                                      nw=nw,
                                      nh=nh,
                                      shot=shot,
                                      name=name,
                                      tunit=tunit,
                                      title=title,
                                      nbbbs=nbbbs)

    def plotFlux(self,fill=True,mask=True):
        """streamlined plotting of flux contours directly from psi grid

        Keyword Args:
            fill (Boolean): Default True.  Set True to plot filled contours of 
                flux delineated by black outlines.  Set False to instead plot 
                color-coded line contours on a blank background.
            mask (Boolean): Default True.  Set True to draw a clipping mask 
                based on the limiter outline for the flux contours.  Set False 
                to draw the full RZ grid.
        """
        plt.ion()

        try:
            psiRZ = self.getFluxGrid()[0]
            rGrid = self.getRGrid()
            zGrid = self.getZGrid()

            RLCFS = self.getRLCFS()[0]
            ZLCFS = self.getZLCFS()[0]

            Rlim,Zlim = self.getMachineCrossSection()
        except ValueError:
            raise AttributeError('cannot plot EFIT flux map.')       

        fig = plt.figure(figsize=(6,11))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlabel('$R$ (m)')
        ax.set_ylabel('$Z$ (m)')
        ax.set_title(self._gfilename)

        if fill:
            ax.contourf(rGrid,zGrid,psiRZ,50,zorder=2)
            ax.contour(rGrid,zGrid,psiRZ,50,colors='k',linestyles='solid',
                       zorder=3)
        else:
            ax.contour(rGrid,zGrid,psiRZ,50,linestyles='solid',linewidth=2,
                       zorder=2)
        ax.plot(RLCFS,ZLCFS,'r',linewidth=3)

        # generate graphical mask for limiter wall
        if mask:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bound_verts = [(xlim[0],ylim[0]),(xlim[0],ylim[1]),(xlim[1],ylim[1]),(xlim[1],ylim[0]),(xlim[0],ylim[0])]
            poly_verts = [(Rlim[i],Zlim[i]) for i in range(len(Rlim) - 1, -1, -1)]
            
            bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
            poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]
            
            path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
            patch = mpatches.PathPatch(path,facecolor='white',edgecolor='none')
            patch = ax.add_patch(patch)
            patch.set_zorder(4)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        ax.plot(Rlim,Zlim,'k',linewidth=3,zorder=5)
        fig.show()

        return fig
