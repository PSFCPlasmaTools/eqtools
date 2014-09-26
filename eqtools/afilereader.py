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
This module contains the AFileReader class, a lightweight data
handler for a-file (time-history) datasets.

Classes:
    AFileReader: 
        Data-storage class for a-file data.  Reads
        data from ASCII a-file, storing as copy-safe object
        attributes.
"""

import numpy as np
import re
import csv
import warnings

class AFileReader(object):
    """
    Class to read ASCII a-file (time-history data storage) into lightweight, 
    user-friendly data structure.

    A-files store data blocks of scalar time-history data for EFIT 
    plasma equilibrium.  Each parameter is read into a pseudo-private object 
    attribute (marked by a leading underscore), followed by the standard
    EFIT variable names.
    
    initialize object, reading from file.

    Args:
        afile (String): file path to a-file

    Examples:
        Load a-file data located at `file_path`::

            afr = eqtools.AFileReader(file_path)

        Recover a datapoint (for example, `shot`, stored as `afr._shot`),
        using copy-protected __getattribute__ method::

            shot = afr.shot

        Assign a new attribute to afr -- note that this will raise an
        AttributeError if attempting to overwrite a previously-stored
        attribute::

            afr.attribute = val
    """
    def __init__(self, afile):
        self._afile = afile

        with open(afile,'r') as readfile:
            # skip delimiter, return as single string let regex handle splitting 
            # Use csv.reader for StopIteration error handling.
            reader = csv.reader(readfile)
            # date header line
            line = next(reader)[0].split()
            self._date = line[1]    # date a-file was created

            # shot header line
            line = next(reader)[0].split()
            self._shot = int(line[0])   # shot index
            
            # time index line
            line = next(reader)[0].split()
            self._time = float(line[0])     # time point in ms

            # header line
            line = next(reader)[0]
            line = re.findall('[\w.]+',line)
            self._jflag = int(line[1])  # error flag
            self._lflag = int(line[2])  # error flag
            self._limloc = line[3]      # limiter location (string)
            self._mco2v = int(line[4])  # number of vertical CO2 laser chords
            self._mco2r = int(line[5])  # number of horizontal CO2 laser chords
            self._qmflag = line[6]      # flag indicating fixed q0 for fit

            # read tsaisq(?), mag-axis R, bcentr, pasmat
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._tsaisq = float(line[0])   # chi-squared for equilibrium solution
            self._rcencm = float(line[1])   # nominal center (cm) - used for F=RB in vacuum
            self._bcentr = float(line[2])   # Btor at rcentr
            self._pasmat = float(line[3])   # measured plasma current

            # read cpasma, rout, zout, aout
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._cpasma = float(line[0])   # calculated plasma current
            self._rout = float(line[1])     # major radius of geometric center
            self._zout = float(line[2])     # Z of LCFS (constructed)
            self._aout = float(line[3])     # minor radius of LCFS

            # read eout, doutu, doutl, vout
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._eout = float(line[0])     # elongation of LCFS
            self._doutu = float(line[1])    # upper triangularity of LCFS
            self._doutl = float(line[2])    # lower triangularity of LCFS
            self._vout = float(line[3])     # volume of LCFS

            # read rcurrt, zcurrt, qsta, betat
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._rcurrt = float(line[0])   # current-averaged major radius
            self._zcurrt = float(line[1])   # current-averaged Z
            self._qsta = float(line[2])     # q* (GA definition)
            self._betat = float(line[3])    # toroidal beta (calculated)

            # read betap, ali, oleft, oright
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._betap = float(line[0])    # poloidal beta (calculated)
            self._ali = float(line[1])      # internal inductance
            self._oleft = float(line[2])    # inner gap
            self._oright = float(line[3])   # outer gaps

            # read otop, obott, qpsib, vertn
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._otop = float(line[0])     # top gap
            self._obott = float(line[1])    # bottom gap
            self._qpsib = float(line[2])    # q(psi) at 95% flux
            self._vertn = float(line[3])    # decay index at current centroid

            # read next mco2v values for rco2v, dco2v
            nrows = self._mco2v/4
            if self._mco2v % 4 != 0:
                nrows += 1

            self._rco2v = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._rco2v.append(float(val))
            self._rco2v = np.array(self._rco2v)     # chord length of vertical CO2 laser chords

            self._dco2v = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._dco2v.append(float(val))
            self._dco2v = np.array(self._dco2v)     # line-averaged density along vertical CO2 chords

            # read next mco2r values for rco2r, dco2r
            nrows = self._mco2r/4
            if self._mco2r % 4 != 0:
                nrows += 1

            self._rco2r = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._rco2r.append(float(val))
            self._rco2r = np.array(self._rco2r)     # chord length of horizontal CO2 chords

            self._dco2r = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._dco2r.append(float(val))
            self._dco2r = np.array(self._dco2r)     # line-averaged density along horizontal CO2 chords

            # read shearb, bpolav, s1, s2
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._shearb = float(line[0])   # shear parameter at boundary
            self._bpolav = float(line[1])   # average poloidal field
            self._s1 = float(line[2])       # first Shafranov integral
            self._s2 = float(line[3])       # second Shafranov integral

            # read s3, qout, olefs, orighs
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._s3 = float(line[0])       # third Shafranov integral
            self._qout = float(line[1])     # q(psi) at LCFS
            self._olefs = float(line[2])    # inner gap to secondary separatrix
            self._orighs = float(line[3])   # outer gap to secondary separatrix

            # read otops, sibdry, areao, wplasm
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._otops = float(line[0])    # top gap to secondary separatrix
            self._sibdry = float(line[1])   # psi at boundary
            self._areao = float(line[2])    # area of LCFS
            self._wplasm = float(line[3])   # EFIT-calculated stored energy

            # read terror, elongm, qqmagx, cdflux
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._terror = float(line[0])   # convergence parameter - norm. change in flux at last iteration
            self._elongm = float(line[1])   # elongation at magnetic axis
            self._qqmagx = float(line[2])   # q(psi) at psi=0 (q0)
            self._cdflux = float(line[3])   # computed diamagnetic flux

            # read alpha, rttt, psiref, xndnt
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._alpha = float(line[0])    # Shafranov boundary line integral parameter
            self._rttt = float(line[1])     # Shafranov boundary line integral parameter
            self._psiref = float(line[2])   # reference flux (flux on loop #0)
            self._xndnt = float(line[3])    # indentation

            # read rseps[0], zseps[0], rseps[1], zseps[1]
            self._rseps = [0,0]     # radial positions of upper,lower x-points
            self._zseps = [0,0]     # Z positions of upper,lower x-points
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._rseps[0] = float(line[0])
            self._zseps[0] = float(line[1])
            self._rseps[1] = float(line[2])
            self._zseps[1] = float(line[3])

            # read sepexp, obots, btaxp, btaxv
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._sepexp = float(line[0])   # separatrix radial expansion
            self._obots = float(line[1])    # bottom gap to secondary separatrix
            self._btaxp = float(line[2])    # Btor on-axis = F(0)/rmaxis
            self._btaxv = float(line[3])    # vacuum Btor on-axis = F(1)/rmaxis

            # read aaq1, aaq2, aaq3, seplim
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._aaq1 = float(line[0])     # radius of q=1 surface
            self._aaq2 = float(line[1])     # radius of q=2 surface
            self._aaq3 = float(line[2])     # radius of q=3 surface
            self._seplim = float(line[3])   # minimum gap to limiter

            # read rmagx, zmagx, simagx, taumhd
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._rmagx = float(line[0])    # major radius of magnetic axis
            self._zmagx = float(line[1])    # Z of magnetic axis
            self._simagx = float(line[2])   # flux at magnetic axis
            self._taumhd = float(line[3])   # EFIT-calculated energy confinement time

            # read betapd, betatd, wplasmd, fluxx
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._betapd = float(line[0])           # diamagnetic-loop poloidal beta
            self._betatd = float(line[1])           # diamagnetic-loop toroidal beta
            self._wplasmd = float(line[2])          # diamagnetic-loop stored energy
            self._diamag = float(line[3])/1.e3      # diamagnetic flux

            # read vloopt, taudia, cmerci, tavem
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._vloopt = float(line[0])   # loop voltage
            self._taudia = float(line[1])   # energy confinement time from diamagnetic measurements
            self._qmerci = float(line[2])   # Mercier stability criterion on axial q(0), q(0) > QMERCI for stability
            self._tavem = float(line[3])    # time for averaging magnetic data

            # header line: read nsilop, magpri, nfcoil, nesum
            line = next(reader)[0].split()
            nsilop = int(line[0])
            magpri = int(line[1])
            nfcoil = int(line[2])
            nesum = int(line[3])

            # read csilop, cmpr2
            # for god knows what reason, these are written as a single nsilop+magpri block.
            npts = nsilop+magpri
            nrows = npts/4
            if npts % 4 != 0:
                nrows += 1

            dat = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    dat.append(float(val))
            self._csilop = np.array(dat[0:nsilop+1])    # calculated psi loop signals
            self._cmpr2 = np.array(dat[nsilop+1:])      # calculated Bpol coil signals

            # read ccbrsp
            nrows = nfcoil/4
            if nfcoil % 4 != 0:
                nrows += 1

            self._ccbrsp = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._ccbrsp.append(float(val))
            self._ccbrsp = np.array(self._ccbrsp)       # calculated F-coil currents

            # read eccurt
            nrows = nesum/4
            if nesum % 4 != 0:
                nrows += 1

            self._eccurt = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._eccurt.append(float(val))
            self._eccurt = np.array(self._eccurt)       # calculated E-coil currents

            # read pbinj, rvsin, zvsin, rvsout
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._pbinj = float(line[0])    # injected power
            self._rvsin = float(line[1])    # R of inner strike point
            self._zvsin = float(line[2])    # Z of inner strike point
            self._rvsout = float(line[3])   # R of outer strike point

            # read zvsout, vsurfa, wpdot, wbdot
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._zvsout = float(line[0])   # Z of outer strike point
            self._vsurfa = float(line[1])   # surface voltage
            self._wpdot = float(line[2])    # time-derivative of plasma energy
            self._wbdot = float(line[3])    # time-derivative of magnetic energy

            # read slantu, slantl, zuperts, chipre
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._slantu = float(line[0])   # gap to upper outboard limiter
            self._slantl = float(line[1])   # gap to lower outboard limiter
            self._zuperts = float(line[2])  # intersection of LCFS and TS laser chord
            self._chipre = float(line[3])   # chi-squared of kinetic pressure data

            # read cjor95, pp95, ssep, yyy2
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._cjor95 = float(line[0])   # flux-surface-averaged current density normalized to I/A at 95% flux
            self._pp95 = float(line[1])     # p-prime at 95% flux
            self._ssep = float(line[2])     # null position measurements (~1 are USN ,~-1 are LSN, ~0 are DN.  Defaults to 40 for limited shapes)
            self._yyy2 = float(line[3])     # current moment y2

            # read xnnc, cprof, oring, cjor0
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._xnnc = float(line[0])     # vertical stability parameter
            self._cprof = float(line[1])    # profile flag for edge current profile (for consistency with g-file)
            self._oring = float(line[2])    # gap to inner ring coil
            self._cjor0 = float(line[3])    # flux-surface-averaged current density normalized to I/A at axis

            # this completes the old-style (pre-1997) a-file write.
            # further values written within error handler for legacy
            # support of older a-files.
            try:
                # read fexpan, qqmin, chigamt, ssi01
                line = next(reader)[0]
                lastline = line     # store previous line for next read - error handler will catch at empty read, last line retains footer
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._fexpan = float(line[0])   # flux expansion at x-point
                self._qqmin = float(line[1])    # minimum safety factor qmin
                self._chigamt = float(line[2])  # total chi-squared of MSE
                self._ssi01 = float(line[3])    # magnetic shear at 1% poloidal flux

                # read fexpvs, sepnose, ssi95, rqqmin
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._fexpvs = float(line[0])   # flux expansion at outer lower vessel strike point
                self._sepnose = float(line[1])  # radial distance between x-point and external field line at ZNOSE
                self._ssi95 = float(line[2])    # magnetic shear at 95% poloidal flux
                self._rqqmin = float(line[3])   # position of qmin (sqrt of normalized volume)

                # read cjor99, cj1ave, rmidin, rmidout
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._cjor99 = float(line[0])   # flux-surface averaged current density normalized to I/A at 99% flux
                self._cj1ave = float(line[1])   # flux-surface averaged in plasma outer 5% poloidal flux
                self._rmidin = float(line[2])   # inboard major radius at Z=0.0 (LCFS position)
                self._rmidout = float(line[3])  # outboard major radius at Z=0.0 (LCFS position)

                # read psurfa, peak, dminux, dminlx
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._psurfa = float(line[0])   # plasma boundary surface area in m^2
                self._peak = float(line[1])     # peak to average plasma pressure
                self._dminux = float(line[2])   # distance between limiter and upper x-point
                self._dminlx = float(line[3])   # distance between limiter and lower x-point

                # read dolubaf, dolubafm, diludom, diludomm
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._dolubaf = float(line[0])  # distance from outer leg to upper baffle
                self._dolubafm = float(line[1]) # distance at outboard midplane between LCFS and flux surf. intersecting upper baffle
                self._diludom = float(line[2])  # distance from inner leg to upper dome
                self._diludomm = float(line[3]) # distance at inner midplane between LCFS and flux surf. intersecting upper dome

                # read ratsol, rvsiu, zvsiu, rvsid
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._ratsol = float(line[0])   # ratio of flux expansion at inner midplane vs outer midplane
                self._rvsiu = float(line[1])    # major radius of inner upper strikepoint
                self._zvsiu = float(line[2])    # Z of inner upper strikepoint
                self._rvsid = float(line[3])    # major radius of inner lower strikepoint

                # read zvsid, rvsou, zvsou, rvsod
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._zvsid = float(line[0])    # Z of inner lower strikepoint
                self._rvsou = float(line[1])    # major radius of outer upper strikepoint
                self._zvsou = float(line[2])    # Z of outer upper strike point
                self._rvsod = float(line[3])    # major radius of outer lower strike point

                # read zvsod, condno, dollbaf, dollbafm
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._zvsod = float(line[0])    # Z of outer lower strikepoiunt
                self._condno = float(line[1])   # condition number from least-squares fitting routine
                self._dollbaf = float(line[2])  # distance from outer leg to lower baffle
                self._dollbafm = float(line[3]) # distance from outer midplane LCFS to flux surf. intersecting lower baffle

                # read dilldom, dilldomm, dummy vars
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._dilldom = float(line[0])  # distance from inner leg to lower dome
                self._dilldomm = float(line[1]) # distance between inner midplane LCFS and surface intersecting lower dome

                # read end tag line
                line = next(reader)[0].split()
                self._efittype = line[-1]   # tag for EFIT type ('MAG','KINETIC',etc)

            except:
                warnings.warn('Old-style a-file.'
                              '  Some parameters are depreciated.',
                              UserWarning)
                self._fexpan = None
                self._qqmin = None
                self._chigamt = None
                self._ssi01 = None
                self._fexpvs = None
                self._sepnose = None
                self._ssi95 = None
                self._rqqmin = None
                self._cjor99 = None
                self._cj1ave = None
                self._rmidin = None
                self._rmidout = None
                self._psurfa = None
                self._peak = None
                self._dminux = None
                self._dminlx = None
                self._dolubaf = None
                self._dolubafm = None
                self._diludom = None
                self._diludomm = None
                self._ratsol = None
                self._rvsiu = None
                self._zvsiu = None
                self._rvsid = None
                self._zvsid = None
                self._rvsou = None
                self._zvsou = None
                self._rvsod = None
                self._zvsod = None
                self._condno = None
                self._dollbaf = None
                self._dollbafm = None
                self._dilldom = None
                self._dilldomm = None
                self._efittype = lastline.split()[-1]

    def __str__(self):
        """overrides default `__str__` method with more useful output.
        """
        return 'a-file data from '+self._afile

    def __getattribute__(self, name):
        """
        Copy-safe attribute retrieval method overriding default 
        `object.__getattribute__`.

        Tries to retrieve attribute as-written (first check for default object 
        attributes).  If that fails, looks for pseudo-private attributes, 
        marked by preceding underscore, to retrieve data values.  If this 
        fails, raise AttributeError.

        Args:
            name (String): Name (without leading underscore for data variables) 
                of attribute.

        Raises:
            AttributeError: if no attribute can be found.
        """
        try:
            return super(AFileReader,self).__getattribute__(name)
        except AttributeError:
            try:
                attr = super(AFileReader,self).__getattribute__('_'+name)
                return attr
            except AttributeError:
                raise AttributeError('No attribute "%s" found' % name)

    def __setattr__(self, name, value):
        """
        Copy-safe attribute setting method overriding default 
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
            raise AttributeError("AFileReader object already has data attribute"
                                 " '_%(n)s', creating attribute '%(n)s' will"
                                 " conflict with automatic property generation."
                                 % {'n': name})
        else:
            super(AFileReader, self).__setattr__(name, value)