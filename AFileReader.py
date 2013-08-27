import numpy as np
import re
import csv


class AFileReader(object):
    """
    class to read a-file (EFIT time-history data storage).
    """
    def __init__(self,afile):
        """
        initialize object, reading from file.

        INPUTS:
        afile: (str) path to a-file
        """
        self._afile = afile

        with open(afile,'r') as readfile:
            reader = csv.reader(readfile)   # skip delimiter, return as single string - let regex handle splitting.  Use csv.reader for StopIteration error handling.
            # date header line
            line = next(reader)[0].split()
            self._date = line[1]

            # shot header line
            line = next(reader)[0].split()
            self._shot = int(line[0])
            
            # time index line
            line = next(reader)[0].split()
            self._time = np.array(float(line[0]))     # time point in ms

            # header line
            line = next(reader)[0].split()
            jflag = int(line[2])
            lflag = int(line[3])
            self._limloc = line[4]  # limiter location (string)
            mco2v = int(line[5])
            mco2r = int(line[6])
            qmflag = line[7]

            # read tsaisq(?), mag-axis R, bcentr, pasmat
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._tsaisq = float(line[0])
            self._rcencm = float(line[1])
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
            nrows = mco2v/4
            if mco2v % 4 != 0:
                nrows += 1

            self._rco2v = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._rco2v.append(float(val))
            self._rco2v = np.array(self._rco2v)

            self._dco2v = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._dco2v.append(float(val))
            self._dco2v = np.array(self._dco2v)

            # read next mco2r values for rco2r, dco2r
            nrows = mco2r/4
            if mco2r % 4 != 0:
                nrows += 1

            self._rco2r = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._rco2r.append(float(val))
            self._rco2r = np.array(self._rco2r)

            self._dco2r = []
            for i in range(nrows):
                line = next(reader)[0]
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                for val in line:
                    self._dco2r.append(float(val))
            self._dco2r = np.array(self._dco2r)

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
            self._olefs = float(line[2])
            self._orighs = float(line[3])

            # read otops, sibdry, areao, wplasm
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._otops = float(line[0])
            self._sibdry = float(line[1])   # psi at boundary
            self._areao = float(line[2])    # area of LCFS
            self._wplasm = float(line[3])   # EFIT-calculated stored energy

            # read terror, elongm, qqmagx, cdflux
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._terror = float(line[0])
            self._elongm = float(line[1])   # elongation at magnetic axis
            self._qqmagx = float(line[2])   # q(psi) at psi=0 (q0)
            self._cdflux = float(line[3])

            # read alpha, rttt, psiref, xndnt
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._alpha = float(line[0])
            self._rttt = float(line[1])
            self._psiref = float(line[2])
            self._xndnt = float(line[3])

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
            self._sepexp = float(line[0])
            self._obots = float(line[1])
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
            self._cmerci = float(line[2])
            self._tavem = float(line[3])

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
            self._csilop = np.array(dat[0:nsilop+1])
            self._cmpr2 = np.array(dat[nsilop+1:])

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
            self._ccbrsp = np.array(self._ccbrsp)

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
            self._eccurt = np.array(self._eccurt)

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
            self._slantu = float(line[0])
            self._slantl = float(line[1])
            self._zuperts = float(line[2])
            self._chipre = float(line[3])

            # read cjor95, pp95, ssep, yyy2
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._cjor95 = float(line[0])   # flux-surface-averaged current density normalized to I/A at 95% flux
            self._pp95 = float(line[1])
            self._ssep = float(line[2])     # null position measurements (~1 are USN ,~-1 are LSN, ~0 are DN.  Defaults to 40 for limited shapes)
            self._yyy2 = float(line[3])

            # read xnnc, cprof, oring, cjor0
            line = next(reader)[0]
            line = re.findall('-?\d.\d*E[-+]\d*',line)
            self._xnnc = float(line[0])
            self._cprof = float(line[1])
            self._oring = float(line[2])
            self._cjor0 = float(line[3])    # flux-surface-averaged current density normalized to I/A at axis

            # this completes the old-style (pre-1997) a-file write.
            # further values written within error handler for legacy
            # support of older a-files.
            try:
                # read fexpan, qqmin, chigamt, ssi01
                line = next(reader)[0]
                lastline = line     # store previous line for next read - error handler will catch at empty read, last line retains footer
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._fexpan = float(line[0])
                self._qqmin = float(line[1])
                self._chigamt = float(line[2])
                self._ssi01 = float(line[3])

                # read fexpvs, sepnose, ssi95, rqqmin
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._fexpvs = float(line[0])
                self._sepnose = float(line[1])
                self._ssi95 = float(line[2])
                self._rqqmin = float(line[3])

                # read cjor99, cj1ave, rmidin, rmidout
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._cjor99 = float(line[0])
                self._cj1ave = float(line[1])
                self._rmidin = float(line[2])
                self._rmidout = float(line[3])  # out major radius at Z=0.0 (LCFS position)

                # read psurfa, peak, dminux, dminlx
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._psurfa = float(line[0])
                self._peak = float(line[1])
                self._dminux = float(line[2])
                self._dminlx = float(line[3])

                # read dolubaf, dolubafm, diludom, diludomm
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._dolubaf = float(line[0])
                self._dolubafm = float(line[1])
                self._diludom = float(line[2])
                self._diludomm = float(line[3])

                # read ratsol, rvsiu, zvsiu, rvsid
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._ratsol = float(line[0])
                self._rvsiu = float(line[1])
                self._zvsiu = float(line[2])
                self._rvsid = float(line[3])

                # read zvsid, rvsou, zvsou, rvsod
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._zvsid = float(line[0])
                self._rvsou = float(line[1])
                self._zvsou = float(line[2])
                self._rvsod = float(line[3])

                # read zvsod, condno, dollbaf, dollbafm
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._zvsod = float(line[0])
                self._condno = float(line[1])
                self._dollbaf = float(line[2])
                self._dollbafm = float(line[3])

                # read dilldom, dilldomm, dummy vars
                line = next(reader)[0]
                lastline = line
                line = re.findall('-?\d.\d*E[-+]\d*',line)
                self._dilldom = float(line[0])
                self._dilldomm = float(line[1])

                # read end tag line
                line = next(reader)[0].split()
                self._efittype = line[-1]

            except:
                print('Old-style a-file.  Some parameters are depreciated.')
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
        """
        overrides default __str__method with more useful output.
        """
        return 'a-file data from '+self._afile

    def __getattribute__(self, name):
        """
        Tries to get attribute as written.  If this fails, trys to call the attribute
        with preceding underscore, marking a pseudo-private variable.  If this exists,
        returns a copy-safe value.  If this fails, raises AttributeError.  Generates a
        copy-safe version of each data attribute.
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
        Raises AttributeError if the object already has a method get[name], as
        creation of such an attribute would interfere with the automatic
        property generation in __getattribute__.
        """
        if hasattr(self, '_'+name):
            raise AttributeError("AFileReader object already has data attribute "
                                 "'_%(n)s', creating attribute '%(n)s' will"
                                 " conflict with automatic property generation."
                                 % {'n': name})
        else:
            super(AFileReader, self).__setattr__(name, value)














