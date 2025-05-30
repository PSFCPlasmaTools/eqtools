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

import numpy
import scipy.interpolate
import warnings
import time

from . import core

try:
    import matplotlib.pyplot as plt
    _has_plt = False
except Exception:
    warnings.warn(
        "Matplotlib.pyplot module could not be loaded -- classes that use "
        "pyplot will not work.",
        ModuleWarning
    )
    _has_plt = False

try:
    from . import trispline
    _has_trispline = True
except ImportError:
    warnings.warn("trispline module could not be loaded -- tricubic spline "
                  "interpolation will not be available.",
                  core.ModuleWarning)
    _has_trispline = False


def gfile(
    obj, tin, nw=None, nh=None, shot=None, name=None, tunit='ms',
    title='EQTOOLS', nbbbs=100
):
    """Generates an EFIT gfile with gfile naming convention

        Args:
            obj (eqtools Equilibrium Object): Object which describes the tokamak
                This functionality is dependent on matplotlib, and is not
                not retained in core.py for this reason. It is a hidden
                function which takes an arbitrary equilibrium object and
                generates a gfile.
            tin (scalar float): Time of equilibrium to
                generate the gfile from. This will use the specified
                spline functionality to do so.

        Keyword Args:
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

            Generate a gfile at t=0.26s, output of g1001.26::

                gfile(Eq_instance,.26)

        """

    if shot is None:
        shot = obj._shot

    timeConvertDict = {'ms': 1000., 's': 1.}
    stin = str(
        int(
            float(tin) * timeConvertDict[tunit] /
            timeConvertDict[obj._defaultUnits['_time']]
        )
    )

    if name is None:
        name = 'g'+str(shot)+'.'+stin

    if nw is None:
        nw = len(obj.getRGrid())

    if nh is None:
        nh = len(obj.getZGrid())

    if len(title) > 10:
        raise ValueError('title is too long')

    header = (
        title + (11 - len(title)) * ' ' +
        time.strftime('%m/%d/%Y') +
        '   '+str(shot)+' ' + stin + tunit
    )

    header = (
        header + (51 - len(header)) * ' ' + '3 ' + str(nw) + ' ' + str(nh) +
        '\n'
    )

    rgrid = numpy.linspace(obj.getRGrid()[0], obj.getRGrid()[-1], nw)
    zgrid = numpy.linspace(obj.getZGrid()[0], obj.getZGrid()[-1], nh)
    rgrid2, zgrid2 = numpy.meshgrid(rgrid, zgrid)
    print(header)

    gfiler = open(name, 'wb')
    gfiler.write(header)

    gfiler.write(_fmt([obj.getRGrid()[-1]-obj.getRGrid()[0],
                       obj.getZGrid()[-1]-obj.getZGrid()[0],
                       obj.getRCentr(),
                       obj.getRGrid()[0],
                       obj.getZGrid()[-1]/2.+obj.getZGrid()[0]/2.]))

    rcent = obj.getMagRSpline()(tin)
    zcent = obj.getMagZSpline()(tin)

    if obj._tricubic:
        psiLCFS = -1*obj.getCurrentSign()*obj._getLCFSPsiSpline()(tin)
        psi0 = -1*obj.getCurrentSign()*obj._getPsi0Spline()(tin)
        bcent = trispline.UnivariateInterpolator(
            obj.getTimeBase(),
            obj.getBCentr(),
            k=3
        )
        bcent0 = bcent(tin)

    else:
        try:
            idx = obj._getNearestIdx(tin, obj.getTimeBase())
        except ValueError:  # correction necessary for eqdskfiles
            idx = 0

        psiLCFS = obj.getFluxLCFS()[idx]
        psi0 = obj.getFluxAxis()[idx]
        bcent0 = obj.getBCentr()[idx]

    gfiler.write(_fmt([rcent,
                       zcent,
                       psi0,
                       psiLCFS,
                       bcent0]))

    if obj._tricubic:
        temp = trispline.UnivariateInterpolator(obj.getTimeBase(),
                                                obj.getIpCalc(),
                                                k=3)
        Ip = temp(tin)
    else:
        Ip = obj.getIpCalc()[idx]

    gfiler.write(_fmt([Ip,
                       psi0,
                       0.,
                       rcent,
                       0.]))

    gfiler.write(_fmt([zcent,
                       0.,
                       psiLCFS,
                       0.,
                       0.]))

    pts1 = numpy.linspace(0., 1., nw)
    # this needs to be time mapped (sigh)
    if not obj._tricubic:
        for i in [obj.getF(),
                  obj.getFluxPres(),
                  obj.getFFPrime(),
                  obj.getPPrime()]:

            pts0 = numpy.linspace(0., 1., i.shape[-1])  # find original nw
            temp = scipy.interpolate.interp1d(pts0,
                                              numpy.atleast_2d(i)[idx],
                                              kind='nearest',
                                              bounds_error=False)
            gfiler.write(_fmt(temp(pts1).ravel()))

    else:
        tempt = tin*numpy.ones(pts1.shape)
        for i in [obj.getF(),
                  obj.getFluxPres(),
                  obj.getFFPrime(),
                  obj.getPPrime()]:

            pts0 = numpy.linspace(0., 1., i.shape[-1])  # find original nw
            temp = scipy.interpolate.RectBivariateSpline(obj.getTimeBase(),
                                                         pts0,
                                                         numpy.atleast_2d(i))
            gfiler.write(_fmt(temp.ev(tempt, pts1).ravel()))

    psiRZ = -1*obj.getCurrentSign()*obj.rz2psi(rgrid2,
                                               zgrid2,
                                               tin)
    gfiler.write(_fmt(psiRZ.ravel()))  # spline with new rz grid

    if not obj._tricubic:
        temp = scipy.interpolate.interp1d(
            pts0,
            numpy.atleast_2d(obj.getQProfile())[idx],
            kind='nearest',
            bounds_error=False
        )

        gfiler.write(_fmt(temp(pts1).ravel()))

    else:
        temp = scipy.interpolate.RectBivariateSpline(
            obj.getTimeBase(),
            pts0,
            numpy.atleast_2d(obj.getQProfile())
        )

        gfiler.write(_fmt(temp.ev(tempt, pts1).ravel()))

    # find plasma boundary
    out = _findLCFS(rgrid,
                    zgrid,
                    psiRZ,
                    rcent,
                    zcent,
                    psiLCFS,
                    nbbbs=nbbbs)

    # write boundary
    lim = numpy.array(obj.getMachineCrossSection()).T

    gfiler.write('  '+str(int(len(out)))+'   '+str(int(len(lim)))+'\n')

    gfiler.write(_fmt(out.ravel()))

    gfiler.write(_fmt(lim.ravel()))

    gfiler.close()


def _findLCFS(rgrid, zgrid, psiRZ, rcent, zcent, psiLCFS, nbbbs=100):
    """ internal function for finding the last closed flux surface
    based off of a Equilibrium instance"""

    if not _has_plt:
        raise RuntimeError('Need matplotlib to run _findLCFS!')

    ang = numpy.linspace(-numpy.pi, numpy.pi, nbbbs)

    plt.ioff()
    fig = plt.figure()
    cs = plt.contour(rgrid,
                     zgrid,
                     numpy.squeeze(psiRZ),
                     numpy.atleast_1d(psiLCFS))

    splines = []
    for i in cs.collections[0].get_paths():
        temp = i.vertices
        # turn points into polar coordinates about the plasma center
        rvals = numpy.sqrt((temp[:, 0] - rcent)**2 + (temp[:, 1] - zcent)**2)
        thetvals = numpy.arctan2(temp[:, 1] - zcent, temp[:, 0] - rcent)

        # find all monotonic sections of contour line in r,theta space
        temp = numpy.diff(thetvals)
        idx = 0
        sign = numpy.sign(temp[0])
        for j in range(len(temp)-1):

            if (numpy.sign(temp[j+1]) != sign):
                sign = numpy.sign(temp[j+1])
                # only write data if the jump at the last point is well resolved

                if (j+2-idx > 2):  # abs(thetvals[idx]-thetvals[j+1]) < 7*numpy.pi/4) and
                    plt.plot(thetvals[idx:j+2], rvals[idx:j+2], 'o')
                    sortang = numpy.argsort(thetvals[idx:j+2])
                    splines += [
                        scipy.interpolate.interp1d(
                            thetvals[sortang+idx],
                            rvals[sortang+idx],
                            kind='linear',
                            bounds_error=False,
                            fill_value=numpy.inf
                        )
                    ]
                idx = j+1

        if (len(thetvals) - idx > 2):
            plt.plot(thetvals[idx:], rvals[idx:], 'o')
            sortang = numpy.argsort(thetvals[idx:])
            splines += [scipy.interpolate.interp1d(thetvals[sortang+idx],
                                                   rvals[sortang+idx],
                                                   kind='linear',
                                                   bounds_error=False,
                                                   fill_value=numpy.inf)]

    # construct a set of angles about the center, and use the splines
    # to find the closest part of the contour to the center at that
    # angle, this is the LCFS, store value. If no value is found, store
    # an infite value, which is then tossed out.
    outr = numpy.empty((nbbbs,))

    for i in range(nbbbs):
        temp = numpy.inf
        for j in splines:
            pos = j(ang[i])
            if pos < temp:
                temp = pos
        outr[i] = temp

    # remove infinites
    ang = ang[numpy.isfinite(outr)]
    outr = outr[numpy.isfinite(outr)]

    # move back to r,z space
    output = numpy.empty((2, len(ang) + 1))
    output[0, :-1] = outr*numpy.cos(ang) + rcent
    output[1, :-1] = outr*numpy.sin(ang) + zcent
    output[0, -1] = output[0, 0]
    output[1, -1] = output[1, 0]

    # turn off plotting stuff
    plt.ion()
    plt.clf()
    plt.close(fig)
    plt.ioff()

    return output.T


def _fmt(val):
    """ data formatter for gfiles, which doesnt follow normal conventions..."""
    try:
        temp = '0{: 0.8E}'.format(float(val)*10)
        out = ''.join([temp[1], temp[0], temp[3], temp[2], temp[4:]])
    except TypeError:
        out = ''
        idx = 0
        for i in val:
            out += _fmt(i)
            idx += 1
            if (idx == 5):
                out += '\n'
                idx = 0
        if (idx != 0):
            out += '\n'
    return out
