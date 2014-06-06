import scipy 
import scipy.interpolate
import warnings
import time
import core
#import trispline

def gfile(obj, tin, nw=None, nh=None, shot=None, name=None, tunit = 'ms', title='EQTOOLS'):
 
    if shot is None:
        shot = obj._shot

    timeConvertDict = {'ms':1000.,'s':1.}
    stin = str(int(float(tin)*timeConvertDict[tunit]/timeConvertDict[obj._defaultUnits['_time']]))

    if name is None:
        name = 'g'+str(shot)+'.'+stin

    if nw is None:
        nw = len(obj.getRGrid())

    if nh is None:
        nh = len(obj.getZGrid())

    if len(title) > 10:
        raise ValueError('title is too long')

    header = title+ (11-len(title))*' '+ \
                         time.strftime('%m/%d/%Y')+ \
                         '   '+str(shot)+' '+ stin + tunit
                     
    header = header + (51-len(header))*' '+ '3 '+str(nw)+' '+str(nh)+'\n'
   
    rgrid = scipy.linspace(obj.getRGrid()[0],obj.getRGrid()[-1],nw)
    zgrid = scipy.linspace(obj.getZGrid()[0],obj.getZGrid()[-1],nh)
    rgrid,zgrid = scipy.meshgrid(rgrid,zgrid)
    print(header)

    gfiler =open(name, 'wb')
    gfiler.write(header)
    
    gfiler.write(_fmt([obj.getRGrid()[-1]-obj.getRGrid()[0],
                       obj.getZGrid()[-1]-obj.getZGrid()[0],
                       0.,
                       obj.getRGrid()[0],
                       obj.getZGrid()[-1]/2.+obj.getZGrid()[0]/2.]))
    
    gfiler.write(_fmt([obj.getMagRSpline()(tin),
                       obj.getMagZSpline()(tin),
                       -1*obj.getCurrentSign()*obj._getPsi0Spline()(tin),
                       -1*obj.getCurrentSign()*obj._getLCFSPsiSpline()(tin),
                       0.])) # need to get bzero getter...      

    if obj._tricubic:
        temp = scipy.interpolate.interp1d(obj.getTimeBase(),
                                          obj.getIpCalc(),
                                          kind='cubic',
                                          bounds_error=False)
        Ip = temp(tin)
    else:
        idx = obj._getNearestIdx(obj.getTimeBase(),tin)
        Ip = obj.getIpCalc()[idx]
                                          
    gfiler.write(_fmt([Ip,
                       -1*obj.getCurrentSign()*obj._getPsi0Spline()(tin),
                       0.,
                       obj.getMagRSpline()(tin),
                       0.]))
    gfiler.write(_fmt([obj.getMagZSpline()(tin),
                       0.,
                       -1*obj.getCurrentSign()*obj._getLCFSPsiSpline()(tin),
                       0.,
                       0.]))
    
    pts0 = scipy.linspace(0.,1.,obj.getRGrid().size) #find original nw
    pts1 = scipy.linspace(0.,1.,nw)
    
    # this needs to be time mapped (sigh)
    if not obj._tricubic: 

        for i in [obj.getF(),
                  obj.getFluxPres(),
                  obj.getFFPrime(),
                  obj.getPPrime()]:

            temp = scipy.interpolate.interp1d(pts0,
                                              scipy.atleast_2d(i)[idx],
                                              kind='nearest',
                                              bounds_error=False)
            gfiler.write(_fmt(temp(pts1).ravel()))



    gfiler.write(_fmt(-1*obj.getCurrentSign()*obj.rz2psi(rgrid,
                                                         zgrid,
                                                         tin).ravel())) #spline with new rz grid

    if not obj._tricubic:
        temp = scipy.interpolate.interp1d(pts0,
                                          scipy.atleast_2d(obj.getQProfile())[idx],
                                          kind='nearest',
                                          bounds_error=False)


        gfiler.write(_fmt(temp(pts1).ravel()))
        #gfiler.write(fmt([2*

    gfiler.close()
    #        nbbs,limtr #luckily these are easy
    #        RLCFS,ZCLFS  #this will require some work need to recalc this
    #        crosssection
    #else:
    #    dump
            
    
def _fmt(val):
    """ data formatter for gfiles, which doesnt follow normal conventions..."""
    try:
        temp = '0{: 0.8E}'.format(float(val)*10)
        out =''.join([temp[1],temp[0],temp[3],temp[2],temp[4:]])
    except TypeError:
        out = ''
        idx = 0
        for i in val:
            out += _fmt(i)
            idx += 1
            if (idx == 5):
                out+='\n'
                idx = 0
        if (idx != 0):
            out+='\n'
    return out
