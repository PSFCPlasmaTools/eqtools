import scipy 
import warnings
import time
import core


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

    print(header)

    gfiler =open(name, 'wb')
    gfiler.write(header)
    print(fmt([obj.getRGrid()[-1]-obj.getRGrid()[0],
               obj.getZGrid()[-1]-obj.getZGrid()[0],
               0.,
               obj.getRGrid()[0],
               obj.getZGrid()[-1]/2.+obj.getZGrid()[0]/2.]))

    
    gfiler.write(fmt([obj.getRGrid()[-1]-obj.getRGrid()[0],
                      obj.getZGrid()[-1]-obj.getZGrid()[0],
                      0.,
                      obj.getRGrid()[0],
                      obj.getZGrid()[-1]/2.+obj.getZGrid()[0]/2.]))
    gfiler.write(fmt([obj.getMagRSpline()(tin),
                      obj.getMagZSpline()(tin),
                      obj._getPsi0Spline()(tin),
                      obj._getLCFSPsiSpline()(tin),
                      0.])) # need to get bzero getter...  
    #gfiler.write(fmt([obj.getIpCalc(tin),
   #                   obj.getPsi0Spline(tin),
   #                   0.,
   #                   obj.getMagRSpline(tin),
   #                   0.]))
    gfiler.write(fmt([obj.getMagZSpline()(tin),
                      0.,
                      obj._getLCFSPsiSpline()(tin),
                      0.,
                      0.]))
    gfiler.write(fmt([obj.getMagZSpline()(tin),
                      0.,
                      obj._getLCFSPsiSpline()(tin),
                      0.,
                      0.]))
    gfiler.close()

    #dump = [obj.getFpol().flatten(),
    #        obj.getFluxPres().flatten(),
    #        obj.getFFPrime().flatten()]
    #if obj._tricub:
    #    dump
            
    
def fmt(val):
    try:
        temp = '0{: 0.8E}'.format(val*10)
        out =''.join([temp[1],temp[0],temp[3],temp[2],temp[4:]])
    except ValueError:
        out = ''
        idx = 0
        for i in val:
            out += fmt(i)
            idx += 1
            if (idx == 5):
                out+='\n'
                idx = 0

    return out
