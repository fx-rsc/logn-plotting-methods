import matplotlib as mpl
mpl.use('Agg')
import pylab as pl

import numpy as np


def norm_pdf(x, mu, sig):
    S = - (x - mu)**2/(2*sig**2)
    f_x = 1./(sig*np.sqrt(2*np.pi)) * np.exp(S)
    return f_x

def lognorm_pdf(x, mu, sig):
    S = - (np.log(x) - mu)**2/(2*sig**2)
    f_x = 1./(x*sig*np.sqrt(2*np.pi)) * np.exp(S)
    return f_x


mus =  [1., 1., 1.]
sigs = [1., 1.2, 1.5]


from scipy.stats import lognorm
from scipy.stats import norm




pl.clf()
pl.figure()
xs = np.arange(0.01,70,0.01)


for mu, sig in zip(mus, sigs):
    pl.plot(np.log(xs), lognorm_pdf(xs, mu, sig))
    pl.plot(np.log(xs), norm_pdf(np.log(xs), mu, sig))
    #vs = norm.rvs(1., size=10000)
    #print np.mean(vs)
    #pl.plot(xs, lognorm.pdf(xs, 1., loc=1., scale=1.))
    #pl.hist(np.exp(vs), normed=True, histtype='stepfilled', alpha=0.2, bins=100)
    #pl.plot(xs, norm_pdf(np.log(xs), mu, sig))

    
#pl.xscale('log')
# pl.savefig("plot.png")


# pl.clf()
# vs = norm.rvs(1., size=10000)
# pl.hist(np.exp(vs), normed=True, histtype='stepfilled', alpha=0.2, bins=100)
# pl.plot(xs, norm.pdf(xs,1.))
# pl.savefig('plot.png')




# pl.clf()
# vs = lognorm.rvs(1., size=10000)
# pl.hist(np.log(vs), normed=True, histtype='stepfilled', alpha=0.2, bins=100)
# xs = np.arange(-4,4,0.01)
# pl.plot(xs, norm.pdf(xs))
# pl.savefig('plot.png')


# pl.clf()
# xs = np.arange(1.01,70,0.1)
# #pl.plot(xs, lognorm.pdf(xs,1))
# #pl.plot(xs, norm.pdf(np.exp(xs)))
# pl.plot(xs, lognorm.pdf(np.log(xs),1))
# #pl.xscale('log')

pl.savefig('plot.png')
