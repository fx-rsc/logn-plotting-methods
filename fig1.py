
import matplotlib as mpl
mpl.use('Agg')

import pylab as pl
import numpy as np


def lognorm_pdf(x, mu, sig):
    S = - (np.log(x) - mu)**2/(2.*sig**2)
    f_x = 1./(x*sig*np.sqrt(2.*np.pi)) * np.exp(S)
    return f_x

from scipy.stats import norm, lognorm

mu = 2.
sigma = 1.

xs = lognorm.rvs(s=sigma, scale=np.exp(mu), size=500000)

fig, ((ax1, ax2),(ax3,ax4)) = pl.subplots(2,2)
axs = [ax1,ax2,ax3,ax4]

fig.set_size_inches(8,6.)

bin_min, bin_max = 0,80
bin_N = 1000
s=0.01

xr = np.arange(bin_min, bin_max+s, s)


# ------- ax1: linear x-scale ---------
ax1.set_xlim(0,bin_max)
ax1.hist(xs, np.linspace(bin_min, bin_max, bin_N), normed=True)
ax1.plot(xr, lognorm_pdf(xr, mu, sigma), 'r')

fs, floc, fscale = lognorm.fit(xs, floc=0)
f_rv = lognorm(fs, loc=0, scale=fscale)

ax1.plot(xr, f_rv.pdf(xr), lw=2, color='grey',
         linestyle='--', dashes=(1, 1.6))



# ------- ax2: log vals on linear x-scale no normalization ---------
ax2.plot(np.log10(xr), lognorm_pdf(xr, mu, sigma), 'r')
_, bins, _ = ax2.hist(np.log10(xs), bins=bin_N, normed=True)


floc, fscale = norm.fit(np.log10(xs))
f_rv = norm(loc=floc, scale=fscale)

xsp = np.arange(-2, 3, 0.01)
xr = np.arange(10.**(bins[0]), 10.**(bins[-1])*1.1, s)
ax2.plot(xsp, f_rv.pdf(xsp), lw=2, color='grey',
         linestyle='--', dashes=(1, 1.6))

fs, floc, fscale = lognorm.fit(xs, floc=0)
f_rv = lognorm(fs, loc=0, scale=fscale)

# ax2.plot(np.log10(xr), f_rv.pdf(xr), lw=2, color='green',
#          linestyle='--', dashes=(1, 1.6))
ax2.plot(xsp, f_rv.pdf(10**xsp), lw=2, color='green',
         linestyle='--', dashes=(1, 1.6))



# ---------
counts, edges = np.histogram(xs, bins=np.linspace(bin_min, bin_max, bin_N),
                             density=True)
centers = (edges[1:] + edges[:-1])/2.
ax3.plot(np.log10(centers), counts, '.', color='k')
ax3.plot(np.log10(xr), lognorm_pdf(xr, mu, sigma), 'r')



ax1.set_xlabel('x')
ax2.set_xlabel('log(x)')
ax3.set_xlabel('x')
ax4.set_xlabel('x')

for ax in axs:
    ax.set_ylabel('f(x)')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

ax4.hist(xs, 10**np.linspace(-2,2,bin_N), normed=True)
ax4.set_xscale('log')
ax4.plot(xr, lognorm_pdf(xr, mu, sigma), 'r')


import os
fname = os.path.splitext(os.path.basename(__file__))[0]

pl.tight_layout()
pl.savefig("{}.png".format(fname), dpi=300, bbox_inches='tight')



