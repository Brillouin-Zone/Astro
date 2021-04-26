import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy.optimize as sci
from scipy import interpolate, integrate
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import scipy.constants as const

import astropy
from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.modeling import fitting
import scipy.ndimage as snd
from astropy.io import fits as pyfits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

import statistics
import random
from decimal import Decimal
from lmfit.model import Model, ModelResult
from lmfit import Parameters, minimize, fit_report, Minimizer, report_fit
import pandas as pd
import seaborn as sns
import time
from datetime import timedelta

start_time = time.monotonic()
##########################################################################################################################
# P A R T   5  : CREATE C-IV AND C-III] PEAK OF TARGETS 48, 53, 75, 76, 88, 118, 131, 185, 200, 204, 211, 218, 435, 5199, 7876, 22429
    # not : 53, 75, 218, 7876, 5199, 76, 200
    # unclear: 48, 131, 185, 211, 118
    # choose: 88, 204, 435, 22429

    # C-III] peak       :  arXiv:1902.05960v1 [astro-ph.GA]; Narayakkara
    # C-IV peak         :  arXiv:1911.09999 [astro-ph.GA]; Saxena

NAME = 'EXTRACTION'
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
IDEN = np.array([88, 204, 435, 22429])
C_IV = 1548.19
CIII_sf = 1908.73

rest_vac_wavelen = [[] for j in range(len(IDEN))]
flux_dens_tot = [[] for j in range(len(IDEN))]
flux_dens_err = [[] for j in range(len(IDEN))]

C_IV_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the C-IV peak
rest_wavelen_C_IV = [[] for j in range(len(IDEN))]
flux_C_IV = [[] for j in range(len(IDEN))]
noise_C_IV = [[] for j in range(len(IDEN))]

CIII_sf_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the CIII_sf peak
rest_wavelen_CIII_sf = [[] for j in range(len(IDEN))]
flux_CIII_sf = [[] for j in range(len(IDEN))]
noise_CIII_sf = [[] for j in range(len(IDEN))]

figNr = 0
for i in range(len(IDEN)):
    target_nr = str(IDEN[i])
    flux_dens_tot[i] = DATA.field('fluxdensity_total_%s' % (target_nr))  # extract the total-flux-density-column for each galaxy
    flux_dens_err[i] = DATA.field('fluxdensity_total_ERR_%s' % (target_nr))  # extract the total-flux-density-error-column for each galaxy
    rest_vac_wavelen[i] = DATA.field('rest_vac_wavelen_%s' % (target_nr))  # extract the rest-wavelength-column for each galaxy

    C_IV_indices.append(np.where((rest_vac_wavelen[i] > 1541) & (rest_vac_wavelen[i] < 1555)))
    flux_C_IV[i] = np.array(flux_dens_tot[i])[C_IV_indices[i]]
    noise_C_IV[i] = np.array(flux_dens_err[i])[C_IV_indices[i]]
    rest_wavelen_C_IV[i] = np.array(rest_vac_wavelen[i])[C_IV_indices[i]]

    CIII_sf_indices.append(np.where((rest_vac_wavelen[i] > 1902) & (rest_vac_wavelen[i] < 1916)))
    flux_CIII_sf[i] = np.array(flux_dens_tot[i])[CIII_sf_indices[i]]
    noise_CIII_sf[i] = np.array(flux_dens_err[i])[CIII_sf_indices[i]]
    rest_wavelen_CIII_sf[i] = np.array(rest_vac_wavelen[i])[CIII_sf_indices[i]]

    figure = plt.figure(figNr)
    # plt.figure(figNr)
    plt.step(rest_wavelen_C_IV[i], flux_C_IV[i], 'b', label='flux')
    plt.step(rest_wavelen_C_IV[i], noise_C_IV[i], 'k', label='noise')
    plt.axvline(x=C_IV, color='c')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel(r'wavelength in rest-frame $[\AA]$')
    plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
    plt.title('C-IV rest-peak of target %s' % (IDEN[i]))
    plt.savefig('plots/ALL_PART5/CIV_target_%s.pdf' % (target_nr))
    plt.clf()
    figNr += 1

    figure = plt.figure(figNr)
    # plt.figure(figNr)
    plt.step(rest_wavelen_CIII_sf[i], flux_CIII_sf[i], 'b', label='flux')
    plt.step(rest_wavelen_CIII_sf[i], noise_CIII_sf[i], 'k', label='noise')
    plt.axvline(x=CIII_sf, color='c')
    plt.grid(True)
    plt.legend(loc='best')
    plt.xlabel(r'wavelength in rest-frame $[\AA]$')
    plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
    plt.title('C-III] rest-peak of target %s' % (IDEN[i]))
    plt.savefig('plots/ALL_PART5/CIII_sf_target_%s.pdf' % (target_nr))
    plt.clf()
    figNr += 1


# END
########################
print('finished part 5')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))