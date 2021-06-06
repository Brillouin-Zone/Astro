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
# P A R T   6  : CREATE LINE-RATIO-PLOTS ACCORDING TO NANAYAKKARA FOR TARGETS 88, 204, 435, 22429 #########################
        # (x,y) = (SiIII] / CIII] , CIII] / OIII])   and (OIII] / HeII, CIII] / HeII)
        # latter one to distinguish between AGN- and stellar-ionizing photons

# 1) extract data from tables created in main_part2
###################################################
NAME = 'EXTRACTION'
iden_str = ['88', '204', '435', '22429', '538', '23124', '5199'] # string version
iden_int = [88, 204, 435, 22429, 538, 23124, 5199] # integer version
REDSHIFT = [2.9541607, 3.1357558, 3.7247474, 2.9297342, 4.1764603, 3.59353921008408, 3.063]

EXTRACTIONS_HeII = 'HeII_selected_ALL_%s.fits'%(NAME)
extractions_HeII = pyfits.open(EXTRACTIONS_HeII)
DATA_HeII = extractions_HeII[1].data

EXTRACTIONS_CIII = 'CIII_selected_ALL_%s.fits'%(NAME)
extractions_CIII = pyfits.open(EXTRACTIONS_CIII)
DATA_CIII = extractions_CIII[1].data

EXTRACTIONS_OIII = 'OIII_selected_ALL_%s.fits'%(NAME)
extractions_OIII = pyfits.open(EXTRACTIONS_OIII)
DATA_OIII = extractions_OIII[1].data

EXTRACTIONS_SiIII = 'SiIII_selected_ALL_%s.fits'%(NAME)
extractions_SiIII = pyfits.open(EXTRACTIONS_SiIII)
DATA_SiIII = extractions_SiIII[1].data

# 2) define variables
###################################################
rest_wavelen_HeII_88 = DATA_HeII.field('rest_wavelen_He-II_88')
flux_HeII_88 = DATA_HeII.field('flux_He-II_88')
noise_HeII_88 = DATA_HeII.field('noise_He-II_88')
rest_wavelen_HeII_118 = DATA_HeII.field('rest_wavelen_He-II_118')
flux_HeII_118 = DATA_HeII.field('flux_He-II_118')
noise_HeII_118 = DATA_HeII.field('noise_He-II_118')
rest_wavelen_HeII_131 = DATA_HeII.field('rest_wavelen_He-II_131')
flux_HeII_131 = DATA_HeII.field('flux_He-II_131')
noise_HeII_131 = DATA_HeII.field('noise_He-II_131')
rest_wavelen_HeII_218 = DATA_HeII.field('rest_wavelen_He-II_218')
flux_HeII_218 = DATA_HeII.field('flux_He-II_218')
noise_HeII_218 = DATA_HeII.field('noise_He-II_218')
rest_wavelen_HeII_435 = DATA_HeII.field('rest_wavelen_He-II_435')
flux_HeII_435 = DATA_HeII.field('flux_He-II_435')
noise_HeII_435 = DATA_HeII.field('noise_He-II_435')
rest_wavelen_HeII_7876 = DATA_HeII.field('rest_wavelen_He-II_7876')
flux_HeII_7876 = DATA_HeII.field('flux_He-II_7876')
noise_HeII_7876 = DATA_HeII.field('noise_He-II_7876')
rest_wavelen_HeII = [rest_wavelen_HeII_88, rest_wavelen_HeII_118, rest_wavelen_HeII_131, rest_wavelen_HeII_218,
                     rest_wavelen_HeII_435, rest_wavelen_HeII_7876]
flux_HeII = [flux_HeII_88, flux_HeII_118, flux_HeII_131, flux_HeII_218, flux_HeII_435, flux_HeII_7876]
noise_HeII = [noise_HeII_88, noise_HeII_118, noise_HeII_131, noise_HeII_218, noise_HeII_435, noise_HeII_7876]

rest_wavelen_CIII_88 = DATA_CIII.field('rest_wavelen_C-III_88')
flux_CIII_88 = DATA_CIII.field('flux_C-III_88')
noise_CIII_88 = DATA_CIII.field('noise_C-III_88')
rest_wavelen_CIII_118 = DATA_CIII.field('rest_wavelen_C-III_118')
flux_CIII_118 = DATA_CIII.field('flux_C-III_118')
noise_CIII_118 = DATA_CIII.field('noise_C-III_118')
rest_wavelen_CIII_131 = DATA_CIII.field('rest_wavelen_C-III_131')
flux_CIII_131 = DATA_CIII.field('flux_C-III_131')
noise_CIII_131 = DATA_CIII.field('noise_C-III_131')
rest_wavelen_CIII_218 = DATA_CIII.field('rest_wavelen_C-III_218')
flux_CIII_218 = DATA_CIII.field('flux_C-III_218')
noise_CIII_218 = DATA_CIII.field('noise_C-III_218')
rest_wavelen_CIII_435 = DATA_CIII.field('rest_wavelen_C-III_435')
flux_CIII_435 = DATA_CIII.field('flux_C-III_435')
noise_CIII_435 = DATA_CIII.field('noise_C-III_435')
rest_wavelen_CIII_7876 = DATA_CIII.field('rest_wavelen_C-III_7876')
flux_CIII_7876 = DATA_CIII.field('flux_C-III_7876')
noise_CIII_7876 = DATA_CIII.field('noise_C-III_7876')
rest_wavelen_CIII = [rest_wavelen_CIII_88, rest_wavelen_CIII_118, rest_wavelen_CIII_131, rest_wavelen_CIII_218,
                     rest_wavelen_CIII_435, rest_wavelen_CIII_7876]
flux_CIII = [flux_CIII_88, flux_CIII_118, flux_CIII_131, flux_CIII_218, flux_CIII_435, flux_CIII_7876]
noise_CIII = [noise_CIII_88, noise_CIII_118, noise_CIII_131, noise_CIII_218, noise_CIII_435, noise_CIII_7876]

rest_wavelen_OIII_88 = DATA_OIII.field('rest_wavelen_O-III_88')
flux_OIII_88 = DATA_OIII.field('flux_O-III_88')
noise_OIII_88 = DATA_OIII.field('noise_O-III_88')
rest_wavelen_OIII_118 = DATA_OIII.field('rest_wavelen_O-III_118')
flux_OIII_118 = DATA_OIII.field('flux_O-III_118')
noise_OIII_118 = DATA_OIII.field('noise_O-III_118')
rest_wavelen_OIII_131 = DATA_OIII.field('rest_wavelen_O-III_131')
flux_OIII_131 = DATA_OIII.field('flux_O-III_131')
noise_OIII_131 = DATA_OIII.field('noise_O-III_131')
rest_wavelen_OIII_218 = DATA_OIII.field('rest_wavelen_O-III_218')
flux_OIII_218 = DATA_OIII.field('flux_O-III_218')
noise_OIII_218 = DATA_OIII.field('noise_O-III_218')
rest_wavelen_OIII_435 = DATA_OIII.field('rest_wavelen_O-III_435')
flux_OIII_435 = DATA_OIII.field('flux_O-III_435')
noise_OIII_435 = DATA_OIII.field('noise_O-III_435')
rest_wavelen_OIII_7876 = DATA_OIII.field('rest_wavelen_O-III_7876')
flux_OIII_7876 = DATA_OIII.field('flux_O-III_7876')
noise_OIII_7876 = DATA_OIII.field('noise_O-III_7876')
rest_wavelen_OIII = [rest_wavelen_OIII_88, rest_wavelen_OIII_118, rest_wavelen_OIII_131, rest_wavelen_OIII_218,
                     rest_wavelen_OIII_435, rest_wavelen_OIII_7876]
flux_OIII = [flux_OIII_88, flux_OIII_118, flux_OIII_131, flux_OIII_218, flux_OIII_435, flux_OIII_435, flux_OIII_7876]
noise_OIII = [noise_OIII_88, noise_OIII_118, noise_OIII_131, noise_OIII_218, noise_OIII_435, noise_OIII_7876]

rest_wavelen_SiIII_88 = DATA_SiIII.field('rest_wavelen_Si-III_88')
flux_SiIII_88 = DATA_SiIII.field('flux_Si-III_88')
noise_SiIII_88 = DATA_SiIII.field('noise_Si-III_88')
rest_wavelen_SiIII_118 = DATA_SiIII.field('rest_wavelen_Si-III_118')
flux_SiIII_118 = DATA_SiIII.field('flux_Si-III_118')
noise_SiIII_118 = DATA_SiIII.field('noise_Si-III_118')
rest_wavelen_SiIII_131 = DATA_SiIII.field('rest_wavelen_Si-III_131')
flux_SiIII_131 = DATA_SiIII.field('flux_Si-III_131')
noise_SiIII_131 = DATA_SiIII.field('noise_Si-III_131')
rest_wavelen_SiIII_218 = DATA_SiIII.field('rest_wavelen_Si-III_218')
flux_SiIII_218 = DATA_SiIII.field('flux_Si-III_218')
noise_SiIII_218 = DATA_SiIII.field('noise_Si-III_218')
rest_wavelen_SiIII_435 = DATA_SiIII.field('rest_wavelen_Si-III_435')
flux_SiIII_435 = DATA_SiIII.field('flux_Si-III_435')
noise_SiIII_435 = DATA_SiIII.field('noise_Si-III_435')
rest_wavelen_SiIII_7876 = DATA_SiIII.field('rest_wavelen_Si-III_7876')
flux_SiIII_7876 = DATA_SiIII.field('flux_Si-III_7876')
noise_SiIII_7876 = DATA_SiIII.field('noise_Si-III_7876')
rest_wavelen_SiIII = [rest_wavelen_SiIII_88, rest_wavelen_SiIII_118, rest_wavelen_SiIII_131, rest_wavelen_SiIII_218,
                      rest_wavelen_SiIII_435, rest_wavelen_SiIII_7876]
flux_SiIII = [flux_SiIII_88, flux_SiIII_118, flux_SiIII_131, flux_SiIII_218, flux_SiIII_435 , flux_SiIII_7876]
noise_SiIII = [noise_SiIII_88, noise_SiIII_118, noise_SiIII_131, noise_SiIII_218, noise_SiIII_435, noise_SiIII_7876]

# 3) create the line-ratio plot (x,y) = (SiIII] / CIII], CIII] / OIII])          (x,y) = (OIII] / HeII, CIII] / HeII)
##################################################################################################################################################
# 88, 118, 131, 218, 435, 7876
index_max_flux_HeII = [np.argmax(flux_HeII_88), np.argmax(flux_HeII_118), np.argmax(flux_HeII_131),
                       np.argmax(flux_HeII_218), 17, np.argmax(flux_HeII_7876)]
index_max_flux_CIII = [13, np.argmax(flux_CIII_118), np.argmax(flux_CIII_131), 20,
                       np.argmax(flux_CIII_435), np.argmax(flux_CIII_7876)]
index_max_flux_OIII = [8, np.argmax(flux_OIII_118), np.argmax(flux_OIII_131), np.argmax(flux_OIII_218),
                       np.argmax(flux_OIII_435), np.argmax(flux_OIII_7876)]
index_max_flux_SiIII = [0, np.argmax(flux_SiIII_118)] # rest not important

SiC = flux_SiIII[1][index_max_flux_SiIII[1]] / flux_CIII[1][index_max_flux_CIII[1]]
CO = flux_CIII[1][index_max_flux_CIII[1]] / flux_OIII[1][index_max_flux_OIII[1]]
OHe = [0, 0, 0, 0, 0, 0]
CHe = [0, 0, 0, 0, 0, 0, 0]
for i in range(len(index_max_flux_HeII)):
    OHe[i] = flux_OIII[i][index_max_flux_OIII[i]] / flux_HeII[i][index_max_flux_HeII[i]]
    CHe[i] = flux_CIII[i][index_max_flux_CIII[i]] / flux_HeII[i][index_max_flux_HeII[i]]

figNr = 0

figure = plt.figure(figNr)
# only for 118
plt.plot(np.log10(SiC), np.log10(CO), '+b', label='118', markersize=13)
#plt.plot(SiC[1], CO[1], 'ok', label='204')
#plt.plot(SiC[2], CO[2], 'og', label='435')
#plt.plot(SiC[3], CO[3], 'or', label='22429')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'log10(SiIII]/CIII])')
plt.ylabel(r'log10(CIII]/OIII])')
plt.savefig('plots/ALL_PART6/lineratio_SiC_CO.pdf')
plt.savefig('MAIN_LATEX/PLOTS/lineratio_SiC_CO.pdf')
plt.clf()
figNr += 1

figure = plt.figure(figNr)
# for 88, 118, 131, 218, 435 and 7876
plt.plot(np.log10(OHe[0]), np.log10(CHe[0]), '+b', label='88', markersize=13)
plt.plot(np.log10(OHe[1]), np.log10(CHe[1]), '+k', label='118', markersize=13)
plt.plot(np.log10(OHe[2]), np.log10(CHe[2]), '+g', label='131', markersize=13)
plt.plot(np.log10(OHe[3]), np.log10(CHe[3]), '+m', label='218', markersize=13)
plt.plot(np.log10(OHe[4]), np.log10(CHe[4]), '+c', label='435', markersize=13)
plt.plot(np.log10(OHe[5]), np.log10(CHe[5]), '+y', label='7876', markersize=13)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'log10(OIII]/HeII])')
plt.ylabel(r'log10(CIII]/HeII])')
plt.savefig('plots/ALL_PART6/lineratio_OHe_CHe.pdf')
plt.savefig('MAIN_LATEX/PLOTS/lineratio_OHe_CHe.pdf')
plt.clf()
figNr += 1

# END
########################
print('finished part 6')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
