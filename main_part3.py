import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import scipy.optimize as sci
from scipy import interpolate
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
import os

#niceValue = os.nice(19)	#  nice: value in [-20 important, 19 unimportant]
start_time = time.monotonic()
##########################################################################################################################
########## P A R T   3  : ANALYSE THE PEAKS OF TARGETS 88, 204, 435 #####################################################
# https://lmfit.github.io/lmfit-py/
NAME = 'EXTRACTION'
EXTRACTIONS_Lya = 'Lya_selected_ALL_%s.fits'%(NAME)
extractions_Lya = pyfits.open(EXTRACTIONS_Lya)
DATA_Lya = extractions_Lya[1].data
EXTRACTIONS_HeII = 'HeII_selected_ALL_%s.fits'%(NAME)
extractions_HeII = pyfits.open(EXTRACTIONS_HeII)
DATA_HeII = extractions_HeII[1].data

# possible targets:
iden_str = ['88', '204', '435'] # string version
iden_int = [88, 204, 435] # integer version
REDSHIFT = [2.9541607, 3.1357558, 3.7247474]
Ly_alpha_rest = 1215.67
HeII = 1640.42
z_min = 2.86
z_max = 4.7

rest_wavelen_HeII_88 = DATA_HeII.field('rest_wavelen_He-II_88')
flux_HeII_88 = DATA_HeII.field('flux_He-II_88')
noise_HeII_88 = DATA_HeII.field('noise_He-II_88')
rest_wavelen_HeII_204 = DATA_HeII.field('rest_wavelen_He-II_204')
flux_HeII_204  = DATA_HeII.field('flux_He-II_204')
noise_HeII_204  = DATA_HeII.field('noise_He-II_204')
rest_wavelen_HeII_435 = DATA_HeII.field('rest_wavelen_He-II_435')
flux_HeII_435 = DATA_HeII.field('flux_He-II_435')
noise_HeII_435 = DATA_HeII.field('noise_He-II_435')

rest_wavelen_HeII = [rest_wavelen_HeII_88, rest_wavelen_HeII_204, rest_wavelen_HeII_435]
flux_HeII = [flux_HeII_88, flux_HeII_204, flux_HeII_435]
noise_HeII = [noise_HeII_88, noise_HeII_204, noise_HeII_435]

FWHM_HeII_1 = 1.5
FWHM_HeII_2 = 1.
Delta_wavelen_HeII = 3  # from plots
Delta_v_HeII = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII) / HeII  # in km/s

#for i in range(len(iden_str)):
def Gauss_double(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
    #return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * np.exp(-(x - cen1) ** 2 / (2. * sigma1 ** 2)) + amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * np.exp(-(x - cen2) ** 2 / (2. * sigma2 ** 2))
    return amp1 * np.exp(-(x - cen1) ** 2 / (2. * sigma1 ** 2)) + amp2 * np.exp(-(x - cen2) ** 2 / (2. * sigma2 ** 2))

def objective88(params, x, data):
    amp1 = params['amp1_%s' % (iden_str[0])].value   # larger peak
    cen1 = params['cen1_%s' % (iden_str[0])].value
    sigma1 = params['sigma1_%s' % (iden_str[0])].value
    amp2= params['amp2_%s' % (iden_str[0])].value   # smaller peak
    cen2 = params['cen2_%s' % (iden_str[0])].value
    sigma2 = params['sigma2_%s' % (iden_str[0])].value
    model = Gauss_double(x, amp1, cen1, sigma1, amp2, cen2, sigma2)
    return model - data

def objective204(params, x, data):
    amp1 = params['amp1_%s' % (iden_str[1])].value   # larger peak
    cen1 = params['cen1_%s' % (iden_str[1])].value
    sigma1 = params['sigma1_%s' % (iden_str[1])].value
    amp2= params['amp2_%s' % (iden_str[1])].value # smaller peak
    cen2 = params['cen2_%s' % (iden_str[1])].value
    sigma2 = params['sigma2_%s' % (iden_str[1])].value
    model = Gauss_double(x, amp1, cen1, sigma1, amp2, cen2, sigma2)
    return model - data

def objective435(params, x, data):
    amp1 = params['amp1_%s' % (iden_str[2])].value   # larger peak
    cen1 = params['cen1_%s' % (iden_str[2])].value
    sigma1 = params['sigma1_%s' % (iden_str[2])].value
    amp2= params['amp2_%s' % (iden_str[2])].value   # smaller peak
    cen2 = params['cen2_%s' % (iden_str[2])].value
    sigma2 = params['sigma2_%s' % (iden_str[2])].value
    model = Gauss_double(x, amp1, cen1, sigma1, amp2, cen2, sigma2)
    return model - data

G2_model = Model(Gauss_double, nan_policy='propagate')
    #G2_model.set_param_hint('amp1', min=2.5)
    #G2_model.set_param_hint('amp2', min=1.8)
    #G2_model.set_param_hint('cen1', min=HeII - Delta_wavelen_HeII, max=HeII)
    #G2_model.set_param_hint('cen2', min=HeII - (Delta_wavelen_HeII + 1), max=HeII)
    #G2_model.set_param_hint('sigma1', max=FWHM_HeII_1 /(2 * np.sqrt(2*np.log(2))))
    #G2_model.set_param_hint('sigma2', max=FWHM_HeII_2 /(2 * np.sqrt(2*np.log(2))))
        # FWHM = 2 sqrt(2 * ln(2)) * sigma
    #params_HeII = G2_model.make_params() # G2_model.make_params(amp1 = 100., sigma1=2, cen1=1638.) - but what about the set_param_hint?


# TARGET 88:

FWHM_HeII_1_88 = 1.
FWHM_HeII_2_88 = 1.5
Delta_wavelen_HeII_88 = 2  # from plots
Delta_v_HeII_88 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_88) / HeII  # in km/s
ind_max_flux_88 = np.argmax(flux_HeII_88)
params_He88 = Parameters()
params_He88.add('amp1_%s' % (iden_str[0]), value=max(flux_HeII_88), min=15)
params_He88.add('cen1_%s' % (iden_str[0]), value=1639.5, min=HeII - Delta_wavelen_HeII_88, max=HeII)
params_He88.add('sigma1_%s' % (iden_str[0]), value=3, min=0.01, max=FWHM_HeII_1_88 /(2 * np.sqrt(2*np.log(2))))
params_He88.add('amp2_%s' % (iden_str[0]), value=10, min=10)
params_He88.add('cen2_%s' % (iden_str[0]), value=1637.5, min=HeII - (Delta_wavelen_HeII_88 + 1.5), max=HeII)
params_He88.add('sigma2_%s' % (iden_str[0]), value=3, min=0.01, max=FWHM_HeII_2_88 /(2 * np.sqrt(2*np.log(2))))
x = rest_wavelen_HeII_88
data = flux_HeII_88
minner88 = Minimizer(objective88, params_He88, fcn_args=(x, data))
#print(dir(minner88)) # associated variables
result_HeII88 = minner88.minimize()
final_HeII_88 = flux_HeII[0] + result_HeII88.residual

plt.figure()
plt.step(rest_wavelen_HeII[0], flux_HeII[0], 'b', label='flux')
plt.step(rest_wavelen_HeII[0], noise_HeII[0], 'k', label='noise')
plt.plot(rest_wavelen_HeII[0], final_HeII_88, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[0], str(REDSHIFT[0])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[0]))
plt.clf()

# TARGET 204:
FWHM_HeII_1_204 = 0.6
FWHM_HeII_2_204 = 0.4
Delta_wavelen_HeII_204 = 1  # from plots
Delta_v_HeII_204 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_204) / HeII  # in km/s
ind_max_flux_204 = np.argmax(flux_HeII_204)
# print(flux_HeII_204) # index: peak 1: 21 // peak 2: 19
params_He204 = Parameters()
params_He204.add('amp1_%s' % (iden_str[1]), value=max(flux_HeII_204), min=2.5, max=3)
params_He204.add('cen1_%s' % (iden_str[1]), value=rest_wavelen_HeII_204[ind_max_flux_204], min=HeII - Delta_wavelen_HeII_204, max=HeII)
params_He204.add('sigma1_%s' % (iden_str[1]), value=1, min=0.01, max=FWHM_HeII_1_204 /(2 * np.sqrt(2*np.log(2))))
params_He204.add('amp2_%s' % (iden_str[1]), value=flux_HeII_204[19], min=1.7, max=2)
params_He204.add('cen2_%s' % (iden_str[1]), value=rest_wavelen_HeII_204[19], min=HeII - (Delta_wavelen_HeII_204 + 1.5), max=HeII)
params_He204.add('sigma2_%s' % (iden_str[1]), value=1, min=0.01, max=FWHM_HeII_2_204 /(2 * np.sqrt(2*np.log(2))))
x = rest_wavelen_HeII_204
data = flux_HeII_204
minner204 = Minimizer(objective204, params_He204, fcn_args=(x, data))
result_HeII204 = minner204.minimize()
final_HeII_204 = flux_HeII[1] + result_HeII204.residual

plt.figure()
plt.step(rest_wavelen_HeII_204, flux_HeII_204, 'b', label='flux')
plt.step(rest_wavelen_HeII_204, noise_HeII_204, 'k', label='noise')
plt.plot(rest_wavelen_HeII_204, final_HeII_204, 'r')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[1], str(REDSHIFT[1])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[1]))
plt.clf()

# TARGET 435:
FWHM_HeII_1_435 = 1.5
FWHM_HeII_2_435 = 0.5
Delta_wavelen_HeII_435 = 2.8  # from plots
Delta_v_HeII_435 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_435) / HeII  # in km/s
#print(flux_HeII_435) # need index 14 & 17 for peak 2 and 1
params_He435 = Parameters()
params_He435.add('amp1_%s' % (iden_str[2]), value=flux_HeII_435[17])#, min=4, max=6)
params_He435.add('cen1_%s' % (iden_str[2]), value=rest_wavelen_HeII_435[17])#, min=HeII - Delta_wavelen_HeII_435, max=HeII)
params_He435.add('sigma1_%s' % (iden_str[2]), value=1, min=0.01, max=FWHM_HeII_1_435 /(2 * np.sqrt(2*np.log(2))))
params_He435.add('amp2_%s' % (iden_str[2]), value=flux_HeII_435[14], min=1.8, max=2.5)
params_He435.add('cen2_%s' % (iden_str[2]), value=rest_wavelen_HeII_435[14], min=HeII - (Delta_wavelen_HeII_435 + 1.5), max=HeII)
params_He435.add('sigma2_%s' % (iden_str[2]), value=1, min=0.01, max=FWHM_HeII_2_435 /(2 * np.sqrt(2*np.log(2))))
x = rest_wavelen_HeII_435
data = flux_HeII_435
minner435 = Minimizer(objective435, params_He435, fcn_args=(x, data))
result_HeII435 = minner435.minimize()
final_HeII_435 = flux_HeII[2] + result_HeII435.residual
ModelResult435 = ModelResult(G2_model, params_He435, weights=True, nan_policy='propagate')

plt.figure()
plt.step(rest_wavelen_HeII_435, flux_HeII_435, 'b', label='flux')
plt.step(rest_wavelen_HeII_435, noise_HeII_435, 'k', label='noise')
plt.plot(rest_wavelen_HeII_435, final_HeII_435, 'r', label='fit')
#TODO: ModelResult435.plot_residuals()
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1633, 1643)
plt.axvline(x=HeII, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[2], str(REDSHIFT[2])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[2]))
plt.clf()


print('finished part 3')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))