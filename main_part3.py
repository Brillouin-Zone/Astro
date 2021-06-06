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
import lmfit
from lmfit.model import Model, ModelResult
from lmfit import Parameters, minimize, fit_report, Minimizer, report_fit
import time
from datetime import timedelta

start_time = time.monotonic()
######################################################################################################################
########## P A R T   3  : ANALYSE THE He II and Lya PEAKS OF TARGETS 88, 204, 435, 22429, 538, 5199, 23124 ###########
# Peaks: He-II             C-III] and C-IV, Si-III], O-III] in PART5
# Analysis includes: Gaussian fit with parameters and their standard deviations,
#                                   SNR of He-II peak, bootstrap experiment, FWHM
# SNR and integrated flux of Lya peak
# https://lmfit.github.io/lmfit-py/
NAME = 'EXTRACTION'
EXTRACTIONS_Lya = 'Lya_selected_ALL_%s.fits' % NAME
extractions_Lya = pyfits.open(EXTRACTIONS_Lya)
DATA_Lya = extractions_Lya[1].data
EXTRACTIONS_HeII = 'HeII_selected_ALL_%s.fits' % NAME
extractions_HeII = pyfits.open(EXTRACTIONS_HeII)
DATA_HeII = extractions_HeII[1].data

EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits' % NAME
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
rest_vac_wavelen_48 = DATA.field('rest_vac_wavelen_48')
rest_vac_wavelen_88 = DATA.field('rest_vac_wavelen_88')
rest_vac_wavelen_118 = DATA.field('rest_vac_wavelen_118')
rest_vac_wavelen_131 = DATA.field('rest_vac_wavelen_131')
rest_vac_wavelen_204 = DATA.field('rest_vac_wavelen_204')
rest_vac_wavelen_218 = DATA.field('rest_vac_wavelen_218')
rest_vac_wavelen_435 = DATA.field('rest_vac_wavelen_435')
rest_vac_wavelen_538 = DATA.field('rest_vac_wavelen_538')
rest_vac_wavelen_5199 = DATA.field('rest_vac_wavelen_5199')
rest_vac_wavelen_7876 = DATA.field('rest_vac_wavelen_7876')
rest_vac_wavelen_23124 = DATA.field('rest_vac_wavelen_23124')
rest_vac_wavelen_22429 = DATA.field('rest_vac_wavelen_22429')

iden_str = ['88', '204', '435', '22429', '538', '5199', '23124', '48', '118', '131', '7876', '218']
iden_int = [88, 204, 435, 22429, 538, 5199, 23124, 48, 118, 131, 7876, 218]
REDSHIFT = [2.9541607, 3.1357558, 3.7247474, 2.9297342, 4.1764603, 3.063, 3.59353921008408,
            2.9101489, 3.0024831, 3.0191996, 2.993115, 2.865628]
Ly_alpha_rest = 1215.67
HeII = 1640.42
z_min = 2.86
z_max = 4.7
N = 100

rest_wavelen_HeII_88 = DATA_HeII.field('rest_wavelen_He-II_88')
flux_HeII_88 = DATA_HeII.field('flux_He-II_88')
noise_HeII_88 = DATA_HeII.field('noise_He-II_88')
rest_wavelen_HeII_204 = DATA_HeII.field('rest_wavelen_He-II_204')
flux_HeII_204 = DATA_HeII.field('flux_He-II_204')
noise_HeII_204 = DATA_HeII.field('noise_He-II_204')
rest_wavelen_HeII_435 = DATA_HeII.field('rest_wavelen_He-II_435')
flux_HeII_435 = DATA_HeII.field('flux_He-II_435')
noise_HeII_435 = DATA_HeII.field('noise_He-II_435')
rest_wavelen_HeII_22429 = DATA_HeII.field('rest_wavelen_He-II_22429')
flux_HeII_22429 = DATA_HeII.field('flux_He-II_22429')
noise_HeII_22429 = DATA_HeII.field('noise_He-II_22429')
rest_wavelen_HeII_538 = DATA_HeII.field('rest_wavelen_He-II_538')
flux_HeII_538 = DATA_HeII.field('flux_He-II_538')
noise_HeII_538 = DATA_HeII.field('noise_He-II_538')
rest_wavelen_HeII_5199 = DATA_HeII.field('rest_wavelen_He-II_5199')
flux_HeII_5199 = DATA_HeII.field('flux_He-II_5199')
noise_HeII_5199 = DATA_HeII.field('noise_He-II_5199')
rest_wavelen_HeII_23124 = DATA_HeII.field('rest_wavelen_He-II_23124')
flux_HeII_23124 = DATA_HeII.field('flux_He-II_23124')
noise_HeII_23124 = DATA_HeII.field('noise_He-II_23124')
rest_wavelen_HeII_48 = DATA_HeII.field('rest_wavelen_He-II_48')
flux_HeII_48 = DATA_HeII.field('flux_He-II_48')
noise_HeII_48 = DATA_HeII.field('noise_He-II_48')
rest_wavelen_HeII_118 = DATA_HeII.field('rest_wavelen_He-II_118')
flux_HeII_118 = DATA_HeII.field('flux_He-II_118')
noise_HeII_118 = DATA_HeII.field('noise_He-II_118')
rest_wavelen_HeII_131 = DATA_HeII.field('rest_wavelen_He-II_131')
flux_HeII_131 = DATA_HeII.field('flux_He-II_131')
noise_HeII_131 = DATA_HeII.field('noise_He-II_131')
rest_wavelen_HeII_7876 = DATA_HeII.field('rest_wavelen_He-II_7876')
flux_HeII_7876 = DATA_HeII.field('flux_He-II_7876')
noise_HeII_7876 = DATA_HeII.field('noise_He-II_7876')
rest_wavelen_HeII_218 = DATA_HeII.field('rest_wavelen_He-II_218')
flux_HeII_218 = DATA_HeII.field('flux_He-II_218')
noise_HeII_218 = DATA_HeII.field('noise_He-II_218')
rest_wavelen_HeII = [rest_wavelen_HeII_88, rest_wavelen_HeII_204, rest_wavelen_HeII_435, rest_wavelen_HeII_22429,
                     rest_wavelen_HeII_538, rest_wavelen_HeII_5199, rest_wavelen_HeII_23124, rest_wavelen_HeII_48,
                     rest_wavelen_HeII_118, rest_wavelen_HeII_131, rest_wavelen_HeII_7876, rest_wavelen_HeII_218]
flux_HeII = [flux_HeII_88, flux_HeII_204, flux_HeII_435, flux_HeII_22429,
             flux_HeII_538, flux_HeII_5199, flux_HeII_23124, flux_HeII_48,
             flux_HeII_118, flux_HeII_131, flux_HeII_7876, flux_HeII_218]
noise_HeII = [noise_HeII_88, noise_HeII_204, noise_HeII_435, noise_HeII_22429,
              noise_HeII_538, noise_HeII_5199, noise_HeII_23124, noise_HeII_48,
              noise_HeII_118, noise_HeII_131, noise_HeII_7876, noise_HeII_218]

rest_wavelen_Lya_88 = DATA_Lya.field('rest_wavelen_Lya_88')
flux_Lya_88 = DATA_Lya.field('flux_Lya_88')
noise_Lya_88 = DATA_Lya.field('noise_Lya_88')
rest_wavelen_Lya_204 = DATA_Lya.field('rest_wavelen_Lya_204')
flux_Lya_204 = DATA_Lya.field('flux_Lya_204')
noise_Lya_204 = DATA_Lya.field('noise_Lya_204')
rest_wavelen_Lya_435 = DATA_Lya.field('rest_wavelen_Lya_435')
flux_Lya_435 = DATA_Lya.field('flux_Lya_435')
noise_Lya_435 = DATA_Lya.field('noise_Lya_435')
rest_wavelen_Lya_22429 = DATA_Lya.field('rest_wavelen_Lya_22429')
flux_Lya_22429 = DATA_Lya.field('flux_Lya_22429')
noise_Lya_22429 = DATA_Lya.field('noise_Lya_22429')
rest_wavelen_Lya_538 = DATA_Lya.field('rest_wavelen_Lya_538')
flux_Lya_538 = DATA_Lya.field('flux_Lya_538')
noise_Lya_538 = DATA_Lya.field('noise_Lya_538')
rest_wavelen_Lya_5199 = DATA_Lya.field('rest_wavelen_Lya_5199')
flux_Lya_5199 = DATA_Lya.field('flux_Lya_5199')
noise_Lya_5199 = DATA_Lya.field('noise_Lya_5199')
rest_wavelen_Lya_23124 = DATA_Lya.field('rest_wavelen_Lya_23124')
flux_Lya_23124 = DATA_Lya.field('flux_Lya_23124')
noise_Lya_23124 = DATA_Lya.field('noise_Lya_23124')
rest_wavelen_Lya_48 = DATA_Lya.field('rest_wavelen_Lya_48')
flux_Lya_48 = DATA_Lya.field('flux_Lya_48')
noise_Lya_48 = DATA_Lya.field('noise_Lya_48')
rest_wavelen_Lya_118 = DATA_Lya.field('rest_wavelen_Lya_118')
flux_Lya_118 = DATA_Lya.field('flux_Lya_118')
noise_Lya_118 = DATA_Lya.field('noise_Lya_118')
rest_wavelen_Lya_131 = DATA_Lya.field('rest_wavelen_Lya_131')
flux_Lya_131 = DATA_Lya.field('flux_Lya_131')
noise_Lya_131 = DATA_Lya.field('noise_Lya_131')
# target 7876 has no Lya peak
# target 218 has no Lya peak
rest_wavelen_Lya = [rest_wavelen_Lya_88, rest_wavelen_Lya_204, rest_wavelen_Lya_435, rest_wavelen_Lya_22429,
                    rest_wavelen_Lya_538, rest_wavelen_Lya_5199, rest_wavelen_Lya_23124, rest_wavelen_Lya_48,
                    rest_wavelen_Lya_118, rest_wavelen_Lya_131]
flux_Lya = [flux_Lya_88, flux_Lya_204, flux_Lya_435, flux_Lya_22429, flux_Lya_538, flux_Lya_5199,
            flux_Lya_23124, flux_Lya_48, flux_Lya_118, flux_Lya_131]
noise_Lya = [noise_Lya_88, noise_Lya_204, noise_Lya_435, noise_Lya_22429,
             noise_Lya_538, noise_Lya_5199, noise_Lya_23124, noise_Lya_48, noise_Lya_118, noise_Lya_131]


def Gauss(x, amp, cen, sigma, continuum):
    return continuum + amp * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))
def objective88(params, x, data):
    amp = params['amp_%s' % (iden_str[0])].value
    cen = params['cen_%s' % (iden_str[0])].value
    sigma = params['sigma_%s' % (iden_str[0])].value
    continuum = params['continuum_%s' % (iden_str[0])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective204(params, x, data):
    amp = params['amp_%s' % (iden_str[1])].value
    cen = params['cen_%s' % (iden_str[1])].value
    sigma = params['sigma_%s' % (iden_str[1])].value
    continuum = params['continuum_%s' % (iden_str[1])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective435(params, x, data):
    amp = params['amp_%s' % (iden_str[2])].value
    cen = params['cen_%s' % (iden_str[2])].value
    sigma = params['sigma_%s' % (iden_str[2])].value
    continuum = params['continuum_%s' % (iden_str[2])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective22429(params, x, data):
    amp = params['amp_%s' % (iden_str[3])].value
    cen = params['cen_%s' % (iden_str[3])].value
    sigma = params['sigma_%s' % (iden_str[3])].value
    continuum = params['continuum_%s' % (iden_str[3])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective538(params, x, data):
    amp = params['amp_%s' % (iden_str[4])].value
    cen = params['cen_%s' % (iden_str[4])].value
    sigma = params['sigma_%s' % (iden_str[4])].value
    continuum = params['continuum_%s' % (iden_str[4])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective5199(params, x, data):
    amp = params['amp_%s' % (iden_str[5])].value
    cen = params['cen_%s' % (iden_str[5])].value
    sigma = params['sigma_%s' % (iden_str[5])].value
    continuum = params['continuum_%s' % (iden_str[5])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective23124(params, x, data):
    amp = params['amp_%s' % (iden_str[6])].value
    cen = params['cen_%s' % (iden_str[6])].value
    sigma = params['sigma_%s' % (iden_str[6])].value
    continuum = params['continuum_%s' % (iden_str[6])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective48(params, x, data):
    amp = params['amp_%s' % (iden_str[7])].value
    cen = params['cen_%s' % (iden_str[7])].value
    sigma = params['sigma_%s' % (iden_str[7])].value
    continuum = params['continuum_%s' % (iden_str[7])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data

def objective118(params, x, data):
    amp = params['amp_%s' % (iden_str[8])].value
    cen = params['cen_%s' % (iden_str[8])].value
    sigma = params['sigma_%s' % (iden_str[8])].value
    continuum = params['continuum_%s' % (iden_str[8])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective131(params, x, data):
    amp = params['amp_%s' % (iden_str[9])].value
    cen = params['cen_%s' % (iden_str[9])].value
    sigma = params['sigma_%s' % (iden_str[9])].value
    continuum = params['continuum_%s' % (iden_str[9])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective7876(params, x, data):
    amp = params['amp_%s' % (iden_str[10])].value
    cen = params['cen_%s' % (iden_str[10])].value
    sigma = params['sigma_%s' % (iden_str[10])].value
    continuum = params['continuum_%s' % (iden_str[10])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective218(params, x, data):
    amp = params['amp_%s' % (iden_str[11])].value
    cen = params['cen_%s' % (iden_str[11])].value
    sigma = params['sigma_%s' % (iden_str[11])].value
    continuum = params['continuum_%s' % (iden_str[11])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
Gauss_model = Model(Gauss, nan_policy='propagate')

# TARGET 88: wavelengths 1638.3:1640.1

##################################################################################################################
FWHM_HeII_88 = 1.
Delta_wavelen_HeII_88 = 2  # from plots
Delta_v_HeII_88 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_88) / HeII  # in km/s
x = rest_wavelen_HeII_88
data = flux_HeII_88

# HE -II ANALYSIS
params_He88 = Parameters()
params_He88.add('amp_%s' % (iden_str[0]), value=max(flux_HeII_88), min=15)
params_He88.add('cen_%s' % (iden_str[0]), value=1639.5, min=HeII - Delta_wavelen_HeII_88, max=HeII)
params_He88.add('sigma_%s' % (iden_str[0]), value=3, min=0.01, max=FWHM_HeII_88)
f = flux_HeII_88[flux_HeII_88 != 0]
params_He88.add('continuum_%s' % (iden_str[0]), value=np.nanmedian(f), vary=False)

minner88 = Minimizer(objective88, params_He88, fcn_args=(x, data))  # print(dir(minner88)) # associated variables
result_HeII88 = minner88.minimize()
Continuum_88 = result_HeII88.params['continuum_88'].value
final_HeII_88 = flux_HeII[0] + result_HeII88.residual
n0_final_HeII_88 = final_HeII_88[final_HeII_88 != 0] # no zeros
indexs = np.array(np.argwhere(n0_final_HeII_88 != 0))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_88 = n0_final_HeII_88[0:len(n0_x)]
n0_noise_HeII_88 = noise_HeII_88[0:len(n0_x)]
ModelResult88 = ModelResult(Gauss_model, params_He88, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_88 = np.argmax(flux_HeII_88)
SNR_HeII_88 = max(flux_HeII_88) / noise_HeII_88[index_max_flux_88]
print('THE He-II SNR FOR TARGET 88 is: ', SNR_HeII_88)

# integrated flux of He II peak:
n0_final_HeII_int_flux_88 = n0_final_HeII_88[n0_final_HeII_88 > 6.61]  # ignore the continuum part below gaussian fit
index_int_flux = np.array(np.argwhere(n0_final_HeII_88 > 6.61))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_88 = simps(n0_final_HeII_int_flux_88 - Continuum_88, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 88 is: ', integrate_88)

# Parameters of fit:
Amplitude_88 = result_HeII88.params['amp_88'].value
Centre_88 = result_HeII88.params['cen_88'].value
Sigma_88 = result_HeII88.params['sigma_88'].value
FWHM_HeII_Gauss_88 = 2 * np.sqrt(2*np.log(2)) * Sigma_88
print('THE He-II FWHM FOR TARGET 88 IS: ', FWHM_HeII_Gauss_88)
for key in result_HeII88.params:
    print(key, "=", result_HeII88.params[key].value, "+/-", result_HeII88.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_88 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_88 = integrate_88 / (Continuum_88 * (1 + REDSHIFT[0]))
print('THE He-II EW FOR TARGET 88 IS: ', EW_HeII_88)

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
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[0]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_88 = []
integrate_new_88 = []
perr_gauss_88 = []
for n in range(N):
    y_new_88 = np.random.normal(n0_final_HeII_88, n0_noise_HeII_88)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_88, p0=[max(y_new_88), HeII, 3,
                                                                  np.nanmedian(flux_HeII_88)], maxfev=10000000)
    perr_gauss_88.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_88.append(simps(y_new_88, n0_x))
    this_x_fit_88 = np.random.normal(0, 1.)
    x_centre_88.append(this_x_fit_88)
median_x_centre_88 = np.nanmedian(x_centre_88)
std_x_centre_88 = np.nanstd(x_centre_88)
print('std-errors gaussian of target 88 (amp, cen, sigma, continuum): ', perr_gauss_88)
print('median x_centre of target 88: ', median_x_centre_88)
print('standard deviation from x0=1640.42A of target 88: ', std_x_centre_88)

# LYA ANALYSIS:
index_max_Lyaflux_88 = np.argmax(flux_Lya_88)
SNR_Lya_88 = max(flux_Lya_88) / noise_Lya_88[index_max_Lyaflux_88]
print('The Lya SNR FOR TARGET 88 is: ', SNR_Lya_88)  # =4.717403
FWHM_Lya_88 = 2.
Delta_wavelen_Lya_88 = 1  # from plots
Delta_v_Lya_88 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_88) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_88
dataLya = flux_Lya_88
params_Lya88 = Parameters()
params_Lya88.add('amp_%s' % (iden_str[0]), value=max(flux_Lya_88), min=15)
params_Lya88.add('cen_%s' % (iden_str[0]), value=1215)  # , min=Ly_alpha_rest - Delta_wavelen_Lya_88, max=Ly_alpha_rest)
params_Lya88.add('sigma_%s' % (iden_str[0]), value=3)  # , min=0.01, max=FWHM_Lya_88 / (2 * np.sqrt(2 * np.log(2))))
params_Lya88.add('continuum_%s' % (iden_str[0]), value=np.nanmedian(flux_Lya_88))
minner88 = Minimizer(objective88, params_Lya88, fcn_args=(xLya, dataLya))  # print(dir(minner88)) # associated variables
result_Lya88 = minner88.minimize()
final_Lya_88 = flux_Lya[0] + result_Lya88.residual
n0_final_Lya_88 = final_Lya_88[final_Lya_88 > 3]  # no zeros

indexs = np.array(np.argwhere(n0_final_Lya_88 > 3))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_88 = noise_Lya_88[Index]
integrate_Lya_88 = simps(n0_final_Lya_88, n0_x)  # = 54.78916698179091
print('INTEGRATED FLUX OF Lya FOR TARGET 88 is: ', integrate_Lya_88)

plt.figure()
plt.step(rest_wavelen_Lya_88, flux_Lya_88, 'b', label='flux')
plt.step(rest_wavelen_Lya_88, noise_Lya_88, 'k', label='noise')
plt.plot(rest_wavelen_Lya_88, final_Lya_88, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[0], str(REDSHIFT[0])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[0]))
plt.clf()

print('#############################################################################################################')
# TARGET 204:  wavelengths 1638:1641.5
##################################################################################################################
FWHM_HeII_204 = 2.
Delta_wavelen_HeII_204 = 1.8
Delta_v_HeII_204 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_204) / HeII
index_max_flux_204 = np.argmax(flux_HeII_204)
f = flux_HeII_204[flux_HeII_204 != 0]
params_He204 = Parameters()
params_He204.add('amp_%s' % (iden_str[1]), value=max(flux_HeII_204), min=2.2)  #, max=3)
params_He204.add('cen_%s' % (iden_str[1]), value=rest_wavelen_HeII_204[index_max_flux_204],
                 min=HeII - Delta_wavelen_HeII_204, max=HeII)
params_He204.add('sigma_%s' % (iden_str[1]), value=3, min=0.4, max=FWHM_HeII_204)
params_He204.add('continuum_%s' % (iden_str[1]), value=np.nanmedian(f), vary=False)

x = rest_wavelen_HeII_204
data = flux_HeII_204
minner204 = Minimizer(objective204, params_He204, fcn_args=(x, data))
result_HeII204 = minner204.minimize()
final_HeII_204 = flux_HeII[1] + result_HeII204.residual
n0_final_HeII_204 = final_HeII_204[final_HeII_204 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_204 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_204 = n0_final_HeII_204[0:len(n0_x)]
n0_noise_HeII_204 = noise_HeII_204[0:len(n0_x)]
ModelResult204 = ModelResult(Gauss_model, params_He204, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
SNR_HeII_204 = max(flux_HeII_204) / noise_HeII_204[index_max_flux_204]
print('THE He-II SNR FOR TARGET 204 is: ', SNR_HeII_204)  # = 3.2824476

# integrated flux of He II peak:
Continuum_204 = result_HeII204.params['continuum_204'].value
n0_final_HeII_int_flux_204 = n0_final_HeII_204[n0_final_HeII_204 > 0.634]
index_int_flux = np.array(np.argwhere(n0_final_HeII_204 > 0.634))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_204 = simps(n0_final_HeII_int_flux_204 - Continuum_204, n0_x_int_flux)
print('INTEGRATED FLUX OF TARGET He-II of 204 is: ', integrate_204)

# Parameters of fit:
Amplitude_204 = result_HeII204.params['amp_204'].value
Centre_204 = result_HeII204.params['cen_204'].value
Sigma_204 = result_HeII204.params['sigma_204'].value
FWHM_HeII_Gauss_204 = 2 * np.sqrt(2*np.log(2)) * Sigma_204
print('THE He-II FWHM FOR TARGET 204 IS: ', FWHM_HeII_Gauss_204)
for key in result_HeII204.params:
    print(key, "=", result_HeII204.params[key].value, "+/-", result_HeII204.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_204 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_204 = integrate_204 / (Continuum_204 * (1 + REDSHIFT[1]))
print('THE He-II EW FOR TARGET 204 IS: ', EW_HeII_204)

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
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[1]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_204 = []
integrate_new_204 = []
perr_gauss_204 = []
for n in range(N):
    y_new_204 = np.random.normal(n0_final_HeII_204, n0_noise_HeII_204)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_204, p0=[max(y_new_204), HeII, 3,
                                                                   np.nanmedian(flux_HeII_204)], maxfev=10000000)
    perr_gauss_204.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_204.append(simps(y_new_204, n0_x))
    this_x_fit_204 = np.random.normal(0, 1.)
    x_centre_204.append(this_x_fit_204)
median_x_centre_204 = np.nanmedian(x_centre_204)
std_x_centre_204 = np.nanstd(x_centre_204)
print('std-errors gaussian of target 204 (amp, cen, sigma, continuum): ', perr_gauss_204)
print('median x_centre of target 204: ', median_x_centre_204)
print('standard deviation from x0=1640.42A of target 204: ', std_x_centre_204)

# LYA ANALYSIS:
index_max_Lyaflux_204 = np.argmax(flux_Lya_204)
SNR_Lya_204 = max(flux_Lya_204) / noise_Lya_204[index_max_Lyaflux_204]
print('The Lya SNR FOR TARGET 204 is: ', SNR_Lya_204)  # = 16.23129
FWHM_Lya_204 = 2.
Delta_wavelen_Lya_204 = 1  # from plots
Delta_v_Lya_204 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_204) / Ly_alpha_rest
xLya = rest_wavelen_Lya_204
dataLya = flux_Lya_204
params_Lya204 = Parameters()
params_Lya204.add('amp_%s' % (iden_str[1]), value=max(flux_Lya_204), min=15)
params_Lya204.add('cen_%s' % (iden_str[1]), value=1215)
params_Lya204.add('sigma_%s' % (iden_str[1]), value=3)
params_Lya204.add('continuum_%s' % (iden_str[1]), value=np.nanmedian(flux_Lya_204))
minner204 = Minimizer(objective204, params_Lya204, fcn_args=(xLya, dataLya))
# print(dir(minner88)) # associated variables
result_Lya204 = minner204.minimize()
final_Lya_204 = flux_Lya[1] + result_Lya204.residual
n0_final_Lya_204 = final_Lya_204[final_Lya_204 > 1.1]

indexs = np.array(np.argwhere(n0_final_Lya_204 > 1.1))
Index = np.concatenate(indexs) # Join a sequence of arrays along an existing axis
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_204 = noise_Lya_204[Index]
integrate_Lya_204 = simps(n0_final_Lya_204, n0_x)  # = 33.342346350022126
print('INTEGRATED FLUX OF Lya of 204 is: ', integrate_Lya_204)

plt.figure()
plt.step(rest_wavelen_Lya_204, flux_Lya_204, 'b', label='flux')
plt.step(rest_wavelen_Lya_204, noise_Lya_204, 'k', label='noise')
plt.plot(rest_wavelen_Lya_204, final_Lya_204, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[1], str(REDSHIFT[1])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[1]))
plt.clf()

print('###############################################################################################################')
# TARGET 435:
##################################################################################################################
FWHM_HeII_435 = 1.5
Delta_wavelen_HeII_435 = 2.8
Delta_v_HeII_435 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_435) / HeII
x = rest_wavelen_HeII_435
data = flux_HeII_435

# HE -II ANALYSIS
params_He435 = Parameters()
params_He435.add('amp_%s' % (iden_str[2]), value=flux_HeII_435[17])
params_He435.add('cen_%s' % (iden_str[2]), value=rest_wavelen_HeII_435[17])
params_He435.add('sigma_%s' % (iden_str[2]), value=1, min=0.01, max=FWHM_HeII_435)
f = flux_HeII_435[flux_HeII_435 != 0]
params_He435.add('continuum_%s' % (iden_str[2]), value=np.nanmedian(f), vary=False)

minner435 = Minimizer(objective435, params_He435, fcn_args=(x, data))
result_HeII435 = minner435.minimize()
Continuum_435 = result_HeII435.params['continuum_435'].value
final_HeII_435 = flux_HeII[2] + result_HeII435.residual
n0_final_HeII_435 = final_HeII_435[final_HeII_435 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_435 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_noise_HeII_435 = noise_HeII_435[0:len(n0_x)]
n0_final_HeII_435 = n0_final_HeII_435[0:len(n0_x)]
ModelResult435 = ModelResult(Gauss_model, params_He435, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_435 = 17
SNR_HeII_435 = max(flux_HeII_435) / noise_HeII_435[index_max_flux_435]
print('The He-II SNR FOR TARGET 435 is: ', SNR_HeII_435)

# integrated flux of He II peak:
n0_final_HeII_int_flux_435 = n0_final_HeII_435[n0_final_HeII_435 > 1.39]
index_int_flux = np.array(np.argwhere(n0_final_HeII_435 > 1.39))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_435 = simps(n0_final_HeII_int_flux_435 - Continuum_435, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 435 is: ', integrate_435)

# Parameters of fit:
Amplitude_435 = result_HeII435.params['amp_435'].value
Centre_435 = result_HeII435.params['cen_435'].value
Sigma_435 = result_HeII435.params['sigma_435'].value
FWHM_HeII_Gauss_435 = 2 * np.sqrt(2*np.log(2)) * Sigma_435
print('The He-II FWHM FOR TARGET 435 is: ', FWHM_HeII_Gauss_435)   # 1.1189958379952543
for key in result_HeII435.params:
    print(key, "=", result_HeII435.params[key].value, "+/-", result_HeII435.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_435 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_435 = integrate_435 / (Continuum_435 * (1 + REDSHIFT[2]))
print('THE He-II EW FOR TARGET 435 IS: ', EW_HeII_435)

plt.figure()
plt.step(rest_wavelen_HeII_435, flux_HeII_435, 'b', label='flux')
plt.step(rest_wavelen_HeII_435, noise_HeII_435, 'k', label='noise')
plt.plot(rest_wavelen_HeII_435, final_HeII_435, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1633, 1643)
plt.axvline(x=HeII, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[2], str(REDSHIFT[2])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[2]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[2]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_435 = []
integrate_new_435 = []
perr_gauss_435 = []
for n in range(N):
    y_new_435 = np.random.normal(n0_final_HeII_435, n0_noise_HeII_435)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_435, p0=[max(y_new_435), HeII,
                                                                   3, np.nanmedian(flux_HeII_435)], maxfev=10000000)
    perr_gauss_435.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_435.append(simps(y_new_435, n0_x))
    this_x_fit_435 = np.random.normal(0, 1.)
    x_centre_435.append(this_x_fit_435)
median_x_centre_435 = np.nanmedian(x_centre_435)
std_x_centre_435 = np.nanstd(x_centre_435)
print('std-errors gaussian of target 435 (amp, cen, sigma, continuum): ', perr_gauss_435)
print('median x_centre of target 435: ', median_x_centre_435)
print('standard deviation from x0=1640.42A of target 435: ', std_x_centre_435)

# LYA ANALYSIS:
index_max_Lyaflux_435 = np.argmax(flux_Lya_435)
SNR_Lya_435 = max(flux_Lya_435) / noise_Lya_435[index_max_Lyaflux_435]
print('The Lya SNR FOR TARGET 435 is: ', SNR_Lya_435)  # = 84.764755
FWHM_Lya_435 = 2.
Delta_wavelen_Lya_435 = 2  # from plots
Delta_v_Lya_435 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_435) / Ly_alpha_rest
xLya = rest_wavelen_Lya_435
dataLya = flux_Lya_435
params_Lya435 = Parameters()
params_Lya435.add('amp_%s' % (iden_str[2]), value=max(flux_Lya_435), min=15)
params_Lya435.add('cen_%s' % (iden_str[2]), value=rest_wavelen_Lya_435[index_max_Lyaflux_435])
params_Lya435.add('sigma_%s' % (iden_str[2]), value=3)
params_Lya435.add('continuum_%s' % (iden_str[2]), value=np.nanmedian(flux_Lya_435))
minner435 = Minimizer(objective435, params_Lya435, fcn_args=(xLya, dataLya))
result_Lya435 = minner435.minimize()
final_Lya_435 = flux_Lya[2] + result_Lya435.residual
n0_final_Lya_435 = final_Lya_435[final_Lya_435 > 1.4]

indexs = np.array(np.argwhere(n0_final_Lya_435 > 1.4))
Index = np.concatenate(indexs)
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_435 = noise_Lya_435[Index]
integrate_Lya_435 = simps(n0_final_Lya_435, n0_x)  # = 126.11593
print('INTEGRATED FLUX OF Lya of 435 is: ', integrate_Lya_435)

plt.figure()
plt.step(rest_wavelen_Lya_435, flux_Lya_435, 'b', label='flux')
plt.step(rest_wavelen_Lya_435, noise_Lya_435, 'k', label='noise')
plt.plot(rest_wavelen_Lya_435, final_Lya_435, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[2], str(REDSHIFT[2])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[2]))
plt.clf()

print('###############################################################################################################')
# TARGET 22429:
##################################################################################################################
FWHM_HeII_22429 = 1.5
Delta_wavelen_HeII_22429 = 2.8  # from plots
Delta_v_HeII_22429 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_22429) / HeII  # in km/s
x = rest_wavelen_HeII_22429
data = flux_HeII_22429

params_He22429 = Parameters()
params_He22429.add('amp_%s' % (iden_str[3]), value=flux_HeII_22429[12], min=1.5)
params_He22429.add('cen_%s' % (iden_str[3]), value=rest_wavelen_HeII_22429[12])
params_He22429.add('sigma_%s' % (iden_str[3]), value=0.5, min=0.2, max=FWHM_HeII_22429)
f = flux_HeII_22429[flux_HeII_22429 != 0]
params_He22429.add('continuum_%s' % (iden_str[3]), value=np.nanmedian(f), vary=False)

minner22429 = Minimizer(objective22429, params_He22429, fcn_args=(x, data))
result_HeII22429 = minner22429.minimize()
Continuum_22429 = result_HeII22429.params['continuum_22429'].value
final_HeII_22429 = flux_HeII[3] + result_HeII22429.residual
n0_final_HeII_22429 = final_HeII_22429[final_HeII_22429 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_22429 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_noise_HeII_22429 = noise_HeII_22429[0:len(n0_x)]
n0_final_HeII_22429 = n0_final_HeII_22429[0:len(n0_x)]
ModelResult22429 = ModelResult(Gauss_model, params_He22429, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_22429 = 12
SNR_HeII_22429 = max(flux_HeII_22429) / noise_HeII_22429[index_max_flux_22429]
print('The He-II SNR FOR TARGET 22429 is: ', SNR_HeII_22429)  # = 2.6729734

# integrated flux of He II peak:
n0_final_HeII_int_flux_22429 = n0_final_HeII_22429[n0_final_HeII_22429 > 0.927]
index_int_flux = np.array(np.argwhere(n0_final_HeII_22429 > 0.927))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_22429 = simps(n0_final_HeII_int_flux_22429 - Continuum_22429, n0_x_int_flux)  # = 10.044179261014506
print('INTEGRATED FLUX OF He-II of 22429 is: ', integrate_22429)

# Parameters of fit:
Amplitude_22429 = result_HeII22429.params['amp_22429'].value
Centre_22429 = result_HeII22429.params['cen_22429'].value
Sigma_22429 = result_HeII22429.params['sigma_22429'].value
FWHM_HeII_Gauss_22429 = 2 * np.sqrt(2*np.log(2)) * Sigma_22429
print('The He-II FWHM FOR TARGET 22429 is: ', FWHM_HeII_Gauss_22429)    # 1.6845318648081995
#for key in result_HeII22429.params:
#    print(key, "=", result_HeII22429.params[key].value, "+/-", result_HeII22429.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_22429 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_22429 = integrate_22429 / (Continuum_22429 * (1 + REDSHIFT[3]))
print('THE He-II EW FOR TARGET 22429 IS: ', EW_HeII_22429)

plt.figure()
plt.step(rest_wavelen_HeII_22429, flux_HeII_22429, 'b', label='flux')
plt.step(rest_wavelen_HeII_22429, noise_HeII_22429, 'k', label='noise')
plt.plot(rest_wavelen_HeII_22429, final_HeII_22429, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1633, 1643)
plt.axvline(x=HeII, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[3], str(REDSHIFT[3])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[3]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[3]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_22429 = []
integrate_new_22429 = []
perr_gauss_22429 = []
for n in range(N):
    y_new_22429 = np.random.normal(n0_final_HeII_22429, n0_noise_HeII_22429)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_22429, p0=[max(y_new_22429), HeII,
                                                                     3, np.nanmedian(flux_HeII_22429)], maxfev=10000000)
    perr_gauss_22429.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_22429.append(simps(y_new_22429, n0_x))
    this_x_fit_22429 = np.random.normal(0, 1.)
    x_centre_22429.append(this_x_fit_22429)
median_x_centre_22429 = np.nanmedian(x_centre_22429)
std_x_centre_22429 = np.nanstd(x_centre_22429)
print('std-errors gaussian of target 22429 (amp, cen, sigma, continuum): ', perr_gauss_22429)
print('median x_centre of target 22429: ', median_x_centre_22429)
print('standard deviation from x0=1640.42A of target 22429: ', std_x_centre_22429)

# LYA ANALYSIS:
index_max_Lyaflux_22429 = np.argmax(flux_Lya_22429)
SNR_Lya_22429 = max(flux_Lya_22429) / noise_Lya_22429[index_max_Lyaflux_22429]
print('The Lya SNR FOR TARGET 22429 is: ', SNR_Lya_22429)  # = 13.586418
FWHM_Lya_22429 = 2.
Delta_wavelen_Lya_22429 = 2  # from plots
Delta_v_Lya_22429 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_22429) / Ly_alpha_rest
xLya = rest_wavelen_Lya_22429
dataLya = flux_Lya_22429
params_Lya22429 = Parameters()
params_Lya22429.add('amp_%s' % (iden_str[3]), value=max(flux_Lya_22429), min=15)
params_Lya22429.add('cen_%s' % (iden_str[3]), value=rest_wavelen_Lya_22429[index_max_Lyaflux_22429])
params_Lya22429.add('sigma_%s' % (iden_str[3]), value=3)
params_Lya22429.add('continuum_%s' % (iden_str[3]), value=np.nanmedian(flux_Lya_22429))
minner22429 = Minimizer(objective22429, params_Lya22429, fcn_args=(xLya, dataLya))
result_Lya22429 = minner22429.minimize()
final_Lya_22429 = flux_Lya[3] + result_Lya22429.residual
n0_final_Lya_22429 = final_Lya_22429[final_Lya_22429 > 1.5] # no zeros

indexs = np.array(np.argwhere(n0_final_Lya_22429 > 1.5))
Index = np.concatenate(indexs)
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_22429 = noise_Lya_22429[Index]
integrate_Lya_22429 = simps(n0_final_Lya_22429, n0_x)  # = 36.7150053631558
print('INTEGRATED FLUX OF Lya of 22429 is: ', integrate_Lya_22429)

plt.figure()
plt.step(rest_wavelen_Lya_22429, flux_Lya_22429, 'b', label='flux')
plt.step(rest_wavelen_Lya_22429, noise_Lya_22429, 'k', label='noise')
plt.plot(rest_wavelen_Lya_22429, final_Lya_22429, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[3], str(REDSHIFT[3])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[3]))
plt.clf()

print('###############################################################################################################')
# TARGET 538:
##################################################################################################################
FWHM_HeII_538 = 1.
Delta_wavelen_HeII_538 = 1.3  # from plots
Delta_v_HeII_538 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_538) / HeII  # in km/s
index_max_flux_538 = 27
x = rest_wavelen_HeII_538
data = flux_HeII_538

# HE -II ANALYSIS
params_He538 = Parameters()
params_He538.add('amp_%s' % (iden_str[4]), value=flux_HeII_538[index_max_flux_538])
params_He538.add('cen_%s' % (iden_str[4]), value=1639)  # rest_wavelen_HeII_538[index_max_flux_538])
params_He538.add('sigma_%s' % (iden_str[4]), value=3, min=0.01, max=FWHM_HeII_538)
f = flux_HeII_538[flux_HeII_538 != 0]
params_He538.add('continuum_%s' % (iden_str[4]), value=np.nanmedian(f), vary=False)

minner538 = Minimizer(objective538, params_He538, fcn_args=(x, data))
result_HeII538 = minner538.minimize()
Continuum_538 = result_HeII538.params['continuum_538'].value
final_HeII_538 = flux_HeII[4] + result_HeII538.residual
n0_final_HeII_538 = final_HeII_538[final_HeII_538 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_538 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_noise_HeII_538 = noise_HeII_538[0:len(n0_x)]
n0_final_HeII_538 = n0_final_HeII_538[0:len(n0_x)]
ModelResult538 = ModelResult(Gauss_model, params_He538, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
SNR_HeII_538 = max(flux_HeII_538) / noise_HeII_538[index_max_flux_538]
print('The He-II SNR FOR TARGET 538 is: ', SNR_HeII_538)

# integrated flux of He II peak:
n0_final_HeII_int_flux_538 = n0_final_HeII_538[n0_final_HeII_538 > 1.6348]
index_int_flux = np.array(np.argwhere(n0_final_HeII_538 > 1.6348))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_538 = simps(n0_final_HeII_int_flux_538 - Continuum_538, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 538 is: ', integrate_538)

# Parameters of fit:
Amplitude_538 = result_HeII538.params['amp_538'].value
Centre_538 = result_HeII538.params['cen_538'].value
Sigma_538 = result_HeII538.params['sigma_538'].value
FWHM_HeII_Gauss_538 = 2 * np.sqrt(2*np.log(2)) * Sigma_538
print('The He-II FWHM FOR TARGET 538 is: ', FWHM_HeII_Gauss_538)
for key in result_HeII538.params:
    print(key, "=", result_HeII538.params[key].value, "+/-", result_HeII538.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_538 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_538 = integrate_538 / (Continuum_538 * (1 + REDSHIFT[4]))
print('THE He-II EW FOR TARGET 538 IS: ', EW_HeII_538)

plt.figure()
plt.step(rest_wavelen_HeII_538, flux_HeII_538, 'b', label='flux')
plt.step(rest_wavelen_HeII_538, noise_HeII_538, 'k', label='noise')
plt.plot(rest_wavelen_HeII_538, final_HeII_538, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1633, 1643)
plt.axvline(x=HeII, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[4], str(REDSHIFT[4])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[4]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[4]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_538 = []
integrate_new_538 = []
perr_gauss_538 = []
for n in range(N):
    y_new_538 = np.random.normal(n0_final_HeII_538, n0_noise_HeII_538)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_538, p0=[max(y_new_538), HeII,
                                                                   3, np.nanmedian(flux_HeII_538)], maxfev=10000000)
    perr_gauss_538.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_538.append(simps(y_new_538, n0_x))
    this_x_fit_538 = np.random.normal(0, 1.)
    x_centre_538.append(this_x_fit_538)
median_x_centre_538 = np.nanmedian(x_centre_538)
std_x_centre_538 = np.nanstd(x_centre_538)
print('std-errors gaussian of target 538 (amp, cen, sigma, continuum): ', perr_gauss_538)
print('median x_centre of target 538: ', median_x_centre_538)
print('standard deviation from x0=1640.42A of target 538: ', std_x_centre_538)  # 0.9369173758890578

# LYA ANALYSIS:
index_max_Lyaflux_538 = np.argmax(flux_Lya_538)
SNR_Lya_538 = max(flux_Lya_538) / noise_Lya_538[index_max_Lyaflux_538]
print('The Lya SNR FOR TARGET 538 is: ', SNR_Lya_538)  # = 23.508476
FWHM_Lya_538 = 2.
Delta_wavelen_Lya_538 = 2  # from plots
Delta_v_Lya_538 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_538) / Ly_alpha_rest
xLya = rest_wavelen_Lya_538
dataLya = flux_Lya_538
params_Lya538 = Parameters()
params_Lya538.add('amp_%s' % (iden_str[4]), value=max(flux_Lya_538), min=15)
params_Lya538.add('cen_%s' % (iden_str[4]), value=rest_wavelen_Lya_538[index_max_Lyaflux_538])
params_Lya538.add('sigma_%s' % (iden_str[4]), value=3)
params_Lya538.add('continuum_%s' % (iden_str[4]), value=np.nanmedian(flux_Lya_538))
minner538 = Minimizer(objective538, params_Lya538, fcn_args=(xLya, dataLya))
#print(dir(minner88)) # associated variables
result_Lya538 = minner538.minimize()
final_Lya_538 = flux_Lya[4] + result_Lya538.residual
#print(final_Lya_538)
n0_final_Lya_538 = final_Lya_538[final_Lya_538 > 1.6]

indexs = np.array(np.argwhere(n0_final_Lya_538 > 1.6))
Index = np.concatenate(indexs)
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_538 = noise_Lya_538[Index]
integrate_Lya_538 = simps(n0_final_Lya_538, n0_x)  # = 29.329256
print('INTEGRATED FLUX OF Lya of 538 is: ', integrate_Lya_538)

plt.figure()
plt.step(rest_wavelen_Lya_538, flux_Lya_538, 'b', label='flux')
plt.step(rest_wavelen_Lya_538, noise_Lya_538, 'k', label='noise')
plt.plot(rest_wavelen_Lya_538, final_Lya_538, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[4], str(REDSHIFT[4])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[4]))
plt.clf()

print('###############################################################################################################')
# TARGET 5199:
##################################################################################################################
FWHM_HeII_5199 = 1.5
Delta_wavelen_HeII_5199 = 1
Delta_v_HeII_5199 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_5199) / HeII
index_max_flux_5199 = np.argmax(flux_HeII_5199)
params_He5199 = Parameters()
params_He5199.add('amp_%s' % (iden_str[5]), value=flux_HeII_5199[index_max_flux_5199])
params_He5199.add('cen_%s' % (iden_str[5]), value=rest_wavelen_HeII_5199[index_max_flux_5199])
params_He5199.add('sigma_%s' % (iden_str[5]), value=1, min=0.01, max=FWHM_HeII_5199)
f = flux_HeII_5199[flux_HeII_5199 != 0]
params_He5199.add('continuum_%s' % (iden_str[5]), value=np.nanmedian(f), vary=False)

x = rest_wavelen_HeII_5199
data = flux_HeII_5199
minner5199 = Minimizer(objective5199, params_He5199, fcn_args=(x, data))
result_HeII5199 = minner5199.minimize()
Continuum_5199 = result_HeII5199.params['continuum_5199'].value
final_HeII_5199 = flux_HeII[5] + result_HeII5199.residual
n0_final_HeII_5199 = final_HeII_5199[final_HeII_5199 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_5199 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_5199 = n0_final_HeII_5199[0:len(n0_x)]
n0_noise_HeII_5199 = noise_HeII_5199[0:len(n0_x)]
ModelResult5199 = ModelResult(Gauss_model, params_He5199, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
SNR_HeII_5199 = max(flux_HeII_5199) / noise_HeII_5199[index_max_flux_5199]
print('The He-II SNR FOR TARGET 5199 is: ', SNR_HeII_5199)  # = 2.636273

# integrated flux of He II peak:
n0_final_HeII_int_flux_5199 = n0_final_HeII_5199[n0_final_HeII_5199 > 0.86917]
index_int_flux = np.array(np.argwhere(n0_final_HeII_5199 > 0.86917))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_5199 = simps(n0_final_HeII_int_flux_5199 - Continuum_5199, n0_x_int_flux)  # = 4.224164
print('INTEGRATED FLUX OF He-II of 5199 is: ', integrate_5199)

# Parameters of fit:
Amplitude_5199 = result_HeII5199.params['amp_5199'].value
Centre_5199 = result_HeII5199.params['cen_5199'].value
Sigma_5199 = result_HeII5199.params['sigma_5199'].value
FWHM_HeII_Gauss_5199 = 2 * np.sqrt(2*np.log(2)) * Sigma_5199
print('THE He-II FWHM FOR TARGET 5199 is: ', FWHM_HeII_Gauss_5199)      # 1.358878291994959
for key in result_HeII5199.params:
    print(key, "=", result_HeII5199.params[key].value, "+/-", result_HeII5199.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_5199 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_5199 = integrate_5199 / (Continuum_5199 * (1 + REDSHIFT[5]))
print('THE He-II EW FOR TARGET 5199 IS: ', EW_HeII_5199)

plt.figure()
plt.step(rest_wavelen_HeII_5199, flux_HeII_5199, 'b', label='flux')
plt.step(rest_wavelen_HeII_5199, noise_HeII_5199, 'k', label='noise')
plt.plot(rest_wavelen_HeII_5199, final_HeII_5199, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1633, 1643)
plt.axvline(x=HeII, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[5], str(REDSHIFT[5])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[5]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[5]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_5199 = []
integrate_new_5199 = []
perr_gauss_5199 = []
for n in range(N):
    y_new_5199 = np.random.normal(n0_final_HeII_5199, n0_noise_HeII_5199)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_5199, p0=[max(y_new_5199), HeII,
                                                                    3, np.nanmedian(flux_HeII_5199)], maxfev=10000000)
    perr_gauss_5199.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_5199.append(simps(y_new_5199, n0_x))
    this_x_fit_5199 = np.random.normal(0, 1.)
    x_centre_5199.append(this_x_fit_5199)
median_x_centre_5199 = np.nanmedian(x_centre_5199)
std_x_centre_5199 = np.nanstd(x_centre_5199)
print('std-errors gaussian of target 5199 (amp, cen, sigma, continuum): ', perr_gauss_5199)
print('median x_centre of target 5199: ', median_x_centre_5199)
print('standard deviation from x0=1640.42A of target 5199: ', std_x_centre_5199)  # 0.6605748968603495

# LYA ANALYSIS:
index_max_Lyaflux_5199 = np.argmax(flux_Lya_5199)
SNR_Lya_5199 = max(flux_Lya_5199) / noise_Lya_5199[index_max_Lyaflux_5199]
print('The Lya SNR FOR TARGET 5199 is: ', SNR_Lya_5199)  # = 2.6312444
FWHM_Lya_5199 = 0.1
Delta_wavelen_Lya_5199 = 0.2  # from plots
Delta_v_Lya_5199 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_5199) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_5199
dataLya = flux_Lya_5199
params_Lya5199 = Parameters()
params_Lya5199.add('amp_%s' % (iden_str[5]), value=max(flux_Lya_5199), min=4)
params_Lya5199.add('cen_%s' % (iden_str[5]), value=rest_wavelen_Lya_5199[index_max_Lyaflux_5199], max=Ly_alpha_rest)
params_Lya5199.add('sigma_%s' % (iden_str[5]), value=1)
params_Lya5199.add('continuum_%s' % (iden_str[5]), value=np.median(flux_Lya_5199))
minner5199 = Minimizer(objective5199, params_Lya5199, fcn_args=(xLya, dataLya))
result_Lya5199 = minner5199.minimize()
final_Lya_5199 = flux_Lya[5] + result_Lya5199.residual
n0_final_Lya_5199 = final_Lya_5199[final_Lya_5199 > 1] # TODO
print(n0_final_Lya_5199)
#exit()

indexs = np.array(np.argwhere(n0_final_Lya_5199 > 1))
Index = np.concatenate(indexs)
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_5199 = noise_Lya_5199[Index]
integrate_Lya_5199 = simps(n0_final_Lya_5199, n0_x) # = 4.5607185
print('INTEGRATED FLUX OF Lya of 5199 is: ', integrate_Lya_5199)

plt.figure()
plt.step(rest_wavelen_Lya_5199, flux_Lya_5199, 'b', label='flux')
plt.step(rest_wavelen_Lya_5199, noise_Lya_5199, 'k', label='noise')
plt.plot(rest_wavelen_Lya_5199, final_Lya_5199, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[5], str(REDSHIFT[5])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[5]))
plt.clf()

print('###############################################################################################################')
# TARGET 23124:
##################################################################################################################
FWHM_HeII_23124 = 1.5
Delta_wavelen_HeII_23124 = 1
Delta_v_HeII_23124 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_23124) / HeII
index_max_flux_23124 = 24
params_He23124 = Parameters()
params_He23124.add('amp_%s' % (iden_str[6]), value=flux_HeII_23124[index_max_flux_23124], min=0.)
params_He23124.add('cen_%s' % (iden_str[6]), value=rest_wavelen_HeII_23124[index_max_flux_23124])
params_He23124.add('sigma_%s' % (iden_str[6]), value=1, min=0.01, max=FWHM_HeII_23124)
f = flux_HeII_23124[flux_HeII_23124 != 0]
params_He23124.add('continuum_%s' % (iden_str[6]), value=np.nanmedian(flux_HeII_23124))

x = rest_wavelen_HeII_23124
data = flux_HeII_23124
minner23124 = Minimizer(objective23124, params_He23124, fcn_args=(x, data))
result_HeII23124 = minner23124.minimize()
Continuum_23124 = result_HeII23124.params['continuum_23124'].value
final_HeII_23124 = flux_HeII[6] + result_HeII23124.residual
n0_final_HeII_23124 = final_HeII_23124[final_HeII_23124 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_23124 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_23124 = n0_final_HeII_23124[0:len(n0_x)]
n0_noise_HeII_23124 = noise_HeII_23124[0:len(n0_x)]
ModelResult23124 = ModelResult(Gauss_model, params_He23124, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
SNR_HeII_23124 = max(flux_HeII_23124) / noise_HeII_23124[index_max_flux_23124]
print('The He-II SNR FOR TARGET 23124 is: ', SNR_HeII_23124)

# integrated flux of He II peak:
n0_final_HeII_int_flux_23124 = n0_final_HeII_23124[n0_final_HeII_23124 > 1.978]
index_int_flux = np.array(np.argwhere(n0_final_HeII_23124 > 1.978))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_23124 = simps(n0_final_HeII_int_flux_23124 - Continuum_23124, n0_x_int_flux)  # 3.3398381445585983
print('INTEGRATED FLUX OF He-II PEAK FOR 23124 IS: ', integrate_23124)

# Parameters of fit:
Amplitude_23124 = result_HeII23124.params['amp_23124'].value
Centre_23124 = result_HeII23124.params['cen_23124'].value
Sigma_23124 = result_HeII23124.params['sigma_23124'].value
FWHM_HeII_Gauss_23124 = 2 * np.sqrt(2*np.log(2)) * Sigma_23124
print('THE He-II FWHM FOR TARGET 23124 IS: ', FWHM_HeII_Gauss_23124)  # 0.44629818546793065
for key in result_HeII23124.params:
    print(key, "=", result_HeII23124.params[key].value, "+/-", result_HeII23124.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_23124 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_23124 = integrate_23124 / (Continuum_23124 * (1+REDSHIFT[6]))
print('THE He-II EW FOR TARGET 23124 IS: ', EW_HeII_23124)

plt.figure()
plt.step(rest_wavelen_HeII_23124, flux_HeII_23124, 'b', label='flux')
plt.step(rest_wavelen_HeII_23124, noise_HeII_23124, 'k', label='noise')
plt.plot(rest_wavelen_HeII_23124, final_HeII_23124, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1633, 1643)
plt.axvline(x=HeII, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[6], str(REDSHIFT[6])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[6]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[6]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_23124 = []
integrate_new_23124 = []
perr_gauss_23124 = []
for n in range(N):
    y_new_23124 = np.random.normal(n0_final_HeII_23124, n0_noise_HeII_23124)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_23124, p0=[max(y_new_23124), HeII,
                                                                     3, np.nanmedian(flux_HeII_23124)], maxfev=10000000)
    perr_gauss_23124.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_23124.append(simps(y_new_23124, n0_x))
    this_x_fit_23124 = np.random.normal(0, 1.)
    x_centre_23124.append(this_x_fit_23124)
median_x_centre_23124 = np.nanmedian(x_centre_23124)
std_x_centre_23124 = np.nanstd(x_centre_23124)
print('std-errors gaussian of target 23124 (amp, cen, sigma, continuum): ', perr_gauss_23124)
print('median x_centre of target 23124: ', median_x_centre_23124)
print('standard deviation from x0=1640.42A of target 23124: ', std_x_centre_23124)

# LYA ANALYSIS: FIT, integrated flux and SNR
index_max_Lyaflux_23124 = np.argmax(flux_Lya_23124)
SNR_Lya_23124 = max(flux_Lya_23124) / noise_Lya_23124[index_max_Lyaflux_23124]
print('THE LYA SNR FOR TARGET 23124 IS: ', SNR_Lya_23124)  # = 26.23804
FWHM_Lya_23124 = 2.
Delta_wavelen_Lya_23124 = 2
Delta_v_Lya_23124 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_23124) / Ly_alpha_rest
xLya = rest_wavelen_Lya_23124
#print(np.nanmedian(flux_Lya_23124)) #= 2.4780107
dataLya = flux_Lya_23124
params_Lya23124 = Parameters()
params_Lya23124.add('amp_%s' % (iden_str[6]), value=max(flux_Lya_23124), min=15)
params_Lya23124.add('cen_%s' % (iden_str[6]), value=rest_wavelen_Lya_23124[index_max_Lyaflux_23124])
params_Lya23124.add('sigma_%s' % (iden_str[6]), value=3)
params_Lya23124.add('continuum_%s' % (iden_str[6]), value=np.nanmedian(flux_Lya_23124))
minner23124 = Minimizer(objective23124, params_Lya23124, fcn_args=(xLya, dataLya))
result_Lya23124 = minner23124.minimize()
final_Lya_23124 = flux_Lya[6] + result_Lya23124.residual
n0_final_Lya_23124 = final_Lya_23124[final_Lya_23124 > 2]

indexs = np.array(np.argwhere(n0_final_Lya_23124 > 2))
Index = np.concatenate(indexs)
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_23124 = noise_Lya_23124[Index]
integrate_Lya_23124 = simps(n0_final_Lya_23124, n0_x)  # = 73.98148
print('THE INTEGRATED FLUX OF LYA FOR TARGET 23124 IS: ', integrate_Lya_23124)

plt.figure()
plt.step(rest_wavelen_Lya_23124, flux_Lya_23124, 'b', label='flux')
plt.step(rest_wavelen_Lya_23124, noise_Lya_23124, 'k', label='noise')
plt.plot(rest_wavelen_Lya_23124, final_Lya_23124, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[6], str(REDSHIFT[6])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[6]))
plt.clf()

print('###############################################################################################################')
# Target 48
###########################
FWHM_HeII_48 = 0.4
Delta_wavelen_HeII_48 = 0.5  # from plots
Delta_v_HeII_48 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_48) / HeII  # in km/s
x = rest_wavelen_HeII_48
data = flux_HeII_48

# HE -II ANALYSIS
params_He48 = Parameters()
params_He48.add('amp_%s' % (iden_str[7]), value=flux_HeII_48[23], min=2.2)
params_He48.add('cen_%s' % (iden_str[7]), value=1640, min=HeII - Delta_wavelen_HeII_48, max=HeII+1)
params_He48.add('sigma_%s' % (iden_str[7]), value=3, min=0.01, max=FWHM_HeII_48)
f = flux_HeII_48[flux_HeII_48 != 0]
params_He48.add('continuum_%s' % (iden_str[7]), value=np.nanmedian(f), vary=False)

minner48 = Minimizer(objective48, params_He48, fcn_args=(x, data))  # print(dir(minner88)) # associated variables
result_HeII48 = minner48.minimize()
Continuum_48 = result_HeII48.params['continuum_48'].value
final_HeII_48 = flux_HeII[7] + result_HeII48.residual
n0_final_HeII_48 = final_HeII_48[final_HeII_48 != 0] # no zeros
indexs = np.array(np.argwhere(n0_final_HeII_48 != 0))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_48 = n0_final_HeII_48[0:len(n0_x)]
n0_noise_HeII_48 = noise_HeII_48[0:len(n0_x)]
ModelResult48 = ModelResult(Gauss_model, params_He48, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_48 = np.argmax(flux_HeII_48)
SNR_HeII_48 = max(flux_HeII_48) / noise_HeII_48[index_max_flux_48]
print('THE He-II SNR FOR TARGET 48 is: ', SNR_HeII_48)

# integrated flux of He II peak:
n0_final_HeII_int_flux_48 = n0_final_HeII_48[n0_final_HeII_48 > 2.009]
index_int_flux = np.array(np.argwhere(n0_final_HeII_48 > 2.009))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_48 = simps(n0_final_HeII_int_flux_48 - Continuum_48, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 48 is: ', integrate_48)

# Parameters of fit:
Amplitude_48 = result_HeII48.params['amp_48'].value
Centre_48 = result_HeII48.params['cen_48'].value
Sigma_48 = result_HeII48.params['sigma_48'].value
FWHM_HeII_Gauss_48 = 2 * np.sqrt(2*np.log(2)) * Sigma_48
print('THE He-II FWHM FOR TARGET 48 IS: ', FWHM_HeII_Gauss_48)
for key in result_HeII48.params:
    print(key, "=", result_HeII48.params[key].value, "+/-", result_HeII48.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_48 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_48 = integrate_48 / (Continuum_48 * (1 + REDSHIFT[7]))
print('THE He-II EW FOR TARGET 48 IS: ', EW_HeII_48)

plt.figure()
plt.step(rest_wavelen_HeII[7], flux_HeII[7], 'b', label='flux')
plt.step(rest_wavelen_HeII[7], noise_HeII[7], 'k', label='noise')
plt.plot(rest_wavelen_HeII[7], final_HeII_48, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[7], str(REDSHIFT[7])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[7]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[7]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_48 = []
integrate_new_48 = []
perr_gauss_48 = []
for n in range(N):
    y_new_48 = np.random.normal(n0_final_HeII_48, n0_noise_HeII_48)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_48, p0=[max(y_new_48), HeII, 3,
                                                                  np.nanmedian(flux_HeII_48)], maxfev=10000000)
    perr_gauss_48.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_48.append(simps(y_new_48, n0_x))
    this_x_fit_48 = np.random.normal(0, 1.)
    x_centre_48.append(this_x_fit_48)
median_x_centre_48 = np.nanmedian(x_centre_48)
std_x_centre_48 = np.nanstd(x_centre_48)
print('std-errors gaussian of target 48 (amp, cen, sigma, continuum): ', perr_gauss_48)
print('median x_centre of target 48: ', median_x_centre_48)
print('standard deviation from x0=1640.42A of target 48: ', std_x_centre_48)

# LYA ANALYSIS:
index_max_Lyaflux_48 = np.argmax(flux_Lya_48)
SNR_Lya_48 = max(flux_Lya_48) / noise_Lya_48[index_max_Lyaflux_48]
print('The Lya SNR FOR TARGET 48 is: ', SNR_Lya_48)  # =4.717403
FWHM_Lya_48 = 2.
Delta_wavelen_Lya_48 = 1  # from plots
Delta_v_Lya_48 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_48) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_48
dataLya = flux_Lya_48
params_Lya48 = Parameters()
params_Lya48.add('amp_%s' % (iden_str[7]), value=max(flux_Lya_48), min=9.7)
params_Lya48.add('cen_%s' % (iden_str[7]), value=1216)  # , min=Ly_alpha_rest - Delta_wavelen_Lya_88, max=Ly_alpha_rest)
params_Lya48.add('sigma_%s' % (iden_str[7]), value=3)  # , min=0.01, max=FWHM_Lya_88 / (2 * np.sqrt(2 * np.log(2))))
params_Lya48.add('continuum_%s' % (iden_str[7]), value=np.nanmedian(flux_Lya_48))
minner48 = Minimizer(objective48, params_Lya48, fcn_args=(xLya, dataLya))  # print(dir(minner88)) # associated variables
result_Lya48 = minner48.minimize()
final_Lya_48 = flux_Lya[7] + result_Lya48.residual
n0_final_Lya_48 = final_Lya_48[final_Lya_48 > 2.18]  # no zeros

indexs = np.array(np.argwhere(n0_final_Lya_48 > 2.18))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_48 = noise_Lya_48[Index]
integrate_Lya_48 = simps(n0_final_Lya_48, n0_x)  # = 54.78916698179091
print('INTEGRATED FLUX OF Lya FOR TARGET 48 is: ', integrate_Lya_48)

plt.figure()
plt.step(rest_wavelen_Lya_48, flux_Lya_48, 'b', label='flux')
plt.step(rest_wavelen_Lya_48, noise_Lya_48, 'k', label='noise')
plt.plot(rest_wavelen_Lya_48, final_Lya_48, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[7], str(REDSHIFT[7])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[7]))
plt.clf()

print('###############################################################################################################')
# Target 118
###########################
FWHM_HeII_118 = 0.5
Delta_wavelen_HeII_118 = 4  # from plots
Delta_v_HeII_118 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_118) / HeII  # in km/s
x = rest_wavelen_HeII_118
data = flux_HeII_118
redshift_118 = 1

# HE -II ANALYSIS
params_He118 = Parameters()
params_He118.add('amp_%s' % (iden_str[8]), value=max(flux_HeII_118))  # , min=5)
params_He118.add('cen_%s' % (iden_str[8]), value=1637)  # , min=HeII - Delta_wavelen_HeII_118, max=HeII)
params_He118.add('sigma_%s' % (iden_str[8]), value=3, min=0.01, max=FWHM_HeII_118)
f = flux_HeII_118[flux_HeII_118 != 0]
params_He118.add('continuum_%s' % (iden_str[8]), value=np.nanmedian(f), vary=False)

minner118 = Minimizer(objective118, params_He118, fcn_args=(x, data))  # print(dir(minner88)) # associated variables
result_HeII118 = minner118.minimize()
Continuum_118 = result_HeII118.params['continuum_118'].value
final_HeII_118 = flux_HeII[8] + result_HeII118.residual
n0_final_HeII_118 = final_HeII_118[final_HeII_118 != 0]  # no zeros
indexs = np.array(np.argwhere(n0_final_HeII_118 != 0))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_118 = n0_final_HeII_118[0:len(n0_x)]
n0_noise_HeII_118 = noise_HeII_118[0:len(n0_x)]
ModelResult118 = ModelResult(Gauss_model, params_He118, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_118 = np.argmax(flux_HeII_118)
SNR_HeII_118 = max(flux_HeII_118) / noise_HeII_118[index_max_flux_118]
print('THE He-II SNR FOR TARGET 118 is: ', SNR_HeII_118)

# integrated flux of He II peak:
n0_final_HeII_int_flux_118 = n0_final_HeII_118[n0_final_HeII_118 > 3.972]
index_int_flux = np.array(np.argwhere(n0_final_HeII_118 > 3.972))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_118 = simps(n0_final_HeII_int_flux_118 - Continuum_118, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 118 is: ', integrate_118)

# Parameters of fit:
Amplitude_118 = result_HeII118.params['amp_118'].value
Centre_118 = result_HeII118.params['cen_118'].value
Sigma_118 = result_HeII118.params['sigma_118'].value
FWHM_HeII_Gauss_118 = 2 * np.sqrt(2 * np.log(2)) * Sigma_118
print('THE He-II FWHM FOR TARGET 118 IS: ', FWHM_HeII_Gauss_118)
for key in result_HeII118.params:
    print(key, "=", result_HeII118.params[key].value, "+/-", result_HeII118.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_118 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_118 = integrate_118 / (Continuum_118 * (1 + REDSHIFT[8]))
print('THE He-II EW FOR TARGET 118 IS: ', EW_HeII_118)

plt.figure()
plt.step(rest_wavelen_HeII[8], flux_HeII[8], 'b', label='flux')
plt.step(rest_wavelen_HeII[8], noise_HeII[8], 'k', label='noise')
plt.plot(rest_wavelen_HeII[8], final_HeII_118, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[8], str(REDSHIFT[8])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[8]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[8]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_118 = []
integrate_new_118 = []
perr_gauss_118 = []
for n in range(N):
    y_new_118 = np.random.normal(n0_final_HeII_118, n0_noise_HeII_118)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_118, p0=[max(y_new_118), HeII, 3,
                                                                   np.nanmedian(flux_HeII_118)], maxfev=10000000)
    perr_gauss_118.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_118.append(simps(y_new_118, n0_x))
    this_x_fit_118 = np.random.normal(0, 1.)
    x_centre_118.append(this_x_fit_118)
median_x_centre_118 = np.nanmedian(x_centre_118)
std_x_centre_118 = np.nanstd(x_centre_118)
print('std-errors gaussian of target 118 (amp, cen, sigma, continuum): ', perr_gauss_118)
print('median x_centre of target 118: ', median_x_centre_118)
print('standard deviation from x0=1640.42A of target 118: ', std_x_centre_118)

# LYA ANALYSIS:
index_max_Lyaflux_118 = np.argmax(flux_Lya_118)
SNR_Lya_118 = max(flux_Lya_118) / noise_Lya_118[index_max_Lyaflux_118]
print('The Lya SNR FOR TARGET 118 is: ', SNR_Lya_118)  # =4.717403
FWHM_Lya_118 = 2.
Delta_wavelen_Lya_118 = 1  # from plots
Delta_v_Lya_118 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_118) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_118
dataLya = flux_Lya_118
params_Lya118 = Parameters()
params_Lya118.add('amp_%s' % (iden_str[8]), value=max(flux_Lya_118), min=15)
params_Lya118.add('cen_%s' % (iden_str[8]), value=1215)
params_Lya118.add('sigma_%s' % (iden_str[8]), value=3)
params_Lya118.add('continuum_%s' % (iden_str[8]), value=np.nanmedian(flux_Lya_118))
minner118 = Minimizer(objective118, params_Lya118, fcn_args=(xLya, dataLya))
result_Lya118 = minner118.minimize()
final_Lya_118 = flux_Lya[8] + result_Lya118.residual
n0_final_Lya_118 = final_Lya_118[final_Lya_118 > 4.8]  # no zeros

indexs = np.array(np.argwhere(n0_final_Lya_118 > 4.8))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_118 = noise_Lya_118[Index]
integrate_Lya_118 = simps(n0_final_Lya_118, n0_x)  # = 54.78916698179091
print('INTEGRATED FLUX OF Lya FOR TARGET 118 is: ', integrate_Lya_118)

plt.figure()
plt.step(rest_wavelen_Lya_118, flux_Lya_118, 'b', label='flux')
plt.step(rest_wavelen_Lya_118, noise_Lya_118, 'k', label='noise')
plt.plot(rest_wavelen_Lya_118, final_Lya_118, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[8], str(REDSHIFT[8])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[8]))
plt.clf()

print('###############################################################################################################')
# Target 131
###########################
FWHM_HeII_131 = 0.5
Delta_wavelen_HeII_131 = 2  # from plots
Delta_v_HeII_131 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_131) / HeII  # in km/s
x = rest_wavelen_HeII_131
data = flux_HeII_131

# HE -II ANALYSIS
params_He131 = Parameters()
params_He131.add('amp_%s' % (iden_str[9]), value=max(flux_HeII_131), min=2.3)
params_He131.add('cen_%s' % (iden_str[9]), value=1639)#, min=HeII - Delta_wavelen_HeII_131, max=HeII)
params_He131.add('sigma_%s' % (iden_str[9]), value=3, min=0.01, max=FWHM_HeII_131)
f = flux_HeII_131[flux_HeII_131 != 0]
params_He131.add('continuum_%s' % (iden_str[9]), value=np.nanmedian(f), vary=False)

minner131 = Minimizer(objective131, params_He131, fcn_args=(x, data))  # print(dir(minner88)) # associated variables
result_HeII131 = minner131.minimize()
Continuum_131 = result_HeII131.params['continuum_131'].value
final_HeII_131 = flux_HeII[9] + result_HeII131.residual
n0_final_HeII_131 = final_HeII_131[final_HeII_131 != 0] # no zeros
indexs = np.array(np.argwhere(n0_final_HeII_131 != 0))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_131 = n0_final_HeII_131[0:len(n0_x)]
n0_noise_HeII_131 = noise_HeII_131[0:len(n0_x)]
ModelResult131 = ModelResult(Gauss_model, params_He131, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_131 = np.argmax(flux_HeII_131)
SNR_HeII_131 = max(flux_HeII_131) / noise_HeII_131[index_max_flux_131]
print('THE He-II SNR FOR TARGET 131 is: ', SNR_HeII_131)

# integrated flux of He II peak:
n0_final_HeII_int_flux_131 = n0_final_HeII_131[n0_final_HeII_131 > 2.86]
index_int_flux = np.array(np.argwhere(n0_final_HeII_131 > 2.86))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_131 = simps(n0_final_HeII_int_flux_131 - Continuum_131, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 131 is: ', integrate_131)

# Parameters of fit:
Amplitude_131 = result_HeII131.params['amp_131'].value
Centre_131 = result_HeII131.params['cen_131'].value
Sigma_131 = result_HeII131.params['sigma_131'].value
FWHM_HeII_Gauss_131 = 2 * np.sqrt(2 * np.log(2)) * Sigma_131
print('THE He-II FWHM FOR TARGET 131 IS: ', FWHM_HeII_Gauss_131)
for key in result_HeII131.params:
    print(key, "=", result_HeII131.params[key].value, "+/-", result_HeII131.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_131 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_131 = integrate_131 / (Continuum_131  * (1 + REDSHIFT[9]))
print('THE He-II EW FOR TARGET 131 IS: ', EW_HeII_131)

plt.figure()
plt.step(rest_wavelen_HeII[9], flux_HeII[9], 'b', label='flux')
plt.step(rest_wavelen_HeII[9], noise_HeII[9], 'k', label='noise')
plt.plot(rest_wavelen_HeII[9], final_HeII_131, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[9], str(REDSHIFT[9])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[9]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[9]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_131 = []
integrate_new_131 = []
perr_gauss_131 = []
for n in range(N):
    y_new_131 = np.random.normal(n0_final_HeII_131, n0_noise_HeII_131)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_131, p0=[max(y_new_131), HeII, 3,
                                                                   np.nanmedian(flux_HeII_131)], maxfev=10000000)
    perr_gauss_131.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_131.append(simps(y_new_131, n0_x))
    this_x_fit_131 = np.random.normal(0, 1.)
    x_centre_131.append(this_x_fit_131)
median_x_centre_131 = np.nanmedian(x_centre_131)
std_x_centre_131 = np.nanstd(x_centre_131)
print('std-errors gaussian of target 131 (amp, cen, sigma, continuum): ', perr_gauss_131)
print('median x_centre of target 131: ', median_x_centre_131)
print('standard deviation from x0=1640.42A of target 131: ', std_x_centre_131)

# LYA ANALYSIS:
index_max_Lyaflux_131 = np.argmax(flux_Lya_131)
SNR_Lya_131 = max(flux_Lya_131) / noise_Lya_131[index_max_Lyaflux_131]
print('The Lya SNR FOR TARGET 131 is: ', SNR_Lya_131)
FWHM_Lya_131 = 2.
Delta_wavelen_Lya_131 = 1  # from plots
Delta_v_Lya_131 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_131) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_131
dataLya = flux_Lya_131
params_Lya131 = Parameters()
params_Lya131.add('amp_%s' % (iden_str[9]), value=max(flux_Lya_131), min=15)
params_Lya131.add('cen_%s' % (iden_str[9]), value=1215)
params_Lya131.add('sigma_%s' % (iden_str[9]), value=3)
params_Lya131.add('continuum_%s' % (iden_str[9]), value=np.nanmedian(flux_Lya_131))
minner131 = Minimizer(objective131, params_Lya131, fcn_args=(xLya, dataLya))
result_Lya131 = minner131.minimize()
final_Lya_131 = flux_Lya[9] + result_Lya131.residual
n0_final_Lya_131 = final_Lya_131[final_Lya_131 > 1.7]  # no zeros

indexs = np.array(np.argwhere(n0_final_Lya_131 > 1.7))
Index = np.concatenate(indexs)  # Join a sequence of arrays along an existing axis
xLya = xLya[Index]
n0_x = xLya[xLya != 0]
n0_noise_Lya_131 = noise_Lya_131[Index]
integrate_Lya_131 = simps(n0_final_Lya_131, n0_x)
print('INTEGRATED FLUX OF Lya FOR TARGET 131 is: ', integrate_Lya_131)

plt.figure()
plt.step(rest_wavelen_Lya_131, flux_Lya_131, 'b', label='flux')
plt.step(rest_wavelen_Lya_131, noise_Lya_131, 'k', label='noise')
plt.plot(rest_wavelen_Lya_131, final_Lya_131, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.xlim(1208, 1222)
plt.axvline(x=Ly_alpha_rest, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.title('Lya rest-peak of target %s at z = %s' % (iden_str[9], str(REDSHIFT[9])))
plt.savefig('plots/ALL_PART3/Lya_Gauss_target_%s.pdf' % (iden_str[9]))
plt.clf()

print('###############################################################################################################')
# Target 7876

##################################################################################################################
FWHM_HeII_7876 = 1.5
Delta_wavelen_HeII_7876 = 1  # from plots
Delta_v_HeII_7876 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_7876) / HeII  # in km/s
x = rest_wavelen_HeII_7876
data = flux_HeII_7876

# HE -II ANALYSIS
params_He7876 = Parameters()
params_He7876.add('amp_%s' % (iden_str[10]), value=max(flux_HeII_7876))
params_He7876.add('cen_%s' % (iden_str[10]), value=1639.5, min=HeII - Delta_wavelen_HeII_7876, max=HeII)
params_He7876.add('sigma_%s' % (iden_str[10]), value=3, min=0.01, max=FWHM_HeII_7876)
f = flux_HeII_7876[flux_HeII_7876 != 0]
params_He7876.add('continuum_%s' % (iden_str[10]), value=np.nanmedian(f), vary=False)

minner7876 = Minimizer(objective7876, params_He7876, fcn_args=(x, data))
result_HeII7876 = minner7876.minimize()
Continuum_7876 = result_HeII7876.params['continuum_7876'].value
final_HeII_7876 = flux_HeII[10] + result_HeII7876.residual
n0_final_HeII_7876 = final_HeII_7876[final_HeII_7876 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_7876 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_7876 = n0_final_HeII_7876[0:len(n0_x)]
n0_noise_HeII_7876 = noise_HeII_7876[0:len(n0_x)]
ModelResult7876 = ModelResult(Gauss_model, params_He7876, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_7876 = np.argmax(flux_HeII_7876)
SNR_HeII_7876 = max(flux_HeII_7876) / noise_HeII_7876[index_max_flux_7876]
print('THE He-II SNR FOR TARGET 7876 is: ', SNR_HeII_7876)

# integrated flux of He II peak:
n0_final_HeII_int_flux_7876 = n0_final_HeII_7876[n0_final_HeII_7876 > 2.4747]
index_int_flux = np.array(np.argwhere(n0_final_HeII_7876 > 2.4747))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_7876 = simps(n0_final_HeII_int_flux_7876 - Continuum_7876, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 7876 is: ', integrate_7876)

# Parameters of fit:
Amplitude_7876 = result_HeII7876.params['amp_7876'].value
Centre_7876 = result_HeII7876.params['cen_7876'].value
Sigma_7876 = result_HeII7876.params['sigma_7876'].value
FWHM_HeII_Gauss_7876 = 2 * np.sqrt(2 * np.log(2)) * Sigma_7876
print('THE He-II FWHM FOR TARGET 7876 IS: ', FWHM_HeII_Gauss_7876)
for key in result_HeII7876.params:
    print(key, "=", result_HeII7876.params[key].value, "+/-", result_HeII7876.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_7876 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_7876 = integrate_7876 / (Continuum_7876 * (1 + REDSHIFT[10]))
print('THE He-II EW FOR TARGET 7876 IS: ', EW_HeII_7876)

plt.figure()
plt.step(rest_wavelen_HeII[10], flux_HeII[10], 'b', label='flux')
plt.step(rest_wavelen_HeII[10], noise_HeII[10], 'k', label='noise')
plt.plot(rest_wavelen_HeII[10], final_HeII_7876, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[10], str(REDSHIFT[10])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[10]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[10]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_7876 = []
integrate_new_7876 = []
perr_gauss_7876 = []
for n in range(N):
    y_new_7876 = np.random.normal(n0_final_HeII_7876, n0_noise_HeII_7876)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_7876, p0=[max(y_new_7876), HeII, 3,
                                                                    np.nanmedian(flux_HeII_7876)], maxfev=10000000)
    perr_gauss_7876.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_7876.append(simps(y_new_7876, n0_x))
    this_x_fit_7876 = np.random.normal(0, 1.)
    x_centre_7876.append(this_x_fit_7876)
median_x_centre_7876 = np.nanmedian(x_centre_7876)
std_x_centre_7876 = np.nanstd(x_centre_7876)
print('std-errors gaussian of target 7876 (amp, cen, sigma, continuum): ', perr_gauss_7876)
print('median x_centre of target 7876: ', median_x_centre_7876)
print('standard deviation from x0=1640.42A of target 7876: ', std_x_centre_7876)

# LYA ANALYSIS: doesn't exist

print('###############################################################################################################')
# Target 218

##################################################################################################################
FWHM_HeII_218 = 0.5
Delta_wavelen_HeII_218 = 1  # from plots
Delta_v_HeII_218 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_218) / HeII  # in km/s
x = rest_wavelen_HeII_218
data = flux_HeII_218

# HE -II ANALYSIS
params_He218 = Parameters()
params_He218.add('amp_%s' % (iden_str[11]), value=max(flux_HeII_218), min=2)
params_He218.add('cen_%s' % (iden_str[11]), value=1639., min=HeII - Delta_wavelen_HeII_218, max=HeII)
params_He218.add('sigma_%s' % (iden_str[11]), value=3, min=0.01, max=FWHM_HeII_218)
f = flux_HeII_218[flux_HeII_218 != 0]
params_He218.add('continuum_%s' % (iden_str[11]), value=np.nanmedian(f), vary=False)

minner218 = Minimizer(objective218, params_He218, fcn_args=(x, data))
result_HeII218 = minner218.minimize()
Continuum_218 = result_HeII218.params['continuum_218'].value
final_HeII_218 = flux_HeII[11] + result_HeII218.residual
n0_final_HeII_218 = final_HeII_218[final_HeII_218 != 0]
indexs = np.array(np.argwhere(n0_final_HeII_218 != 0))
Index = np.concatenate(indexs)
x = x[Index]
n0_x = x[x != 0]
n0_final_HeII_218 = n0_final_HeII_218[0:len(n0_x)]
n0_noise_HeII_218 = noise_HeII_218[0:len(n0_x)]
ModelResult218 = ModelResult(Gauss_model, params_He218, weights=True, nan_policy='propagate')

# Signal ot noise ratio:
index_max_flux_218 = np.argmax(flux_HeII_218)
SNR_HeII_218 = max(flux_HeII_218) / noise_HeII_218[index_max_flux_218]
print('THE He-II SNR FOR TARGET 218 is: ', SNR_HeII_218)

# integrated flux of He II peak:
n0_final_HeII_int_flux_218 = n0_final_HeII_218[n0_final_HeII_218 > 2.3715944]
index_int_flux = np.array(np.argwhere(n0_final_HeII_218 > 2.3715944))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_218 = simps(n0_final_HeII_int_flux_218 - Continuum_218, n0_x_int_flux)
print('INTEGRATED FLUX OF He-II of 218 is: ', integrate_218)

# Parameters of fit:
Amplitude_218 = result_HeII218.params['amp_218'].value
Centre_218 = result_HeII218.params['cen_218'].value
Sigma_218 = result_HeII218.params['sigma_218'].value
FWHM_HeII_Gauss_218 = 2 * np.sqrt(2 * np.log(2)) * Sigma_218
print('THE He-II FWHM FOR TARGET 218 IS: ', FWHM_HeII_Gauss_218)
for key in result_HeII218.params:
    print(key, "=", result_HeII218.params[key].value, "+/-", result_HeII218.params[key].stderr)

# EQUIVALENT WIDTH: EW = line-flux - [continuum-level = Continuum_..]
dlambda_HeII_218 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_HeII_218 = integrate_218 / (Continuum_218 * (1 + REDSHIFT[11]))
print('THE He-II EW FOR TARGET 218 IS: ', EW_HeII_218)

plt.figure()
plt.step(rest_wavelen_HeII[11], flux_HeII[11], 'b', label='flux')
plt.step(rest_wavelen_HeII[11], noise_HeII[11], 'k', label='noise')
plt.plot(rest_wavelen_HeII[11], final_HeII_218, 'r', label='fit')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.axvline(x=HeII, color='c')
plt.xlim(1633, 1643)
plt.grid(True)
plt.legend(loc='best')
plt.title('He-II rest-peak of target %s at z = %s' % (iden_str[11], str(REDSHIFT[11])))
plt.savefig('plots/ALL_PART3/HeII_Gauss_target_%s.pdf' % (iden_str[11]))
plt.savefig('MAIN_LATEX/PLOTS/HeII_Gauss_target_%s.pdf' % (iden_str[11]))
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_218 = []
integrate_new_218 = []
perr_gauss_218 = []
for n in range(N):
    y_new_218 = np.random.normal(n0_final_HeII_218, n0_noise_HeII_218)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_218, p0=[max(y_new_218), HeII, 3,
                                                                   Continuum_218], maxfev=1000000)
    perr_gauss_218.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_218.append(simps(y_new_218, n0_x))
    this_x_fit_218 = np.random.normal(0, 1.)
    x_centre_218.append(this_x_fit_218)
median_x_centre_218 = np.nanmedian(x_centre_218)
std_x_centre_218 = np.nanstd(x_centre_218)
print('std-errors gaussian of target 218 (amp, cen, sigma, continuum): ', perr_gauss_218)
print('median x_centre of target 218: ', median_x_centre_218)
print('standard deviation from x0=1640.42A of target 218: ', std_x_centre_218)

# LYA ANALYSIS: doesn't exist

print('###############################################################################################################')

# END
########################
print('finished part 3')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

# RESULTS:
'''
THE He-II SNR FOR TARGET 88 is:  2.5999868
INTEGRATED FLUX OF He-II of 88 is:  10.699331
THE He-II FWHM FOR TARGET 88 IS:  0.6656857750720175
amp_88 = 15.000729263612993 +/- None
cen_88 = 1639.5 +/- None
sigma_88 = 0.28269072045514565 +/- None
continuum_88 = 6.601551 +/- None
THE He-II EW FOR TARGET 88 IS:  0.40987962762403957
std-errors gaussian of target 88 (amp, cen, sigma, continuum):  [array([4.04489675, 0.1123751 , 0.1148944 , 0.70690162]), array([1.78314696, 0.41127028, 0.50031013, 0.93945426]), array([1.80793793, 1.31101607, 1.64915481, 0.9914702 ]), array([4.17187729, 0.07077452, 0.04025541, 0.62300873]), array([2.12313967, 0.62641255, 0.72647881, 1.00576332]), array([3.42459357, 0.11862389, 0.12169209, 0.63484913]), array([4.31149289, 0.12621214, 0.12952283, 0.76376867]), array([3.1618381 , 0.0608431 , 0.06265514, 0.62921035]), array([4.50614329, 0.07653922, 0.07663385, 0.73478518]), array([5.4168140e+07, 1.0775566e+00, 8.3698189e+05, 5.4168141e+07]), array([2.51971647, 0.15534429, 0.16175095, 0.58830135]), array([3.25003424, 0.88454977, 2.11759278, 3.43132612]), array([1.28624888e+07, 1.57642952e+00, 2.65220091e+05, 1.28624894e+07]), array([3.70511136, 0.1267157 , 0.13003167, 0.69541018]), array([inf, inf, inf, inf]), array([2.98854647, 0.26336229, 0.27434433, 0.70267842]), array([8.58500684e+06, 1.17038140e+00, 2.22785401e+05, 8.58500761e+06]), array([4.43679253, 0.05695071, 0.07295168, 0.5743124 ]), array([6.95246398, 1.18350222, 5.35588665, 7.52290561]), array([4.29154103e+05, 5.59649250e+01, 1.20617634e+02, 5.75429381e-01]), array([6.78028246e+07, 1.16896922e+01, 1.64617437e+06, 6.78028246e+07]), array([1.44243146e+07, 5.65800825e+03, 9.80969251e+03, 8.41672546e-01]), array([4.61399311, 0.04822702, 0.05762693, 0.64214215]), array([2.82030379e+05, 4.83271371e+01, 1.50675957e+02, 6.99046501e-01]), array([4.79392531, 0.23047139, 0.23659779, 0.89971115]), array([2.00544902, 0.583721  , 0.70165408, 1.06776127]), array([4.62492805, 0.10638189, 0.100615  , 0.73681947]), array([5.35379344, 0.2001752 , 0.18523656, 0.847646  ]), array([8.3189849 , 0.06529488, 0.08809293, 0.64200334]), array([2.92543769, 0.89676091, 0.95190503, 0.84321899]), array([3.55550271e+03, 8.51078321e-01, 3.45239646e+02, 3.55645906e+03]), array([2.41865725, 0.49355268, 0.55718469, 0.99033599]), array([2.90376064, 0.08626716, 0.08909727, 0.60512498]), array([2.7690429 , 0.48238869, 0.50493854, 0.68783896]), array([3.72282979, 0.11866941, 0.08357004, 0.53582569]), array([36.00203593,  2.56484508, 32.29486679, 36.62795751]), array([1.21553050e+06, 6.06983342e+01, 4.18133207e+02, 5.44313749e-01]), array([3.70893393, 0.05926022, 0.06017153, 0.6410429 ]), array([12.93524776,  0.07417562,  0.13442364,  0.67682427]), array([4.89376072, 2.34327393, 6.62866723, 5.30194383]), array([1.64197698e+06, 1.30820931e+02, 3.52527065e+02, 6.24284238e-01]), array([ 4.37406453,  4.88554471, 10.85437193,  4.76444231]), array([3.79907013, 0.07170942, 0.07326962, 0.66931616]), array([5.4949698 , 0.06483923, 0.06566782, 0.65290686]), array([3.15205992, 0.07304442, 0.07517424, 0.61882356]), array([3.6608001 , 0.05825942, 0.05486973, 0.57224841]), array([2.10031423, 1.46166333, 2.27858748, 1.76018645]), array([3.26487201, 0.10972547, 0.11352341, 0.69840149]), array([       nan, 1.05922938,        nan,        nan]), array([3.1002132 , 0.04234359, 0.0425259 , 0.51501222]), array([3.31896598, 0.07694939, 0.07713281, 0.55693566]), array([16.21236454,  0.23075583,  0.17250958,  0.57320588]), array([4.09080885, 0.0625595 , 0.05718586, 0.64441392]), array([3.16491667, 0.05972118, 0.06131095, 0.5865593 ]), array([2.29244654, 1.27827262, 1.91607613, 1.78772466]), array([3.36075319, 0.07631728, 0.07708425, 0.5693886 ]), array([3.51841584, 0.05911359, 0.06307192, 0.55589626]), array([5.41972155e+07, 7.46535980e-01, 5.67471153e+05, 5.41972164e+07]), array([5.57024488, 0.06363743, 0.04792136, 0.63099591]), array([5.67555584, 0.04277514, 0.07099714, 0.59363955]), array([3.55746514, 0.19165964, 0.19717075, 0.69556094]), array([4.40112623, 0.06240711, 0.06236891, 0.72913616]), array([1.60208191e+07, 6.24603064e+02, 1.73213085e+03, 6.18558443e-01]), array([3.62707065, 0.05014524, 0.04970365, 0.59141392]), array([4.51525965e+07, 9.83761608e-01, 5.55949909e+05, 4.51525968e+07]), array([2.88421665, 0.08100934, 0.08333079, 0.50151548]), array([3.04853234, 0.1158195 , 0.12024055, 0.68520721]), array([3.21210579, 0.1231461 , 0.12702159, 0.65553878]), array([2.29407271, 0.29850417, 0.3210678 , 0.72321282]), array([4.07491289, 0.05111561, 0.0585869 , 0.57935162]), array([5.23233845e+06, 5.75189496e+03, 8.27522401e+03, 8.34697552e-01]), array([4.2366556 , 0.09626648, 0.09785486, 0.72916443]), array([2.89049497, 0.05524996, 0.05552073, 0.48447496]), array([25.40825694,  0.03818441,  0.16643017,  0.57483108]), array([4.05954313, 0.06900862, 0.07024454, 0.69319265]), array([2.95795456, 0.28038225, 0.29648637, 0.81440862]), array([1.94179993e+07, 2.49861242e+00, 6.99531418e+05, 1.94179999e+07]), array([3.72565232, 0.0462333 , 0.03117377, 0.57045798]), array([47.59855823,  0.09514281,  0.14132658,  0.49958513]), array([6.12657247, 0.04727834, 0.06173147, 0.71716947]), array([1.60569149, 1.63055676, 2.42211538, 1.23643164]), array([4.71193941, 0.05416955, 0.04280213, 0.71565803]), array([inf, inf, inf, inf]), array([4.01968157, 0.06066794, 0.06208909, 0.68082618]), array([28.55428783,  2.16640497, 20.63423281, 29.14199447]), array([4.04571177, 0.1307496 , 0.13421195, 0.76110318]), array([2.15891974, 1.08133731, 1.53946073, 1.61639398]), array([4.06404727, 0.06571207, 0.0523168 , 0.62621245]), array([5.27654187, 0.07557651, 0.07696342, 0.86977125]), array([3.8610821 , 3.23998279, 9.61232412, 4.21486084]), array([3.01686828, 0.1083693 , 0.11215882, 0.65000816]), array([inf, inf, inf, inf]), array([5.57212884, 0.05218785, 0.07013171, 0.6979455 ]), array([9.12181673, 0.49344471, 0.21407122, 0.75926008]), array([2.79432609, 1.24279467, 1.45208685, 1.30470726]), array([27.93359348,  2.66343083, 24.92583627, 28.49913614]), array([4.55130995, 0.07770734, 0.08018536, 0.75422241]), array([3.75800843, 0.06935639, 0.07137426, 0.60014594]), array([3.81872766, 0.05519413, 0.04367257, 0.58556584]), array([3.66357001, 0.10262875, 0.10514755, 0.66024791])]
median x_centre of target 88:  0.16171468640736192
standard deviation from x0=1640.42A of target 88:  0.9837612007718648
The Lya SNR FOR TARGET 88 is:  4.717403
INTEGRATED FLUX OF Lya FOR TARGET 88 is:  54.78953664051369
#############################################################################################################
THE He-II SNR FOR TARGET 204 is:  3.2824476
INTEGRATED FLUX OF TARGET He-II of 204 is:  2.223904
THE He-II FWHM FOR TARGET 204 IS:  0.9488207490951037
amp_204 = 2.2021410334808458 +/- None
cen_204 = 1639.36181640625 +/- None
sigma_204 = 0.40292707338604017 +/- None
continuum_204 = 0.6325121 +/- None
THE He-II EW FOR TARGET 204 IS:  0.8501436102354819
std-errors gaussian of target 204 (amp, cen, sigma, continuum):  [array([2.66486781e+06, 2.01689526e+00, 1.94172476e+05, 2.66486793e+06]), array([0.7341174 , 0.09670661, 0.09768551, 0.12200555]), array([7.74326467e+04, 1.87455626e+03, 1.70823564e+03, 1.60359130e-01]), array([0.79434077, 0.11502666, 0.1174512 , 0.13949419]), array([0.60997061, 0.11828301, 0.12240656, 0.13110823]), array([0.44891436, 0.64731372, 0.73968819, 0.19439457]), array([0.62678262, 0.7209885 , 0.7782172 , 0.20248166]), array([6.51901748e+06, 2.03974527e+00, 3.16084992e+05, 6.51901753e+06]), array([0.74391148, 0.09586328, 0.09794184, 0.13030491]), array([0.35329135, 0.34895514, 0.41269416, 0.17120581]), array([0.70516326, 0.12817813, 0.13162301, 0.13292598]), array([0.60171728, 0.09029055, 0.0925756 , 0.11014143]), array([0.48330657, 0.9552586 , 1.87092186, 0.4821085 ]), array([0.50557689, 0.10004277, 0.10360291, 0.10978575]), array([1.13677087e+06, 1.97171473e+06, 4.15698621e+05, 1.25125958e+03]), array([8.85049698e+06, 1.11870845e+00, 5.64696052e+05, 8.85049700e+06]), array([1.02360696e+07, 8.38730830e+00, 1.63147459e+06, 1.02360697e+07]), array([0.81371505, 0.10268662, 0.10430124, 0.13719332]), array([5.07778009e+05, 2.05130646e+02, 1.07673323e+03, 1.63565132e-01]), array([1.14118810e+07, 1.62006003e+02, 3.84233993e+06, 1.14118798e+07]), array([0.48306835, 0.72578302, 0.79816181, 0.17451504]), array([       nan, 1.44498656,        nan,        nan]), array([0.88680417, 0.0684608 , 0.07575709, 0.13036658]), array([0.36167358, 0.2723003 , 0.30369331, 0.14024184]), array([0.59671976, 0.11540982, 0.11863903, 0.11495816]), array([inf, inf, inf, inf]), array([1.1831545 , 0.06845741, 0.08773727, 0.14526732]), array([0.76311831, 0.26439851, 0.27071493, 0.13740323]), array([0.56933467, 0.23516386, 0.24609223, 0.1411368 ]), array([0.47040797, 0.21383054, 0.22697264, 0.1339158 ]), array([2.00894154e+06, 2.70598472e+00, 4.46288969e+05, 2.00894168e+06]), array([0.6345085 , 0.10280808, 0.10572724, 0.12317155]), array([9.36791287e+05, 4.83892552e+02, 2.54442743e+03, 1.52365393e-01]), array([0.433509  , 0.19726887, 0.2125699 , 0.13842011]), array([0.66855505, 0.10030224, 0.10310434, 0.12871363]), array([4.60097702e+06, 2.54044149e+01, 1.86550920e+06, 4.60097706e+06]), array([0.59089129, 0.15238256, 0.15866802, 0.13807034]), array([0.54085223, 0.19624684, 0.2047164 , 0.12924766]), array([1.23839161e+06, 1.30793422e+00, 1.65173963e+05, 1.23839176e+06]), array([0.50713303, 0.18154333, 0.19181827, 0.13860678]), array([0.54884664, 0.15370374, 0.15980327, 0.12580891]), array([2.57400370e+04, 1.91745566e+01, 1.19359715e+02, 1.18825157e-01]), array([2.57116375e+06, 1.37096695e+00, 3.43742644e+05, 2.57116394e+06]), array([0.40453125, 0.50484814, 0.57930168, 0.17679457]), array([1.07308756e+07, 3.75812028e+00, 1.22378249e+06, 1.07308757e+07]), array([0.81653919, 0.17175154, 0.17691545, 0.16307523]), array([0.40114844, 0.92880833, 1.46509948, 0.33845198]), array([0.46945214, 0.17157579, 0.18115695, 0.12745794]), array([1.04428074e+07, 1.02340557e+00, 8.84115979e+05, 1.04428075e+07]), array([0.63373785, 0.17277937, 0.1811789 , 0.16065126]), array([0.56828802, 0.0557199 , 0.04947553, 0.08693147]), array([0.57525736, 0.24838685, 0.26156031, 0.15225052]), array([1.13023258, 0.11188784, 0.12547834, 0.16257173]), array([0.79281328, 0.11347376, 0.11549984, 0.13535087]), array([1.01412467, 0.08154574, 0.09219591, 0.14556365]), array([0.78365758, 0.08767875, 0.08928123, 0.13281139]), array([0.51863181, 0.14633635, 0.15233819, 0.12084431]), array([inf, inf, inf, inf]), array([0.47584042, 1.10221703, 2.18895543, 0.47360466]), array([inf, inf, inf, inf]), array([0.64356138, 0.17389974, 0.18070995, 0.14655349]), array([0.54424423, 0.29758228, 0.31478762, 0.15032607]), array([0.52305263, 0.22254615, 0.23628088, 0.14920872]), array([        nan, 14.89299836,         nan,         nan]), array([0.61614118, 0.11033677, 0.11383589, 0.12634618]), array([3.05150678, 0.67562088, 6.28052669, 3.157286  ]), array([5.64449748e+06, 4.24984281e+00, 1.18881214e+06, 5.64449759e+06]), array([0.55243339, 0.12233159, 0.1277379 , 0.13357778]), array([0.39856451, 0.54165074, 0.62702806, 0.17991035]), array([0.36389042, 0.40998005, 0.51421218, 0.2066966 ]), array([0.40515543, 0.98191757, 1.78249404, 0.38772196]), array([0.55070592, 0.27095533, 0.28467055, 0.14242395]), array([0.61567506, 0.14500047, 0.15105557, 0.1447434 ]), array([0.72760575, 0.07942288, 0.08433783, 0.11198723]), array([8.79026196e+04, 4.09046218e+02, 1.00684071e+03, 1.19873436e-01]), array([ 2.68772437,  1.44203644, 10.4218572 ,  2.80683182]), array([0.85084891, 0.12669573, 0.12816109, 0.14258218]), array([0.64931596, 0.2024925 , 0.21085813, 0.15184141]), array([           nan,            nan,            nan, 5.86647283e-07]), array([0.77098528, 0.09653681, 0.09912766, 0.12770392]), array([1.78319795, 0.08493762, 0.08759801, 0.13908495]), array([0.85164109, 0.10761558, 0.12183524, 0.1133044 ]), array([0.41538265, 0.56924335, 0.66197925, 0.19231513]), array([0.72780257, 0.10169292, 0.09900238, 0.11534146]), array([       nan, 6.86268984,        nan,        nan]), array([ 3.81939549,  3.91691088, 13.67121131,  3.90312404]), array([0.3999005 , 0.2690778 , 0.29467283, 0.14115087]), array([0.67015876, 0.09715481, 0.09987978, 0.12936751]), array([0.58576079, 0.20330824, 0.21317436, 0.14835846]), array([0.66599559, 0.12666289, 0.13035675, 0.1310126 ]), array([0.84111958, 0.122331  , 0.12419838, 0.14282577]), array([0.47548278, 0.40380367, 0.44507015, 0.17383095]), array([0.69867417, 0.08136405, 0.08231575, 0.11698996]), array([1.11512163, 0.10110615, 0.06414759, 0.15303665]), array([0.79983487, 1.15014301, 3.41819202, 0.86691133]), array([0.39999872, 0.80148381, 1.14171744, 0.28805972]), array([0.42629568, 0.46702901, 0.52586524, 0.17259356]), array([0.71738567, 0.11821272, 0.12113706, 0.12559584]), array([0.67478451, 0.07756959, 0.07953483, 0.12198752]), array([4344210.6789863 ,    7700.24711551, 2802128.97522786,
       4343958.94717591])]
median x_centre of target 204:  0.14512630578690922
standard deviation from x0=1640.42A of target 204:  0.9824920478504594
The Lya SNR FOR TARGET 204 is:  16.23129
INTEGRATED FLUX OF Lya of 204 is:  36.546649408119265
###############################################################################################################
The He-II SNR FOR TARGET 435 is:  4.929654
INTEGRATED FLUX OF He-II of 435 is:  3.0806642
The He-II FWHM FOR TARGET 435 is:  0.768392330927882
amp_435 = 3.7735498000765424 +/- 2.001471414619817
cen_435 = 1637.7137087035621 +/- 0.23956633530898352
sigma_435 = 0.326306178915588 +/- 0.17236937696130283
continuum_435 = 1.3811159 +/- 0
THE He-II EW FOR TARGET 435 IS:  0.4721017733750371
std-errors gaussian of target 435 (amp, cen, sigma, continuum):  [array([       nan, 1.12808608,        nan,        nan]), array([1.72094206, 2.53833861, 1.98701971, 0.2922139 ]), array([1.13904583, 0.12335255, 0.12730475, 0.23467429]), array([1.70369101, 0.10181908, 0.10375511, 0.2744197 ]), array([0.85376581, 1.29933604, 1.66117304, 0.49628957]), array([1.3822111 , 0.62003899, 0.65988567, 0.40231029]), array([1.73685471, 0.18293797, 0.18806799, 0.33462005]), array([1.19992839, 4.11732427, 8.73709544, 1.22871138]), array([1.43273878, 0.25089854, 0.25985868, 0.3114001 ]), array([       nan, 3.57794713,        nan,        nan]), array([1.81048697, 0.17919338, 0.18395354, 0.33956615]), array([2.42654722e+07, 1.19966826e+00, 6.32616781e+05, 2.42654726e+07]), array([2.74292336, 0.06737365, 0.04076081, 0.20539352]), array([4.19307865e+06, 4.41032833e+00, 5.61172222e+05, 4.19307899e+06]), array([           nan, 2.00007757e+06,            nan, 2.64511925e-01]), array([0.660877  , 0.81336304, 1.09477016, 0.44844176]), array([1.60685067, 0.1413539 , 0.14447476, 0.27484899]), array([1.03180682e+07, 1.41134041e+00, 4.37200471e+05, 1.03180684e+07]), array([0.97306091, 1.13336965, 2.26779917, 0.97263881]), array([0.84037393, 1.75986792, 2.83703869, 0.72724209]), array([2.30317336, 0.1966852 , 0.20187642, 0.33733369]), array([1.76723718, 0.18128939, 0.18553156, 0.31214316]), array([2.29645094e+07, 9.67029464e-01, 6.80896306e+05, 2.29645096e+07]), array([0.56177283, 1.48569924, 2.1591308 , 0.43239289]), array([1.71698131, 0.25324306, 0.2607306 , 0.33970881]), array([       nan, 1.54223073,        nan,        nan]), array([0.86952744, 1.6084961 , 2.53038794, 0.7668419 ]), array([0.76251892, 2.12448603, 3.14912806, 0.60352416]), array([2.96044056e+07, 8.94999009e+00, 1.32092987e+06, 2.96044054e+07]), array([inf, inf, inf, inf]), array([0.81856955, 0.49465103, 0.57694515, 0.38080595]), array([1.71142308, 0.0984881 , 0.10029023, 0.28116014]), array([inf, inf, inf, inf]), array([0.91720421, 1.07133346, 1.46913848, 0.6439926 ]), array([3.22277013, 0.97895654, 5.13681324, 3.45511022]), array([inf, inf, inf, inf]), array([2.31396716e+07, 1.42821464e+00, 1.02208118e+06, 2.31396719e+07]), array([0.90317962, 0.7774707 , 0.97086375, 0.52565557]), array([1.38469944, 0.10534319, 0.10796172, 0.25113642]), array([2.29640699e+07, 9.24706407e-01, 6.43014780e+05, 2.29640699e+07]), array([inf, inf, inf, inf]), array([0.82413288, 1.32205042, 1.56986316, 0.40656856]), array([       nan, 1.49105066,        nan,        nan]), array([       nan, 5.45906045,        nan,        nan]), array([inf, inf, inf, inf]), array([       nan, 4.87804112,        nan,        nan]), array([1.61533678e+07, 1.31620838e+00, 5.51958272e+05, 1.61533679e+07]), array([15.62170733,  0.78297008, 12.24173866, 15.87729941]), array([inf, inf, inf, inf]), array([0.59501837, 1.30507615, 1.81044705, 0.42278892]), array([1.11203786e+07, 1.37985221e+01, 1.30629139e+06, 1.11203789e+07]), array([       nan, 1.42357346,        nan,        nan]), array([0.84740197, 0.71752805, 0.88994746, 0.47031541]), array([           nan, 7.81692247e+07,            nan, 2.82946719e-01]), array([1.14477223, 0.18794842, 0.19375116, 0.2319716 ]), array([       nan, 0.42879166,        nan,        nan]), array([1.58261322, 0.11227079, 0.1147698 , 0.26556282]), array([4.57646455e+06, 1.04223289e+02, 1.55144233e+03, 2.65925190e-01]), array([       nan, 4.67077138,        nan,        nan]), array([0.63171991, 0.82813915, 0.95354749, 0.27963044]), array([0.89428137, 1.0006344 , 1.99890104, 0.92955638]), array([1.86919178e+07, 9.23537026e+00, 1.12438245e+06, 1.86919178e+07]), array([1.5613957 , 0.20440904, 0.21042783, 0.30865163]), array([0.86924333, 0.50975639, 0.98512307, 0.85264923]), array([1.39518605, 0.16446686, 0.1695565 , 0.28220339]), array([2.05510452, 0.22244592, 0.2294925 , 0.30776077]), array([0.64446482, 2.57273568, 2.72332384, 0.3323206 ]), array([0.95601067, 1.08735458, 1.40411822, 0.57796962]), array([2.77633769e+07, 8.23264826e+00, 1.74267261e+06, 2.77633768e+07]), array([1.51563256, 0.04357409, 0.04453275, 0.25048243]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([9.76637081e+06, 1.75699727e+00, 3.93293727e+05, 9.76637104e+06]), array([1.70524288e+07, 5.47447666e-01, 3.54755109e+05, 1.70524290e+07]), array([1.19080823, 0.13024294, 0.13383948, 0.22754154]), array([       nan, 3.61794684,        nan,        nan]), array([0.85790074, 0.66735745, 1.30440312, 0.84710779]), array([2.21060888e+07, 8.02426790e-01, 5.27345086e+05, 2.21060890e+07]), array([1.60156276, 0.29232037, 0.29909549, 0.28163052]), array([0.74158188, 0.23811754, 0.26185713, 0.2678731 ]), array([       nan, 1.63745102,        nan,        nan]), array([0.98475277, 1.56175735, 1.76444598, 0.40310637]), array([0.81787511, 1.66036268, 2.55793615, 0.67845302]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([9.74606486e+05, 5.25755233e+02, 8.78393881e+02, 2.76767660e-01]), array([0.58458094, 0.5214795 , 0.63780654, 0.32323572]), array([inf, inf, inf, inf]), array([0.74537587, 0.86499071, 1.15165775, 0.49107793]), array([inf, inf, inf, inf]), array([0.68879597, 0.70589995, 1.2319086 , 0.63261603]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([0.85950072, 1.10204491, 1.67789557, 0.68194855]), array([0.85884222, 1.00561188, 1.20746374, 0.45327196]), array([1.61623154, 0.1351381 , 0.13805465, 0.26460385]), array([5.87491587e+06, 9.68439154e-01, 3.46591451e+05, 5.87491607e+06]), array([0.87421944, 0.19771538, 0.20596734, 0.20555437]), array([1.14105076, 1.22050714, 2.76876864, 1.19932735]), array([1.53798895, 0.10533257, 0.10845127, 0.23568589])]
median x_centre of target 435:  0.16406190727003156
standard deviation from x0=1640.42A of target 435:  1.0090755966262706
The Lya SNR FOR TARGET 435 is:  84.764755
INTEGRATED FLUX OF Lya of 435 is:  126.11863
###############################################################################################################
The He-II SNR FOR TARGET 22429 is:  2.6729734
INTEGRATED FLUX OF He-II of 22429 is:  2.2319293
The He-II FWHM FOR TARGET 22429 is:  1.4001343962939574
THE He-II EW FOR TARGET 22429 IS:  0.6155119081563235
std-errors gaussian of target 22429 (amp, cen, sigma, continuum):  [array([ 35.69949599, 176.49138719, 112.80063552,  10.64464113]), array([0.64819367, 0.41466271, 0.46078719, 0.24837984]), array([6.99778803e+05, 1.74569330e+01, 1.11510143e+06, 6.99778992e+05]), array([0.85261963, 0.11568176, 0.11831753, 0.15279572]), array([ 0.65322092, 11.34868908, 20.04536362,  0.60233956]), array([5.97630394e+05, 3.38005977e+02, 1.68892909e+03, 2.17196863e-01]), array([       nan, 1.41067251,        nan,        nan]), array([1.23754463e+06, 6.98724170e+02, 2.15726219e+03, 1.72304165e-01]), array([1.77143398e+06, 3.16960195e+00, 2.83563840e+05, 1.77143415e+06]), array([0.68735034, 2.10169469, 3.85279909, 0.71256714]), array([1.57086649e+07, 1.64760231e+00, 6.46258397e+05, 1.57086648e+07]), array([7.73877923e+06, 6.07895171e+00, 7.72631646e+05, 7.73877929e+06]), array([1.56449088, 0.27157486, 0.16510096, 0.19537167]), array([7.13028355, 0.08899788, 0.31736058, 0.15938569]), array([0.51693015, 0.70439619, 0.83447162, 0.25785674]), array([0.80199437, 0.19576903, 0.204876  , 0.19887189]), array([0.52425913, 0.72004028, 1.22518609, 0.47804432]), array([1.40028810e+07, 1.85589769e+00, 6.24289142e+05, 1.40028810e+07]), array([1.88532709e+06, 7.04913763e+02, 1.83324130e+03, 1.70215797e-01]), array([0.73345976, 0.48835827, 0.53285746, 0.25348177]), array([0.58025076, 2.57216526, 4.11450489, 0.52921407]), array([0.57932931, 0.46112874, 0.50420159, 0.20482728]), array([0.95519188, 0.19354859, 0.19954942, 0.19346668]), array([1.20510718, 0.11818926, 0.12025733, 0.19685328]), array([1.21727087e+07, 1.96393735e+00, 7.08878876e+05, 1.21727088e+07]), array([1.01081979e+07, 1.43415669e+00, 5.81507186e+05, 1.01081979e+07]), array([0.51115987, 1.86365848, 2.83280849, 0.4299222 ]), array([0.83356248, 0.34483139, 0.36362422, 0.22371336]), array([0.45171807, 0.59207809, 0.7165479 , 0.24345703]), array([0.57992123, 1.03296392, 1.17539594, 0.24530293]), array([1.58313057, 0.12134686, 0.11878624, 0.21690425]), array([0.9954241 , 0.33194531, 0.34750059, 0.24792305]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([1.38489640e+05, 1.52446164e+01, 7.98524387e+04, 1.38489754e+05]), array([0.51180043, 1.1916123 , 1.58797557, 0.34436886]), array([1.66674744e+07, 1.79138137e+00, 1.05655447e+06, 1.66674744e+07]), array([0.65049452, 0.66064897, 0.72957663, 0.24717922]), array([0.68121717, 0.24680977, 0.26487077, 0.21137379]), array([0.66937103, 0.72296186, 0.86586112, 0.3370048 ]), array([0.67607493, 0.31551063, 0.33483866, 0.19225227]), array([1.24973437, 0.16460361, 0.16970478, 0.25347274]), array([1.23017909, 0.48063475, 0.51318146, 0.19267263]), array([1.37742955, 0.12790614, 0.13523056, 0.20859632]), array([0.53389198, 0.54021928, 0.60339227, 0.20846614]), array([       nan,        nan,        nan, 0.18214103]), array([0.79170079, 0.22899099, 0.24125815, 0.21064529]), array([1.17936275e+07, 9.60669262e-01, 6.22800069e+05, 1.17936276e+07]), array([0.7365449 , 2.55451866, 4.57682738, 0.78266654]), array([0.61521427, 0.2183093 , 0.23616549, 0.20189529]), array([8.36810331e+06, 2.10106611e+00, 8.13326712e+05, 8.36810342e+06]), array([0.72811059, 0.55577522, 0.59630586, 0.22553187]), array([1.67260306, 0.12223013, 0.08570065, 0.21367379]), array([6.82805302e+05, 1.27841433e+02, 1.31414782e+03, 1.89722874e-01]), array([0.6776074 , 0.50835793, 0.57966448, 0.29205232]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([0.96059795, 0.26364324, 0.2775983 , 0.25412629]), array([1.53128704, 0.13893308, 0.20799995, 0.17850343]), array([8.60153414e+06, 5.75962332e+00, 9.97384598e+05, 8.60153418e+06]), array([inf, inf, inf, inf]), array([ 7.45631761,  2.85336551, 23.42799505,  7.636759  ]), array([3136255.4263169 ,    6594.42740159, 3012377.46525094,
       3136123.06146188]), array([34.92281456,  1.87408355, 43.9029292 , 35.09958306]), array([0.50230846, 0.63682108, 0.78282533, 0.28450946]), array([0.84993627, 0.16192474, 0.1674484 , 0.18075232]), array([0.57860177, 1.4996371 , 2.47456996, 0.50690565]), array([0.93070074, 0.1413255 , 0.1453566 , 0.18150284]), array([1.26714541e+08,            nan,            nan, 2.12069207e-01]), array([0.86964434, 0.33314762, 0.34375118, 0.17867304]), array([0.63091547, 0.22547437, 0.23514702, 0.15036495]), array([1.11356202e+07, 1.81731041e+01, 3.10316494e+06, 1.11356203e+07]), array([0.9710757 , 0.11712161, 0.11970018, 0.17561761]), array([       nan, 1.65144955,        nan,        nan]), array([0.63884935, 1.91102378, 3.16585728, 0.56138627]), array([0.57308239, 0.75856143, 0.95555537, 0.3284963 ]), array([1.51534480e+07, 1.78238372e+00, 5.45219630e+05, 1.51534480e+07]), array([2.86625281, 3.49147984, 2.12617144, 0.25430813]), array([0.34951232, 0.86299953, 1.18939724, 0.24579717]), array([       nan, 1.30011427,        nan,        nan]), array([0.7792489 , 0.7594337 , 0.83347842, 0.27837029]), array([1.02751619, 0.15735359, 0.16216735, 0.2070147 ]), array([5.72042982e+06, 5.62541024e+06, 1.01615802e+06, 2.01043116e-01]), array([5.29924984e+06, 3.89139820e+00, 8.86355259e+05, 5.29925003e+06]), array([inf, inf, inf, inf]), array([1.04894834e+05, 3.72092096e+01, 3.93796375e+04, 1.04894020e+05]), array([inf, inf, inf, inf]), array([0.45757999, 0.50739747, 0.62355227, 0.24570903]), array([1.09569378, 0.20050239, 0.20633944, 0.21487356]), array([0.68134528, 0.76786483, 0.85101575, 0.26169471]), array([inf, inf, inf, inf]), array([1.65769878e+07, 8.91591270e+00, 1.64195342e+06, 1.65769879e+07]), array([0.46086675, 0.47920905, 0.56898323, 0.2272735 ]), array([0.81708077, 0.34692912, 0.37122527, 0.24807738]), array([0.77197013, 0.16339363, 0.17000778, 0.17867774]), array([0.49482573, 0.50553678, 0.90373904, 0.462706  ]), array([0.68642033, 0.1741806 , 0.18076181, 0.15349389]), array([0.69223821, 0.96167261, 1.54785451, 0.60043966]), array([0.55884292, 1.47310211, 2.32911518, 0.4795584 ]), array([0.51335197, 0.54902332, 0.66351143, 0.27114312])]
median x_centre of target 22429:  -0.0847655134054095
standard deviation from x0=1640.42A of target 22429:  1.0574385941989213
The Lya SNR FOR TARGET 22429 is:  13.586418
INTEGRATED FLUX OF Lya of 22429 is:  36.714977191455546
###############################################################################################################
The He-II SNR FOR TARGET 538 is:  4.3582087
INTEGRATED FLUX OF He-II of 538 is:  1.2712905
The He-II FWHM FOR TARGET 538 is:  0.9698953698955014
amp_538 = 1.2314305531614236 +/- 0.9299541524300259
cen_538 = 1639.4544583740314 +/- 0.4147971260177174
sigma_538 = 0.4118766408253307 +/- 0.33234142034219916
continuum_538 = 1.6347432 +/- 0
THE He-II EW FOR TARGET 538 IS:  0.15023197781105327
std-errors gaussian of target 538 (amp, cen, sigma, continuum):  [array([0.76430417, 5.08451548, 5.41009544, 0.28129008]), array([0.82139717, 0.89766049, 1.09887383, 0.44317785]), array([2.36406076e+07, 1.61920530e+01, 2.67711853e+06, 2.36406077e+07]), array([       nan, 0.83733051,        nan,        nan]), array([0.83436579, 0.50354779, 0.53811148, 0.25087364]), array([0.65554865, 0.81983607, 1.07842659, 0.41210331]), array([0.92686543, 1.35616172, 2.64009634, 0.91282417]), array([0.63448798, 0.50435887, 0.59305507, 0.30182342]), array([0.80142714, 0.6491594 , 0.78601633, 0.41569194]), array([83.12811337,  0.23948648,  1.43225311,  0.2087421 ]), array([0.77303782, 0.92370645, 1.86666142, 0.77671799]), array([1.35382824, 0.1566289 , 0.15857412, 0.2060742 ]), array([2.51176151e+05, 2.79472759e+00, 5.69669407e+04, 2.51176461e+05]), array([inf, inf, inf, inf]), array([0.43893997, 1.46536927, 2.01667621, 0.29911741]), array([59.08807949,  1.51022185, 65.19571265, 59.31741155]), array([0.80471794, 1.03044418, 1.62680567, 0.67383043]), array([3.22016806e+06, 1.83500523e+03, 2.19930144e+02, 4.25191668e-01]), array([5.62520635e+05, 3.64819559e+01, 4.15073593e+02, 1.88927634e-01]), array([1.71425722e+06, 2.62560885e+03, 3.86951621e+03, 2.08347919e-01]), array([1.15959625, 0.17755244, 0.18137902, 0.19603901]), array([6.79240160e+05, 7.17486115e+02, 1.15253198e+03, 3.25336191e-01]), array([0.81271642, 0.79946523, 1.65176507, 0.8227807 ]), array([1.26625314, 5.52216341, 4.32873261, 0.28998359]), array([0.7528995 , 2.0716658 , 3.63838111, 0.72586726]), array([9.82406408e+06, 2.16422872e+00, 6.21467893e+05, 9.82406413e+06]), array([2.04801832e+07, 1.96093859e+00, 9.12701016e+05, 2.04801834e+07]), array([       nan, 1.25337856,        nan,        nan]), array([3.89793581e+06, 3.20881050e+03, 4.06141410e+03, 2.99040344e-01]), array([1.26261143, 1.08970559, 2.06080469, 1.23905756]), array([2.69244947, 0.99736213, 0.64316089, 0.20586007]), array([2.7856325 , 1.16120568, 4.75294545, 2.95980338]), array([0.9586617 , 0.37057915, 0.39163834, 0.26241992]), array([1.16450301e+05, 2.61531958e+01, 7.51348321e+04, 1.16450383e+05]), array([0.58986648, 0.88294477, 1.12831803, 0.35001781]), array([0.86668916, 1.56759857, 2.44711935, 0.72282723]), array([1.49107866e+07, 5.07194623e+00, 1.49828141e+06, 1.49107870e+07]), array([18.5575884 ,  6.66370234, 59.94241765, 18.91788349]), array([1.47840649, 2.33693814, 5.54084741, 1.60708605]), array([1.72825942, 0.06810665, 0.05958899, 0.22678705]), array([3.24532561e+06, 4.70706115e+02, 7.95563189e+02, 2.10509422e-01]), array([inf, inf, inf, inf]), array([2.1799274 , 0.13640184, 0.12756314, 0.30208732]), array([2.38239945, 0.07079753, 0.04431138, 0.24779411]), array([3.08737684, 0.04853133, 0.09191579, 0.23390764]), array([0.80839297, 1.18803908, 1.55987615, 0.50834976]), array([1.08505314, 3.26731328, 5.88147878, 1.16909243]), array([0.67947482, 0.47293985, 0.52687184, 0.26207307]), array([0.89590647, 0.43084788, 0.48619042, 0.36617484]), array([0.90574869, 0.31378631, 0.32894491, 0.22884924]), array([1.38611493e+07, 1.97791112e+01, 3.80674733e+06, 1.38611493e+07]), array([2.13191887e+18, 2.20136307e+05, 1.36380354e+04, 2.17320264e+00]), array([8.34914965e+06, 1.04927657e+03, 2.43481595e+03, 1.71477725e-01]), array([4.78921805e+05, 6.83804191e+02, 7.91327058e+02, 2.87767970e-01]), array([0.81177202, 1.13360656, 1.37191088, 0.44057906]), array([0.8069076 , 1.69175586, 3.12734781, 0.77220797]), array([1.02535018, 0.4108958 , 0.43754591, 0.29983032]), array([2.45498592, 1.24660284, 0.87203978, 0.31624011]), array([0.5921349 , 1.0018732 , 1.25162657, 0.32298976]), array([2.17043515e+05, 3.16155291e+02, 1.99202932e+02, 2.53416514e-01]), array([5.88466946e+06, 3.52255737e+02, 2.17002564e+03, 1.56302685e-01]), array([8.61556286e+06, 4.09545179e+00, 1.06558455e+06, 8.61556314e+06]), array([4.39756527e+04, 2.74116979e+01, 4.32478211e+01, 1.84357229e-01]), array([0.86903379, 0.3142085 , 0.32700058, 0.20185442]), array([1.18605933e+07, 6.23015676e+00, 1.07551944e+06, 1.18605935e+07]), array([1.74665216, 0.06005017, 0.06758156, 0.21499859]), array([0.6142073 , 0.32626242, 0.36035491, 0.22701905]), array([       nan, 1.40790821,        nan,        nan]), array([1.28777051e+07, 2.80546494e+00, 1.00811676e+06, 1.28777055e+07]), array([0.98574988, 1.26240995, 1.42433249, 0.41217516]), array([1.1726856 , 0.35821247, 0.37341055, 0.27791898]), array([1.1098326 , 0.72517469, 0.7634683 , 0.29434589]), array([inf, inf, inf, inf]), array([8.63901232e+05, 6.12648314e+02, 1.23441349e+03, 2.91267551e-01]), array([0.69226429, 1.39860015, 1.96920107, 0.48919556]), array([1.10270315e+07, 2.28334948e+00, 6.09517039e+05, 1.10270316e+07]), array([4.03618742, 0.22896097, 0.2370052 , 0.28924057]), array([       nan, 4.04308227,        nan,        nan]), array([5.30915769, 0.15358033, 0.09323112, 0.19678655]), array([       nan, 1.49462494,        nan,        nan]), array([2.63309956e+06, 7.55172026e+00, 5.79384966e+05, 2.63309987e+06]), array([1.74971279, 2.31914725, 6.57223665, 1.89626485]), array([0.91055782, 0.45094665, 0.49661186, 0.33137424]), array([0.86457842, 1.12382954, 1.92394401, 0.78904368]), array([0.78895482, 3.11176663, 6.28970659, 0.79148892]), array([1.73006660e+07, 1.41909956e+00, 1.26370071e+06, 1.73006664e+07]), array([1.36477964e+07, 1.86353314e+00, 5.20470332e+05, 1.36477967e+07]), array([0.5318413 , 0.48367043, 0.59804146, 0.29131524]), array([2.48289542, 2.88551976, 8.34904309, 2.68194379]), array([1.23089448, 0.17054583, 0.17520816, 0.23465646]), array([3.48096158e+06, 1.30031472e+00, 2.33055166e+05, 3.48096173e+06]), array([2.05083716e+07, 2.51704890e+00, 8.57580612e+05, 2.05083716e+07]), array([0.53560395, 1.66786091, 2.08007292, 0.31337682]), array([0.97255745, 1.38812228, 3.1274687 , 1.01928714]), array([1.94695842e+06, 4.92666624e+02, 1.11636635e+03, 1.75528758e-01]), array([1.37009012, 0.05577706, 0.0518001 , 0.18687457]), array([7.05786229e+05, 1.55487169e+05, 2.58436296e+04, 2.29856113e-01]), array([1.16395564, 0.58972837, 0.71526869, 0.60656484]), array([1.17363863, 0.15889684, 0.16309882, 0.22012518]), array([1.93986701e+07, 4.01120807e+00, 1.70189105e+06, 1.93986702e+07])]
median x_centre of target 538:  -0.02640908761343688
standard deviation from x0=1640.42A of target 538:  1.0057827140362972
The Lya SNR FOR TARGET 538 is:  23.508476
INTEGRATED FLUX OF Lya of 538 is:  29.329256
###############################################################################################################
The He-II SNR FOR TARGET 5199 is:  2.636273
INTEGRATED FLUX OF He-II of 5199 is:  1.7388546
THE He-II FWHM FOR TARGET 5199 is:  1.1198897427332108
amp_5199 = 1.4586629866934535 +/- 0.669973689001953
cen_5199 = 1640.7473356992018 +/- 0.27794069492055584
sigma_5199 = 0.47557338621112855 +/- 0.23522091676585968
continuum_5199 = 0.8691366 +/- 0
THE He-II EW FOR TARGET 5199 IS:  0.492411779020104
std-errors gaussian of target 5199 (amp, cen, sigma, continuum):  [array([0.64304492, 0.14541363, 0.15158752, 0.15230508]), array([0.48453634, 0.28178001, 0.30191977, 0.14861675]), array([0.38815488, 0.44365364, 0.49491762, 0.15068084]), array([4.36381071e+25, 5.68812416e+04, 2.84290135e+03, 2.66146858e-01]), array([0.71844462, 1.25447705, 2.89366334, 0.76699934]), array([inf, inf, inf, inf]), array([       nan, 3.72875383,        nan,        nan]), array([inf, inf, inf, inf]), array([0.39598745, 0.55614312, 0.69600343, 0.22329163]), array([13.53233274, 18.78176553,  6.9600966 ,  0.20317506]), array([inf, inf, inf, inf]), array([2.10953812e+05, 2.75141356e+01, 3.25748745e+02, 1.43445522e-01]), array([0.41305495, 1.0732109 , 1.3182084 , 0.22267977]), array([0.98016895, 0.17293426, 0.17711733, 0.17568123]), array([0.68476568, 0.09184429, 0.09428082, 0.12899681]), array([0.4669928 , 0.85977598, 0.96313714, 0.18481851]), array([0.63711917, 0.27651759, 0.29051401, 0.16486658]), array([1.97433159, 0.23425651, 0.14478751, 0.14340516]), array([13.75774802,  1.90111189, 31.72227775, 13.9148289 ]), array([3.86459599e+06, 1.73022593e+01, 9.49248962e+05, 3.86459598e+06]), array([0.57843244, 0.41732246, 0.45562493, 0.20062198]), array([1.11868003, 1.71635729, 5.05468866, 1.21064752]), array([0.51657181, 0.24133718, 0.25448133, 0.13852249]), array([0.91676888, 0.1502081 , 0.15432502, 0.17516881]), array([0.35344783, 0.55449297, 0.72468785, 0.22548917]), array([0.69253835, 0.24692156, 0.25951963, 0.17976239]), array([0.6062    , 0.17729974, 0.18541228, 0.14905945]), array([331.60059479,   1.34992932, 322.45794623, 331.76182634]), array([5.79892059, 0.19569392, 0.20733039, 0.13625292]), array([0.80603122, 1.365391  , 3.87049536, 0.871724  ]), array([1.2825974 , 0.1287991 , 0.14508477, 0.13454922]), array([0.9466907 , 0.07978848, 0.08648289, 0.13714511]), array([0.71312283, 0.3558748 , 0.36672268, 0.14312233]), array([0.7847224 , 0.12488509, 0.12886702, 0.16113596]), array([0.45373118, 1.01998964, 1.37901759, 0.2995767 ]), array([0.6004526 , 0.50876248, 0.53891705, 0.16796902]), array([0.8489714 , 0.13749943, 0.14120532, 0.1595617 ]), array([0.39987439, 0.46933589, 0.54566715, 0.18325101]), array([0.39701787, 4.39620101, 6.0576606 , 0.32493174]), array([0.81358554, 0.14810526, 0.15229499, 0.15749405]), array([0.59177584, 1.55019593, 3.16675086, 0.59740634]), array([0.58220321, 0.28046137, 0.30135678, 0.18224987]), array([0.64695823, 0.43602762, 0.46293816, 0.18461493]), array([3.39849118, 0.73655468, 5.64814668, 3.52265437]), array([1.61377751e+29, 3.84747882e+05, 1.80508197e+04, 1.64259612e+00]), array([0.81252603, 0.33801345, 0.35146934, 0.18656231]), array([0.70609476, 0.12454576, 0.12776422, 0.12885793]), array([ 0.87974613, 17.45553096, 16.56650153,  0.62852658]), array([0.41594367, 0.46709965, 0.57185293, 0.22221765]), array([3.56097063, 0.88950569, 6.84459507, 3.71258332]), array([1.23425331e+07, 4.66811480e+00, 2.64663716e+06, 1.23425331e+07]), array([0.39773273, 0.6565476 , 0.85725885, 0.24963364]), array([2.84440045e+04, 2.10417821e+01, 1.13882131e+02, 1.15913557e-01]), array([0.83455989, 0.0770972 , 0.07466947, 0.12848378]), array([0.61681734, 1.17309349, 2.67490525, 0.64773567]), array([1.01481546, 0.25106886, 0.15225624, 0.15230571]), array([0.62534784, 0.3114705 , 0.32897749, 0.17032741]), array([0.76164283, 0.14038018, 0.14368756, 0.1376111 ]), array([1.32857016e+07, 2.26607115e+00, 9.97777556e+05, 1.32857016e+07]), array([0.71702185, 0.2732264 , 0.2835827 , 0.16062724]), array([0.36123689, 1.98470218, 2.65640181, 0.23420828]), array([0.68382989, 0.14283446, 0.14675746, 0.13060036]), array([0.53288763, 0.324587  , 0.3509618 , 0.17409227]), array([4.42120869e+07, 9.08780162e+07,            nan, 1.57036664e-01]), array([0.37417029, 0.52655929, 0.62024503, 0.1789756 ]), array([0.40380839, 0.96855255, 1.32045302, 0.28324966]), array([0.74750043, 0.13288546, 0.13732416, 0.1572925 ]), array([0.55227873, 0.6789831 , 0.77958204, 0.2418941 ]), array([0.6707415 , 0.31849372, 0.33411413, 0.17066487]), array([3.24469582e+06, 8.34668665e+06, 1.48538870e+06, 1.69385203e-01]), array([0.86871591, 1.78734693, 5.09442054, 0.94327827]), array([0.6007979 , 1.42696276, 3.36777415, 0.63859538]), array([0.75381078, 0.72166166, 2.00103896, 0.8080952 ]), array([0.81051347, 0.27428452, 0.28250733, 0.16132759]), array([0.76158996, 0.14077056, 0.14480556, 0.14857907]), array([           nan,            nan, 9.87648248e+09, 9.56284857e-02]), array([0.68486242, 0.17409892, 0.18061972, 0.15240332]), array([1.51026293, 0.08318849, 0.11816804, 0.14794211]), array([0.72466324, 0.18913003, 0.19498552, 0.14675446]), array([1.81098066e+02, 1.86840518e+00, 2.13012980e+00, 1.28559199e-01]), array([0.82281088, 0.2883547 , 0.30054818, 0.19458559]), array([0.64523096, 0.3600089 , 0.37872772, 0.16901778]), array([           nan, 2.19341644e+09,            nan, 1.33337319e-01]), array([0.70837425, 0.32524147, 0.34137284, 0.18133202]), array([5.16134294, 0.81006086, 9.38531696, 5.30986201]), array([0.67497539, 0.11965257, 0.12358269, 0.1408221 ]), array([0.60578532, 0.19251194, 0.20104608, 0.14667686]), array([0.43907486, 0.64428574, 0.75312016, 0.19539161]), array([9.15033817e+06, 1.12128380e+00, 5.49850275e+05, 9.15033830e+06]), array([0.37084451, 0.62735751, 0.98787781, 0.30521198]), array([2.42478665, 0.14909402, 0.19192901, 0.17443838]), array([1.2112308 , 0.10211977, 0.1158467 , 0.14460581]), array([0.97461827, 0.09912756, 0.0613076 , 0.14721081]), array([2.90610653e+05, 1.40601216e+02, 8.69302100e+02, 1.38361105e-01]), array([0.37787343, 1.45140236, 2.45491518, 0.3593233 ]), array([0.60422893, 0.27568118, 0.29208266, 0.16929303]), array([0.44341027, 0.4250264 , 0.47157098, 0.16767557]), array([0.67232426, 0.62008053, 0.66186456, 0.20028192]), array([0.48729846, 0.68795347, 0.77961587, 0.20273236]), array([0.54044089, 0.19723383, 0.20818013, 0.14631633])]
median x_centre of target 5199:  -0.06858742668342516
standard deviation from x0=1640.42A of target 5199:  0.9265770194750825
The Lya SNR FOR TARGET 5199 is:  2.6312444
[1.4867892 2.4333537 3.4993777 4.2562723 4.3182817 3.6513662 2.602533
 1.6163868]
INTEGRATED FLUX OF Lya of 5199 is:  6.914350472965452
###############################################################################################################
The He-II SNR FOR TARGET 23124 is:  2.874524
INTEGRATED FLUX OF He-II PEAK FOR 23124 IS:  1.7344417676576995
THE He-II FWHM FOR TARGET 23124 IS:  0.44629818546793065
amp_23124 = 3.6207803536824414 +/- 2.9431366153937626
cen_23124 = 1639.6855368215363 +/- 0.24366536748656384
sigma_23124 = 0.18952538917344955 +/- 0.15064073209974452
continuum_23124 = 1.9739583332836521 +/- 0.19818349079435918
THE He-II EW FOR TARGET 23124 IS:  0.19128209261383836
std-errors gaussian of target 23124 (amp, cen, sigma, continuum):  [array([0.63280193, 1.57129932, 1.86386449, 0.30957468]), array([        nan, 19.76285252,         nan,         nan]), array([14.59975072,  1.11130105, 22.44815365, 14.80760532]), array([0.44708044, 0.6645864 , 0.89202896, 0.2995361 ]), array([3.72717339e+04, 2.00047079e+01, 7.46464812e+01, 2.45786305e-01]), array([9.58784810e+06, 6.73640604e+00, 1.11099121e+06, 9.58784836e+06]), array([ 51.25293598,  15.40780753, 281.39107867,  51.42196872]), array([0.60716182, 1.33704232, 1.80813853, 0.40563571]), array([0.60766835, 0.43946573, 0.47558107, 0.19962924]), array([0.57566075, 0.7691238 , 0.95060778, 0.31645364]), array([0.89440016, 2.00718006, 4.07789074, 0.90756938]), array([1.74037306e+07, 1.22877740e+01, 2.37160148e+06, 1.74037308e+07]), array([0.89430773, 0.10888228, 0.11165107, 0.16402268]), array([3.78133493, 0.04195043, 0.12344207, 0.19539518]), array([1.77094602, 1.84277611, 8.73221149, 1.87649097]), array([2.131683  , 0.04739334, 0.08668533, 0.18175776]), array([0.65551712, 0.84010252, 0.94061769, 0.25881106]), array([1.03626247, 0.60186127, 0.6406774 , 0.30850222]), array([1.06266988, 0.07243331, 0.07405802, 0.17911814]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([0.64907297, 0.94102385, 1.14403802, 0.35389391]), array([6.73998344e+06, 3.19229520e+03, 3.37335348e+03, 1.87303272e-01]), array([1.89250167e+07, 2.64087701e+01, 3.34200080e+06, 1.89250166e+07]), array([7.93791723e+05, 7.51742331e+01, 4.86889582e+02, 1.68590145e-01]), array([0.63772832, 0.38973311, 0.4286363 , 0.23047656]), array([0.5566183 , 0.45340966, 0.52802373, 0.25644617]), array([inf, inf, inf, inf]), array([0.91342674, 0.22339079, 0.2301143 , 0.18247796]), array([1.06817865, 0.07918174, 0.08162671, 0.16620443]), array([0.59995138, 0.93469665, 1.02293156, 0.21094196]), array([1.29418633, 0.0808534 , 0.04181902, 0.18307731]), array([0.97188772, 0.42176285, 0.44221596, 0.24619487]), array([3.89063914e+04, 4.58203018e+01, 9.52617114e+01, 1.29008637e-01]), array([        nan, 16.76450234,         nan,         nan]), array([1.67050918e+07, 1.24524206e+00, 8.93102875e+05, 1.67050919e+07]), array([5.99888294e+06, 1.18501424e+00, 5.10873974e+05, 5.99888312e+06]), array([1.26803313, 0.0659393 , 0.03167955, 0.16254973]), array([1.16397132e+07, 5.59621410e+00, 2.69410627e+06, 1.16397134e+07]), array([0.57798596, 2.04724757, 3.00671655, 0.45223832]), array([0.73038393, 1.46084082, 2.10856375, 0.55159487]), array([1.31210078, 0.13591777, 0.1398901 , 0.25834028]), array([0.76249123, 0.97218551, 1.43517086, 0.60055928]), array([1.25220082, 0.04390307, 0.03069319, 0.17192547]), array([1.93473602e+05, 1.42166285e+00, 1.77369490e+02, 1.55004636e-01]), array([inf, inf, inf, inf]), array([7.64041790e+05, 8.51819762e+01, 1.09554449e+03, 2.17050033e-01]), array([5.29714957e+06, 4.28089560e+02, 3.48201874e+03, 1.34688714e-01]), array([1.20608716, 0.33930383, 0.34739507, 0.21380701]), array([0.81680047, 2.52001567, 4.7118101 , 0.87809848]), array([5.37103054e+06, 5.31518340e+00, 8.20775069e+05, 5.37103072e+06]), array([           nan, 9.64055603e+07,            nan, 1.31067571e-01]), array([       nan, 9.80981929,        nan,        nan]), array([1.00390634, 0.08677078, 0.08911868, 0.19016206]), array([1.38180726, 0.04213154, 0.03928834, 0.14638537]), array([1.82132525e+07, 1.74154004e+02, 4.71331323e+06, 1.82132502e+07]), array([0.71823646, 0.64500641, 0.72493403, 0.28846134]), array([1.16002296, 0.10945043, 0.11176216, 0.19280659]), array([2.41771385e+06, 5.39715324e+01, 9.27865769e+05, 2.41771360e+06]), array([1.57796503e+06, 5.46467678e+02, 1.31293202e+03, 2.16545746e-01]), array([1.01225206e+07, 1.55836455e+00, 7.24088906e+05, 1.01225208e+07]), array([0.91908181, 4.24327016, 9.72011394, 0.972102  ]), array([15.2334726 ,  0.07586955,  0.24843072,  0.16045782]), array([0.55588073, 0.88725938, 1.1064919 , 0.310948  ]), array([4.90120960e+05, 1.09732973e+02, 4.05869006e+02, 1.97248710e-01]), array([1.07855999, 0.21960803, 0.22704224, 0.2286115 ]), array([4.79352063e+06, 1.12937348e+01, 1.43498925e+06, 4.79352080e+06]), array([2.51011986e+05, 7.32814741e+04, 1.23751633e+04, 1.78753596e-01]), array([0.8064663 , 0.21247381, 0.22307335, 0.2071365 ]), array([8.24558424e+04, 4.88640197e+01, 1.68722590e+02, 1.74334267e-01]), array([inf, inf, inf, inf]), array([1.51356289, 0.13996307, 0.11644358, 0.21484746]), array([1.67701758, 0.04874269, 0.0553431 , 0.1758467 ]), array([0.93961411, 0.2746407 , 0.29059836, 0.26003425]), array([1.64298591, 0.10444592, 0.07755142, 0.1875073 ]), array([           nan, 8.00275424e+05,            nan, 1.61820282e-01]), array([3.54352123e+06, 2.22571472e+03, 6.08878430e+03, 2.55067655e-01]), array([8.87166095e+07, 7.32758894e+07, 1.99557917e+09, 1.84021688e-01]), array([0.99053716, 0.08097941, 0.0830575 , 0.18300364]), array([1.42195757e+06, 4.27672492e+01, 8.44218876e+05, 1.42195764e+06]), array([1.01659510e+06, 2.00668748e+01, 6.71240867e+02, 1.40936788e-01]), array([0.89080833, 0.43865835, 0.48353346, 0.3258395 ]), array([0.61745627, 0.54159651, 0.60233892, 0.23622602]), array([0.75072163, 1.09145426, 1.24906245, 0.32475301]), array([2.07321769, 0.04111517, 0.05472961, 0.16164849]), array([1.20699793, 3.06544739, 7.53570586, 1.31319604]), array([1.36175450e+06, 5.49498249e+01, 1.77329722e+03, 1.78436841e-01]), array([0.93275955, 1.03139181, 2.38171667, 0.97648223]), array([0.71181552, 0.56105318, 0.63409222, 0.29283033]), array([0.6313413 , 1.32098053, 2.5034316 , 0.61247446]), array([1.63103296, 0.08326196, 0.09020247, 0.23165769]), array([1.29147686, 0.05822714, 0.05021207, 0.16986945]), array([0.58170229, 1.18692706, 1.72402039, 0.43834853]), array([2.11893157, 0.23495719, 0.25293427, 0.31720359]), array([0.77454239, 0.77396762, 1.4921727 , 0.75865033]), array([1.25059475, 0.11304164, 0.11759556, 0.18996649]), array([2.06425785e+06, 3.34293649e+02, 6.64810616e+02, 1.81744750e-01]), array([ 2.93844885,  4.79308883, 19.52924125,  3.09077572]), array([1.64908496, 0.05283425, 0.0702398 , 0.19175628]), array([1.64011591, 0.16213786, 0.14763269, 0.23553197])]
median x_centre of target 23124:  0.1683197484945605
standard deviation from x0=1640.42A of target 23124:  0.9457848502856226
THE LYA SNR FOR TARGET 23124 IS:  26.23804
THE INTEGRATED FLUX OF LYA FOR TARGET 23124 IS:  73.98082
###############################################################################################################
THE He-II SNR FOR TARGET 48 is:  2.8355436
INTEGRATED FLUX OF He-II of 48 is:  2.485129
THE He-II FWHM FOR TARGET 48 IS:  0.8385319054345404
amp_48 = 2.789382024006839 +/- None
cen_48 = 1640.6212551285116 +/- None
sigma_48 = 0.3560917137613034 +/- None
continuum_48 = 2.0085459 +/- None
THE He-II EW FOR TARGET 48 IS:  0.31642727164559226
std-errors gaussian of target 48 (amp, cen, sigma, continuum):  [array([1.61952615, 0.13576772, 0.13928953, 0.30097902]), array([1.72285397, 0.10078994, 0.09933   , 0.28529216]), array([1.08735316, 0.28353866, 0.29846299, 0.28662763]), array([2.31081769e+07, 2.15265505e+00, 1.07980143e+06, 2.31081774e+07]), array([2.25652268, 0.24387921, 0.11387171, 0.27629844]), array([0.85563355, 1.27126012, 1.81475145, 0.63711073]), array([ 5.45937239,  3.60076316, 20.15270907,  5.66864681]), array([inf, inf, inf, inf]), array([1.99191336e+07, 5.88439461e+00, 3.17169619e+06, 1.99191337e+07]), array([2.55048108e+07, 1.56236557e+00, 6.45511870e+05, 2.55048107e+07]), array([0.79503889, 0.99528967, 1.54705503, 0.64427171]), array([0.6975457 , 1.26803264, 1.66724623, 0.4497575 ]), array([1.23588678, 2.09727769, 4.6133611 , 1.27828027]), array([0.84362146, 1.08080107, 1.66719192, 0.68128218]), array([0.96504618, 0.46681315, 0.51529648, 0.35563697]), array([0.75564362, 0.88839177, 1.05047937, 0.33443352]), array([0.87613126, 2.5029789 , 4.30001798, 0.84914592]), array([7.05494855e+06, 1.45111935e+03, 9.19473272e+03, 2.94125526e-01]), array([1.15582331, 0.423751  , 0.45260115, 0.34603636]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([1.18656173, 0.97288237, 1.03464509, 0.35100449]), array([1.51378931, 0.16137355, 0.16629817, 0.30462182]), array([1202687.51149541,   60157.26736784, 3195928.77136749,
       1201719.59507465]), array([1.93084165e+07, 2.50039862e+00, 7.18344051e+05, 1.93084166e+07]), array([2.16304008, 0.09727295, 0.09164852, 0.28987566]), array([inf, inf, inf, inf]), array([1.63761784, 0.07248573, 0.05136589, 0.24624274]), array([inf, inf, inf, inf]), array([3.13991952, 1.92854202, 7.55931995, 3.34969898]), array([1.21715004, 0.18589306, 0.19121973, 0.23632734]), array([2.2502339 , 0.37212856, 0.36918891, 0.35137638]), array([2.31312708, 6.62142297, 3.90707083, 0.27349937]), array([0.76743806, 1.21056277, 1.93401386, 0.66193415]), array([inf, inf, inf, inf]), array([101.05451799,   3.41918152, 115.24337216, 101.28746841]), array([3.62189202e+06, 6.91174910e-01, 1.32172332e+05, 3.62189231e+06]), array([0.78482641, 0.45391076, 0.51822346, 0.33640913]), array([1.32868817, 0.63269361, 0.66404678, 0.34012886]), array([11.44669974,  1.45934538, 19.33239748, 11.74256873]), array([7.96684112e+05, 1.85409756e+06, 2.96413104e+05, 2.92437225e-01]), array([inf, inf, inf, inf]), array([0.90028361, 1.27238635, 2.14850004, 0.81534207]), array([2.23254067e+07, 2.28295722e+00, 8.66101572e+05, 2.23254066e+07]), array([inf, inf, inf, inf]), array([1.79579979e+07, 6.44662424e+00, 1.28262741e+06, 1.79579981e+07]), array([1.61687167, 0.14306142, 0.1475268 , 0.26839708]), array([7.19984638e+06, 3.20799147e+00, 6.82476878e+05, 7.19984663e+06]), array([       nan, 2.45401883,        nan,        nan]), array([inf, inf, inf, inf]), array([1.68463593, 0.04748134, 0.05279929, 0.2545061 ]), array([2.67066133, 2.42048041, 7.85411741, 2.87628214]), array([1.75058262, 1.59657149, 5.16254088, 1.88328844]), array([1.23551647, 0.17177441, 0.17744235, 0.25832473]), array([0.92653415, 2.91022084, 5.28680536, 0.88119327]), array([       nan, 1.37319991,        nan,        nan]), array([1.73591464, 0.13270183, 0.13610277, 0.317537  ]), array([2.92943369e+05, 6.26521644e+01, 3.60229848e+02, 3.11113401e-01]), array([2.52635820e+07, 5.81459283e-01, 3.23639994e+05, 2.52635820e+07]), array([0.84857128, 0.36263479, 0.40660215, 0.33716819]), array([0.65991562, 0.92748109, 1.40961225, 0.54037002]), array([1.69971522, 0.11016128, 0.11326862, 0.32810258]), array([1.18861497, 1.58981361, 3.40906696, 1.23160498]), array([ 3.66677672,  5.05453648, 13.21441606,  3.75763449]), array([       nan, 8.84838229,        nan,        nan]), array([2.47558386e+07, 6.40234252e+00, 9.38428055e+05, 2.47558387e+07]), array([1.78682239, 0.11179217, 0.07117002, 0.27293484]), array([1.8239011 , 0.1334406 , 0.1368323 , 0.33156234]), array([0.78572379, 1.33088867, 1.80786064, 0.52665431]), array([1.08210582, 1.44549321, 2.70162953, 1.04518112]), array([inf, inf, inf, inf]), array([5.99175263e+05, 3.52648225e+02, 7.90808274e+02, 2.35267757e-01]), array([1.08369549, 0.41948996, 0.44606602, 0.31314865]), array([1.12980445, 0.7804939 , 0.87908363, 0.45801224]), array([1.06271812, 0.65671832, 0.7098631 , 0.34643318]), array([1.61127459e+07, 6.29662624e-01, 3.52368684e+05, 1.61127463e+07]), array([inf, inf, inf, inf]), array([1.54280726, 0.11500383, 0.11047744, 0.24262663]), array([0.90627174, 1.99106119, 2.43275042, 0.43606994]), array([inf, inf, inf, inf]), array([0.73453505, 0.70279949, 0.82977103, 0.34654254]), array([1.23818745, 0.12393349, 0.12820939, 0.26490705]), array([inf, inf, inf, inf]), array([0.73473307, 0.81834005, 1.07183568, 0.47906151]), array([0.84617283, 0.50135187, 0.56363575, 0.3402607 ]), array([1.09070388, 0.47917979, 0.54026536, 0.27619718]), array([3.44528338, 1.57688663, 8.75939179, 3.62621263]), array([2.47940641, 2.13793126, 7.24297629, 2.66458003]), array([0.7961672 , 0.72525767, 0.86405858, 0.39450792]), array([       nan, 2.12440187,        nan,        nan]), array([0.84251415, 0.39823994, 0.4415545 , 0.3175923 ]), array([1.71873336, 0.10558109, 0.1082847 , 0.29952562]), array([0.97279656, 0.73550776, 0.86327498, 0.46513867]), array([0.65209909, 0.692021  , 1.2462737 , 0.61681382]), array([4.31730362e+09, 5.29743480e+05, 5.78963030e+04, 1.90417258e+02]), array([2.0615708 , 0.06282899, 0.08781257, 0.25414893]), array([2162.15985259,    3.56165589,  762.63833092, 2162.47218377]), array([2.36910672, 0.11558553, 0.10799863, 0.37116842]), array([5.79898746e+06, 3.16317229e+00, 5.04187723e+05, 5.79898781e+06]), array([1.92345945, 0.15666956, 0.16068252, 0.34992324])]
median x_centre of target 48:  -0.014638361419916621
standard deviation from x0=1640.42A of target 48:  0.9108792642824967
The Lya SNR FOR TARGET 48 is:  4.0075665
INTEGRATED FLUX OF Lya FOR TARGET 48 is:  22.789223
###############################################################################################################
THE He-II SNR FOR TARGET 118 is:  5.1530375
INTEGRATED FLUX OF He-II of 118 is:  3.600877238997782
THE He-II FWHM FOR TARGET 118 IS:  0.717531391018344
amp_118 = 4.705163463556583 +/- 3.125208629318119
cen_118 = 1637.0512347630706 +/- 0.2661362952993908
sigma_118 = 0.3047075263914332 +/- 0.17946467694890925
continuum_118 = 3.970727 +/- 0
THE He-II EW FOR TARGET 118 IS:  0.22657332783424397
std-errors gaussian of target 118 (amp, cen, sigma, continuum):  [array([1.21644596, 0.07974867, 0.07660474, 0.19538315]), array([0.93420138, 0.05372637, 0.05513055, 0.16920427]), array([3.29102772, 0.05755921, 0.0606018 , 0.14941631]), array([0.87512508, 0.08295896, 0.08533117, 0.16997674]), array([1.80689531e+08,            nan, 5.29204291e+10, 2.40574429e-01]), array([0.62725434, 0.46927083, 0.50740075, 0.20495055]), array([1.11865634, 0.04609588, 0.04638164, 0.18253619]), array([1.51602749, 3.2360736 , 6.47274719, 1.61035008]), array([1.03644939, 0.07218989, 0.07397455, 0.18923864]), array([0.96187984, 0.07258307, 0.0743992 , 0.176139  ]), array([0.95351841, 0.03817235, 0.03748333, 0.15300763]), array([0.52634564, 0.83062034, 1.063319  , 0.32679972]), array([1.59687683, 0.68541787, 2.8453311 , 1.71954443]), array([5.74084887e+06, 1.09009495e+00, 3.85437419e+05, 5.74084903e+06]), array([7.25834895e+05, 3.03423891e+02, 1.31778876e+03, 2.11867824e-01]), array([inf, inf, inf, inf]), array([0.79819185, 1.6862402 , 3.43025412, 0.80317628]), array([7.67676912e+06, 1.57033578e+00, 5.67173649e+05, 7.67676933e+06]), array([1.3214056 , 0.09396599, 0.10131848, 0.201373  ]), array([0.99715045, 0.05346355, 0.05490817, 0.17252751]), array([0.59447575, 0.99656541, 1.5610114 , 0.50782696]), array([inf, inf, inf, inf]), array([1.07472697, 0.0802029 , 0.08185327, 0.18347029]), array([0.91074018, 0.08003023, 0.0821779 , 0.17258663]), array([0.80347376, 0.1340951 , 0.13940342, 0.18384784]), array([inf, inf, inf, inf]), array([1.30814735, 0.09501745, 0.09628367, 0.22517864]), array([inf, inf, inf, inf]), array([1.1052374 , 0.0545765 , 0.06302397, 0.15611118]), array([0.9543601 , 0.10586225, 0.10934083, 0.19886432]), array([1.64363001, 0.03748303, 0.05340307, 0.18875191]), array([0.79928948, 1.53143241, 3.58018087, 0.84157418]), array([2.02510307e+07, 1.85278588e+01, 2.12171690e+06, 2.02510308e+07]), array([1.31525901e+05, 4.86179798e+01, 1.62358776e+02, 1.46720617e-01]), array([0.53730914, 0.41382587, 0.51392955, 0.30964858]), array([0.99920162, 0.07675099, 0.07878658, 0.18629127]), array([1.14306413, 0.13392237, 0.13797136, 0.19474196]), array([1.23144468, 0.07178374, 0.06194107, 0.18810432]), array([1.19677939, 0.06561183, 0.07501629, 0.17285675]), array([0.64628932, 0.97675479, 1.29267775, 0.41530702]), array([0.88525293, 0.05772316, 0.05889569, 0.15028127]), array([1.17798419, 0.06411285, 0.06464871, 0.19650767]), array([inf, inf, inf, inf]), array([1.15141778, 0.05354878, 0.05441862, 0.17144414]), array([0.65573364, 0.28670982, 0.30488713, 0.18952509]), array([0.47986736, 0.29035486, 0.32484134, 0.18976493]), array([1.11147336, 0.09338748, 0.09575973, 0.20662406]), array([2.40904456, 0.0849618 , 0.07011221, 0.19237769]), array([0.51024905, 0.90506152, 1.13099996, 0.29602468]), array([1.06363672, 0.0772319 , 0.07845485, 0.1759426 ]), array([1.17591075e+07, 1.74151242e+00, 5.64062021e+05, 1.17591076e+07]), array([1.14141013, 0.08095551, 0.08279344, 0.20048626]), array([0.78917722, 0.06649339, 0.06798279, 0.14200981]), array([1.07513742, 0.07481929, 0.0767876 , 0.19994553]), array([1.43859754, 0.09415389, 0.09746064, 0.22376337]), array([0.88842673, 0.05194677, 0.0525829 , 0.1523798 ]), array([0.97657933, 0.0708712 , 0.0722457 , 0.16661892]), array([141.86016269,   0.66520321,  87.8451912 , 142.0635789 ]), array([1.00872842, 0.06042648, 0.06192457, 0.17869661]), array([1.34813169, 0.07855437, 0.08113562, 0.22870926]), array([1.14814174, 0.15080796, 0.15530038, 0.22822993]), array([0.78999871, 0.09879376, 0.10203694, 0.16457782]), array([1.01159479, 0.04134191, 0.04345082, 0.15205115]), array([0.85663987, 0.05751012, 0.05840826, 0.14672511]), array([0.4571787 , 0.61730881, 0.78298017, 0.27915641]), array([1.46236881, 0.05515722, 0.05686206, 0.18999005]), array([1.12294214, 0.09727285, 0.09866758, 0.18609455]), array([1.26145892, 0.0437804 , 0.04705716, 0.19432035]), array([inf, inf, inf, inf]), array([1.04756036, 0.04168954, 0.03843214, 0.1628416 ]), array([2.6876021 , 0.0688175 , 0.08678849, 0.18905881]), array([0.97809082, 0.06442287, 0.07144831, 0.13964584]), array([1.04858982, 0.09335982, 0.09553241, 0.18903574]), array([1.2054991 , 0.10891631, 0.11117232, 0.20796282]), array([1.19576796, 0.07400249, 0.07648234, 0.1905998 ]), array([0.6983655 , 0.64480611, 0.71571672, 0.26460471]), array([1.134193  , 0.05236002, 0.05152753, 0.17562836]), array([0.79808787, 0.10042603, 0.10412816, 0.17646705]), array([1.27842647, 0.07475164, 0.07386649, 0.20611342]), array([0.95485793, 0.04039657, 0.04317089, 0.1342993 ]), array([1.05612544e+12,            nan, 1.57258381e+10,            nan]), array([0.87256404, 0.13316114, 0.13809536, 0.19330426]), array([1.45978723, 0.04429438, 0.06197703, 0.17667667]), array([0.61158878, 0.27607999, 0.3127233 , 0.25496096]), array([4.63343542e+06, 4.48628738e+00, 5.17527367e+05, 4.63343562e+06]), array([1.57394012e+07, 2.80062795e+00, 1.59037974e+06, 1.57394014e+07]), array([0.55817354, 0.56571913, 0.77102249, 0.39018132]), array([           nan,            nan, 1.30180648e+08, 0.00000000e+00]), array([1.23698391, 0.0382699 , 0.02723944, 0.16665342]), array([16.91076689, 49.47794165, 18.0628122 ,  0.24362583]), array([7.32284051e+06, 2.06550369e+00, 7.35545791e+05, 7.32284082e+06]), array([1.95994326, 0.05297862, 0.04270712, 0.16129813]), array([0.91246772, 0.15057621, 0.1570422 , 0.21736899]), array([0.62584901, 1.46843505, 2.24110118, 0.52963463]), array([1.37007185, 0.06353892, 0.06466448, 0.23192214]), array([0.54240254, 0.63103806, 0.7863241 , 0.30201556]), array([1.60117947e+07, 2.12460923e+00, 9.67696187e+05, 1.60117946e+07]), array([1.28086081, 0.06371342, 0.06534024, 0.2163671 ]), array([0.62642715, 1.10202215, 1.40607415, 0.38746202]), array([7.16546947e+05, 8.00849603e+02, 2.07167600e+03, 2.44184905e-01])]
median x_centre of target 118:  0.050983903683901696
standard deviation from x0=1640.42A of target 118:  0.9877238391216795
The Lya SNR FOR TARGET 118 is:  75.49209
INTEGRATED FLUX OF Lya FOR TARGET 118 is:  227.20530507602962
###############################################################################################################
THE He-II SNR FOR TARGET 131 is:  4.7307024
INTEGRATED FLUX OF He-II of 131 is:  1.7662939
THE He-II FWHM FOR TARGET 131 IS:  0.656492256381671
amp_131 = 2.500621647685824 +/- 3.2283469328142576
cen_131 = 1639.2876907106706 +/- 0.35259451290033184
sigma_131 = 0.2787865925326123 +/- 0.5110641104959103
continuum_131 = 2.8594465 +/- 0
THE He-II EW FOR TARGET 131 IS:  0.15368851392343744
std-errors gaussian of target 131 (amp, cen, sigma, continuum):  [array([1.11757603, 0.12660972, 0.12952131, 0.20183876]), array([1.24873638e+07, 3.14815562e+00, 9.21530994e+05, 1.24873638e+07]), array([1.31054859, 0.28934739, 0.29935474, 0.28084057]), array([0.53602911, 1.01829652, 1.22132238, 0.2500532 ]), array([inf, inf, inf, inf]), array([0.59013592, 0.47219279, 0.52875453, 0.23312349]), array([0.9273272 , 0.66778694, 0.71234857, 0.27505409]), array([0.79525585, 0.57045121, 0.60717931, 0.23157745]), array([0.75374355, 0.32866419, 0.35449474, 0.24229317]), array([nan, nan, nan, nan]), array([0.62906983, 0.31514099, 0.3381205 , 0.19480733]), array([0.98606097, 0.30156811, 0.3152735 , 0.24168918]), array([1.62984258, 0.11600713, 0.09597118, 0.18575045]), array([1.19077786, 0.16638434, 0.17133454, 0.2366153 ]), array([3.05773194, 0.12414374, 0.12207725, 0.157427  ]), array([0.90607013, 0.28717146, 0.30152318, 0.23297339]), array([4.37902643e+05, 5.11016232e+00, 8.92888256e+04, 4.37902808e+05]), array([1.60476326e+07, 9.05349621e+02, 5.33968526e+06, 1.60476119e+07]), array([       nan, 1.12790358,        nan,        nan]), array([0.56034831, 2.07903788, 3.11222465, 0.45303958]), array([0.68405425, 3.98927504, 7.65518964, 0.70462981]), array([8.08654166e+06, 1.28801359e+01, 1.39778332e+06, 8.08654188e+06]), array([2.28733346, 0.14168215, 0.25401894, 0.20930255]), array([0.69262958, 0.72464937, 0.82156827, 0.28870571]), array([0.43536037, 1.13720936, 1.70737432, 0.35003423]), array([0.56350713, 1.12848705, 1.4453944 , 0.34779384]), array([       nan, 0.90782063,        nan,        nan]), array([1.08820032e+07, 5.27739908e+01, 2.46455217e+06, 1.08820031e+07]), array([1.69408249e+07, 1.70531553e+00, 9.63842471e+05, 1.69408251e+07]), array([inf, inf, inf, inf]), array([7.18756422e+04, 4.84361163e+02, 5.99426603e+02, 2.07786058e-01]), array([1.26930016, 0.16960486, 0.17383951, 0.23236577]), array([1.52178614e+07, 3.30061264e+01, 4.04871165e+06, 1.52178614e+07]), array([0.58456471, 0.61175867, 0.81237715, 0.3734995 ]), array([0.52381891, 1.09939437, 1.49864233, 0.36641867]), array([1.00551199, 0.2009635 , 0.20905057, 0.23223636]), array([1.53993980e+07, 7.28032116e-01, 3.50797189e+05, 1.53993982e+07]), array([0.72947085, 0.37816884, 0.4001272 , 0.20198879]), array([inf, inf, inf, inf]), array([0.71610736, 0.73899617, 0.83097432, 0.28824198]), array([0.95079879, 0.46964565, 0.48805585, 0.21663557]), array([ 2.31944949,  7.08556135, 26.47679366,  2.4615411 ]), array([1.81628009, 0.09757735, 0.07262481, 0.19841869]), array([0.60657337, 0.70115769, 0.78523365, 0.23975918]), array([       nan, 3.68154535,        nan,        nan]), array([0.94241203, 0.22168798, 0.23059124, 0.21721368]), array([0.81819498, 0.34862515, 0.36806533, 0.22195989]), array([4.75251383e+06, 1.79358060e+00, 4.70595746e+05, 4.75251405e+06]), array([inf, inf, inf, inf]), array([1.42166084e+07, 2.03436960e+01, 9.03541822e+05, 1.42166082e+07]), array([1.02043008, 0.34894215, 0.36886533, 0.28002773]), array([4.69474859, 0.14412157, 0.17387574, 0.194303  ]), array([0.90851295, 0.58269226, 0.6098788 , 0.22554265]), array([0.78917462, 0.71419689, 0.7635477 , 0.23807134]), array([0.68215894, 0.48758611, 0.52311036, 0.21115083]), array([1.01112807, 0.28598301, 0.29693879, 0.22766172]), array([1.99101183e+08, 5.99588285e+06, 6.11237297e+06, 2.19696735e-01]), array([6.81172909e+05, 5.26148897e+02, 2.11587352e+03, 2.03261062e-01]), array([1.03626027e+07, 3.09335797e+00, 4.57587021e+05, 1.03626027e+07]), array([1.41408867e+07, 1.35195862e+00, 7.28051787e+05, 1.41408867e+07]), array([1.99382435e+05, 1.36431691e+02, 2.84927239e+02, 2.53279470e-01]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([1.70393602e+07, 3.70864951e+00, 2.34850774e+06, 1.70393603e+07]), array([0.62573035, 1.20969145, 1.75918326, 0.48152989]), array([6.36813016, 0.99303771, 9.40497083, 6.54992704]), array([0.67183477, 2.71662702, 4.44835335, 0.69032154]), array([1.20414987, 0.20770885, 0.21501615, 0.26019369]), array([0.77237388, 0.24981058, 0.2670416 , 0.2326842 ]), array([0.76910796, 1.39683033, 2.49992755, 0.72144465]), array([5.07412689e+06, 2.83781127e+05, 4.43620029e+04, 1.85173506e+02]), array([       nan, 4.38376345,        nan,        nan]), array([0.54819629, 0.52521975, 0.61690992, 0.25991967]), array([0.90373729, 1.44888358, 1.56572474, 0.29422172]), array([0.83001191, 3.06516333, 3.49389372, 0.3304414 ]), array([inf, inf, inf, inf]), array([1.361841  , 7.82717733, 5.04218896, 0.322398  ]), array([ 6.17250021,  5.91428459, 24.4356682 ,  6.26296242]), array([0.69522375, 0.38912234, 0.43508131, 0.27358335]), array([4.35842128e+03, 2.94203732e+00, 1.94775892e+03, 4.35863030e+03]), array([0.91150961, 0.54582166, 0.56705868, 0.20682368]), array([1.37434149, 0.10016681, 0.09528523, 0.21136523]), array([inf, inf, inf, inf]), array([1.10733787, 0.08177363, 0.071604  , 0.17277279]), array([0.96252641, 1.89443435, 4.8696086 , 1.04276286]), array([1.22682396, 0.1009295 , 0.06513434, 0.17882581]), array([ 0.94503498, 13.29074412,  9.70786675,  0.37621981]), array([0.63376569, 1.03667656, 1.70290889, 0.54821164]), array([0.77443444, 0.493769  , 0.53072473, 0.24294825]), array([       nan, 0.70578342,        nan,        nan]), array([0.55396413, 0.57465664, 0.65736923, 0.23926491]), array([1.60023973e+07, 1.05752571e+00, 6.27652107e+05, 1.60023973e+07]), array([0.57085073, 0.74284389, 1.06901733, 0.43202147]), array([1.49955442, 0.13616973, 0.14128301, 0.24088435]), array([inf, inf, inf, inf]), array([ 8.27779794,  1.6600914 , 18.49341885,  8.47064306]), array([1.37876605, 0.12100439, 0.11634232, 0.21987872]), array([nan, nan, nan, nan]), array([0.46825491, 1.67283417, 2.16831493, 0.29943197]), array([0.71006383, 0.6574292 , 0.72250508, 0.25552598])]
median x_centre of target 131:  -0.04085461435199528
standard deviation from x0=1640.42A of target 131:  0.9487191617438323
The Lya SNR FOR TARGET 131 is:  13.54681
INTEGRATED FLUX OF Lya FOR TARGET 131 is:  48.527027

###############################################################################################################
THE He-II SNR FOR TARGET 7876 is:  4.490123
INTEGRATED FLUX OF He-II of 7876 is:  1.2950246
THE He-II FWHM FOR TARGET 7876 IS:  0.7779385963467925
amp_7876 = 1.5669625630239827 +/- None
cen_7876 = 1640.1024849072728 +/- None
sigma_7876 = 0.3303601045813962 +/- None
continuum_7876 = 2.4743485 +/- None
THE He-II EW FOR TARGET 7876 IS:  0.13107061457925492
std-errors gaussian of target 7876 (amp, cen, sigma, continuum):  [array([       nan,        nan,        nan, 0.19943402]), array([0.89996433, 0.41420627, 0.42831203, 0.19093012]), array([ 4.40664248, 16.36950878, 30.51136204,  4.21860197]), array([0.85576308, 0.26558941, 0.27501774, 0.18536937]), array([0.53918474, 0.66678189, 0.83092499, 0.30120216]), array([0.91248144, 0.40043061, 0.42430325, 0.25593019]), array([0.70603061, 0.48079698, 0.52290242, 0.2393255 ]), array([0.58045591, 0.91444148, 1.03295784, 0.24575203]), array([0.87680523, 0.12154566, 0.12528521, 0.17743187]), array([       nan, 1.43480412,        nan,        nan]), array([0.76598898, 1.20680667, 1.38783548, 0.33751818]), array([1.35000356, 3.88586277, 8.64138909, 1.4691824 ]), array([inf, inf, inf, inf]), array([9.43631362e+05, 1.06268864e+02, 2.42809900e+03, 1.80944075e-01]), array([1.28324550e+07, 1.74687720e+01, 1.66557643e+06, 1.28324550e+07]), array([3.35290314, 1.17992616, 8.93371779, 3.5122876 ]), array([2.35509271e+05, 1.97617192e+02, 5.43619263e+02, 1.78009073e-01]), array([0.78166359, 0.28997325, 0.30116125, 0.17685063]), array([           nan, 7.89641212e+06, 1.61643140e+06, 2.00691308e-01]), array([0.52387894, 0.69669188, 1.14566466, 0.45293389]), array([0.92735655, 0.20828067, 0.21710805, 0.21969124]), array([inf, inf, inf, inf]), array([1.16734256e+07, 3.54966107e+00, 7.25284526e+05, 1.16734256e+07]), array([0.44577011, 1.50536696, 2.00454229, 0.30073315]), array([inf, inf, inf, inf]), array([0.96657107, 0.229033  , 0.23685408, 0.20549003]), array([4.95229334e+06, 3.22126312e+01, 1.02234250e+06, 4.95229312e+06]), array([inf, inf, inf, inf]), array([0.53069754, 1.64877315, 1.94567176, 0.2661895 ]), array([0.77884962, 0.16902503, 0.17499341, 0.16851952]), array([       nan, 2.57792486,        nan,        nan]), array([0.73528465, 0.45305454, 0.49080851, 0.24319308]), array([       nan, 0.45028693,        nan,        nan]), array([0.87796465, 0.24804831, 0.25517928, 0.17114401]), array([2.68468881e+05, 7.67631216e+02, 1.17929915e+03, 2.27261066e-01]), array([1.31778403e+07, 1.87763545e+00, 7.03608850e+05, 1.31778403e+07]), array([1.23689664, 0.18277433, 0.18726454, 0.22586533]), array([2.21564680e+06, 1.73005291e+03, 2.76544009e+03, 1.63125738e-01]), array([9.17336365e+06, 1.75085983e+00, 7.84916869e+05, 9.17336383e+06]), array([0.691227  , 0.72155244, 0.78744563, 0.23915799]), array([0.52956469, 1.60893384, 2.07609835, 0.32423637]), array([3.49834498e+06, 2.20985756e+02, 4.64574073e+03, 2.07562529e-01]), array([0.99575797, 0.2085514 , 0.21400495, 0.18246311]), array([0.65807538, 2.30266128, 3.88121969, 0.58300532]), array([0.6204607 , 0.29322638, 0.31724119, 0.20350951]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([inf, inf, inf, inf]), array([0.41964794, 1.38388794, 1.81083449, 0.26078076]), array([0.61047149, 1.67294965, 2.30588325, 0.42432427]), array([8.00703312e+05, 8.71826788e+01, 2.78259514e+05, 8.00701612e+05]), array([2.73144014e+05, 7.04558455e+01, 5.18710445e+02, 1.46183029e-01]), array([0.64370487, 0.4428166 , 0.48934265, 0.23852869]), array([0.99616808, 0.1176039 , 0.12000373, 0.16957591]), array([1.58854617e+07, 1.05729399e+00, 4.67611179e+05, 1.58854618e+07]), array([0.67995612, 0.4506537 , 0.49788316, 0.25538629]), array([0.74630586, 0.5671073 , 0.62421916, 0.2709128 ]), array([0.92065045, 0.47971021, 0.49753706, 0.20429376]), array([0.81409758, 0.22802965, 0.2375581 , 0.19144026]), array([0.69436613, 1.34975833, 2.13101532, 0.58409444]), array([       nan, 8.98119886,        nan,        nan]), array([3.90998677e+06, 2.82787832e+02, 3.03515235e+03, 1.61836689e-01]), array([1.62698900e+05, 2.43585412e+02, 4.88684511e+02, 1.95320860e-01]), array([0.51596385, 1.02699621, 1.44274835, 0.36751645]), array([1.57459538e+07, 1.03851647e+00, 4.63460387e+05, 1.57459538e+07]), array([2.43113150e+07, 3.29660135e+01, 3.20717258e+07, 2.43113150e+07]), array([0.51413971, 1.96611117, 2.96146119, 0.40809299]), array([0.98686639, 0.12044094, 0.12343469, 0.16661928]), array([4.93865586e+06, 4.04270354e+02, 1.40981981e+06, 4.93864361e+06]), array([0.92397857, 0.20861053, 0.21848298, 0.23121567]), array([0.58057362, 0.64054403, 0.78198651, 0.3102868 ]), array([0.58752821, 0.38200002, 0.42044133, 0.21316523]), array([0.99278639, 0.23363903, 0.24092794, 0.20219379]), array([1.43227494, 0.12575718, 0.11231508, 0.22238637]), array([1.13615291, 0.14171544, 0.14495683, 0.20555674]), array([0.58387385, 0.56152515, 0.61989994, 0.21527992]), array([4.83376466e+05, 9.01496887e+02, 2.07602425e+03, 2.02723811e-01]), array([0.60897079, 0.49323694, 0.5688869 , 0.27121868]), array([1.37067201e+07, 1.78499390e+00, 8.23903847e+05, 1.37067201e+07]), array([1.50014833, 0.18583585, 0.18178071, 0.2224147 ]), array([1.30972044e+07, 1.98339712e+00, 1.41305897e+06, 1.30972047e+07]), array([0.52409969, 0.7973306 , 0.97650081, 0.26436639]), array([0.56856705, 0.38159473, 0.4726396 , 0.31342525]), array([1.41803288, 0.17554959, 0.18206818, 0.23710211]), array([0.64700869, 0.36935605, 0.40529843, 0.23092917]), array([0.92067392, 0.46117244, 0.49414626, 0.28245843]), array([0.58881829, 0.26285185, 0.28058994, 0.17552548]), array([           nan, 0.00000000e+00, 7.35082838e+05, 2.39801015e-01]), array([0.66374372, 2.72231262, 4.62036569, 0.62541594]), array([1.06416974, 0.26854308, 0.27628623, 0.20824887]), array([inf, inf, inf, inf]), array([5.45044204e+05, 1.22528648e+02, 7.50104644e+02, 1.66373640e-01]), array([0.79495134, 0.567196  , 0.60403864, 0.23809057]), array([       nan,        nan,        nan, 0.21233831]), array([inf, inf, inf, inf]), array([           nan, 1.03217155e+06,            nan, 2.43986251e-01]), array([1.13760535, 0.11148752, 0.11308701, 0.19539934]), array([1.57498572e+07, 1.80664559e+00, 8.84634072e+05, 1.57498574e+07]), array([8.94609377e+05, 7.21090722e+01, 8.73659673e+02, 2.24585146e-01]), array([0.84314301, 0.20217139, 0.21214815, 0.21541019])]
median x_centre of target 7876:  0.16482364673835304
standard deviation from x0=1640.42A of target 7876:  0.905897850893634
###############################################################################################################
THE He-II SNR FOR TARGET 218 is:  4.0444508
INTEGRATED FLUX OF He-II of 218 is:  1.3963616
THE He-II FWHM FOR TARGET 218 IS:  0.6442456731765814
amp_218 = 2.00000497894976 +/- None
cen_218 = 1639.42 +/- None
sigma_218 = 0.27358594748505044 +/- None
continuum_218 = 2.3715484 +/- None
THE He-II EW FOR TARGET 218 IS:  0.15231611027459752
std-errors gaussian of target 218 (amp, cen, sigma, continuum):  [array([7.46587309e+06, 1.10620039e+00, 3.04732346e+05, 7.46587312e+06]), array([1.64389113e+07, 1.09139991e+00, 3.51583602e+05, 1.64389115e+07]), array([1.31036970e+07, 8.29661002e-01, 3.61783109e+05, 1.31036973e+07]), array([1.49592056e+06, 4.96167140e+02, 4.75854645e+03, 2.22890138e-01]), array([3.12460459e+05, 1.29543360e+02, 5.71803616e+02, 1.86213859e-01]), array([1.09994284e+07, 1.33686505e+00, 4.17412934e+05, 1.09994284e+07]), array([1.60105402e+07, 1.72838693e+00, 6.68564047e+05, 1.60105403e+07]), array([2.09487834e+05, 5.18042934e+05, 8.79834110e+04, 1.73425661e-01]), array([0.5575917 , 0.70719017, 0.90884279, 0.33460897]), array([0.76265525, 1.64810631, 2.01335685, 0.42475107]), array([0.92495973, 0.4453731 , 0.46985926, 0.24920201]), array([1.64608592, 0.21994137, 0.2144587 , 0.20748569]), array([ 1.21971121, 19.83717993, 18.0334426 ,  0.81368751]), array([     nan, 1.149328,      nan,      nan]), array([0.95428337, 0.25747952, 0.26734865, 0.21500788]), array([1.75180451e+07, 3.16189516e+00, 7.52924161e+05, 1.75180450e+07]), array([1.58869840e+06, 2.32066912e+03, 7.53534711e+03, 2.40945509e-01]), array([9.55634429e+02, 6.50466266e-01, 3.13604844e+02, 9.55830341e+02]), array([ 5.47306338, 49.78280831, 36.67908109,  2.25546293]), array([0.55093575, 0.59648196, 0.70996529, 0.27211552]), array([0.56102454, 1.48737625, 2.58258931, 0.51611059]), array([5.69322947e+07, 5.40447578e+05, 7.27610927e+04, 1.98795545e+02]), array([9.39690303e+10, 6.55608322e+04, 5.80883588e+03, 7.19395544e+00]), array([0.73104627, 1.40028017, 1.55644201, 0.27891265]), array([4.36134199e+06, 1.80815138e+01, 1.15839548e+06, 4.36134211e+06]), array([1.32055531, 0.13183627, 0.13892752, 0.2183681 ]), array([1.48980466, 0.14505727, 0.11861889, 0.23617644]), array([2.28393750e+04, 8.81508790e+04, 1.52205173e+04, 2.17348722e-01]), array([ 4.16316638,  8.97999278, 23.31287496,  4.22583295]), array([0.85684891, 2.91945265, 5.56485918, 0.90708275]), array([1.1812092 , 9.76627124, 7.78499494, 0.63457503]), array([1.60510957e+07, 2.27880388e+00, 9.35154907e+05, 1.60510958e+07]), array([1.13454891e+07, 1.12686730e+01, 2.31862243e+06, 1.13454892e+07]), array([1.32416076e+07, 4.65155091e+01, 2.43423662e+06, 1.32416075e+07]), array([0.77561031, 0.2585359 , 0.27607056, 0.23180439]), array([1.48049700e+07, 1.51239570e+00, 5.55318257e+05, 1.48049701e+07]), array([0.65086283, 0.74011764, 0.89130334, 0.33314847]), array([1.13254871, 0.40196412, 0.4155098 , 0.23938097]), array([2.13355869e+07, 5.75607656e+00, 1.44103327e+06, 2.13355872e+07]), array([0.59486783, 5.61512394, 7.81149182, 0.5263782 ]), array([1.65126831e+07, 1.46861055e+00, 5.23697578e+05, 1.65126831e+07]), array([inf, inf, inf, inf]), array([1.27308084, 0.39669078, 0.41650555, 0.32708646]), array([3.31343628e+06, 1.75911131e+00, 3.72429041e+05, 3.31343653e+06]), array([0.52009977, 0.64828067, 0.77010631, 0.25542071]), array([1.09876697, 0.25104889, 0.26129185, 0.2553577 ]), array([1.82972027, 0.10559642, 0.17081037, 0.19898705]), array([1.28330063, 0.10870398, 0.09462115, 0.20091152]), array([0.58697716, 1.15387664, 1.27507081, 0.1829641 ]), array([1.19993313, 0.10589969, 0.08479402, 0.18517561]), array([18.4140931 ,  1.49792156, 23.87347997, 18.62541823]), array([7.31710649e+06, 1.09499708e+00, 3.03672221e+05, 7.31710668e+06]), array([0.79430511, 1.587823  , 3.02800221, 0.77263667]), array([inf, inf, inf, inf]), array([0.61523274, 0.4216627 , 0.47552033, 0.25076231]), array([0.59656306, 0.60454789, 0.80007721, 0.3869648 ]), array([1.16500971, 0.14504731, 0.15652208, 0.18587058]), array([8.71636254e+06, 2.04057135e+00, 4.19100835e+05, 8.71636260e+06]), array([2.33889751e+06, 9.42188343e+00, 2.90027136e+05, 2.33889756e+06]), array([inf, inf, inf, inf]), array([8.32636677e+06, 1.74764493e+00, 3.43653141e+05, 8.32636673e+06]), array([1.40887661, 0.16515494, 0.09326193, 0.19573624]), array([1.08304103, 0.08022892, 0.05447683, 0.16989684]), array([9.60901703e+05, 1.94191942e+02, 2.14958706e+03, 2.47105498e-01]), array([0.91234409, 0.32861309, 0.34596215, 0.24092128]), array([inf, inf, inf, inf]), array([0.56846277, 1.30549209, 1.6849693 , 0.35696632]), array([inf, inf, inf, inf]), array([1.05230115, 0.12143525, 0.12772418, 0.17476101]), array([2.2718042 , 2.36171711, 9.46646398, 2.41932452]), array([0.60607626, 1.62622091, 1.88999505, 0.23468702]), array([1.72400996e+07, 1.35162246e+00, 4.86865271e+05, 1.72400999e+07]), array([      nan, 7.7681651,       nan,       nan]), array([0.54594564, 1.03947072, 1.3507113 , 0.33398178]), array([1.73944512e+07, 2.22374695e+00, 9.10932071e+05, 1.73944515e+07]), array([inf, inf, inf, inf]), array([0.59615959, 1.13068475, 1.29848322, 0.23505514]), array([0.6610908 , 0.69221059, 0.74619192, 0.21165295]), array([0.7638904 , 1.81921585, 3.24691313, 0.74695904]), array([1.34356021e+06, 1.18085861e+02, 2.08562390e+03, 1.75686468e-01]), array([0.82302454, 0.09547429, 0.09769271, 0.1515079 ]), array([0.44936009, 0.6233126 , 0.73356293, 0.21433559]), array([1.10181994, 0.377055  , 0.39801784, 0.2984957 ]), array([9.25587070e+06, 8.55500826e-01, 4.05696617e+05, 9.25587091e+06]), array([0.52704834, 1.36522658, 1.7984442 , 0.34592949]), array([0.96582808, 0.39845388, 0.41633589, 0.23532275]), array([1.81536913e+07, 3.88934606e+00, 1.68055279e+06, 1.81536915e+07]), array([inf, inf, inf, inf]), array([0.8884    , 0.49926186, 0.52663982, 0.23904588]), array([1.46239139, 0.12798604, 0.08542065, 0.21789688]), array([inf, inf, inf, inf]), array([3.19324827, 0.10267375, 0.13837711, 0.24486012]), array([0.59168287, 0.52957507, 0.62192498, 0.29129368]), array([1.3052059 , 0.1499876 , 0.1259275 , 0.20413444]), array([0.64945054, 0.76303309, 0.91456402, 0.31777553]), array([0.49939331, 0.96635431, 1.22868397, 0.29612004]), array([ 1.97560659,  4.48485245, 12.70891395,  2.10926237]), array([       nan, 7.53245183,        nan,        nan]), array([7.81943884e+06, 2.11459608e+00, 6.68056766e+05, 7.81943907e+06]), array([0.89674917, 0.29745336, 0.31005589, 0.21234251])]
median x_centre of target 218:  -0.13086217413591553
standard deviation from x0=1640.42A of target 218:  0.9483289635274119
###############################################################################################################
finished part 3
0:00:14.875000

Process finished with exit code 0

###############################################################################################################
'''
