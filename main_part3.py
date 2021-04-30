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
import pandas as pd
import seaborn as sns
import time
from datetime import timedelta
import os

#niceValue = os.nice(19)	#  nice: value in [-20 important, 19 unimportant]
start_time = time.monotonic()
##########################################################################################################################
########## P A R T   3  : ANALYSE THE PEAKS OF TARGETS 88, 204, 435, 22429 ###############################################
            # Peaks: He-II             C-III] and C-IV in PART5
            # Analysis includes: Gaussian fit with parameters and their stddevs, SNR of He-II peak, bootstrap experiment
            # SNR and integrated flux of Lya peak
# https://lmfit.github.io/lmfit-py/
NAME = 'EXTRACTION'
EXTRACTIONS_Lya = 'Lya_selected_ALL_%s.fits'%(NAME)
extractions_Lya = pyfits.open(EXTRACTIONS_Lya)
DATA_Lya = extractions_Lya[1].data
EXTRACTIONS_HeII = 'HeII_selected_ALL_%s.fits'%(NAME)
extractions_HeII = pyfits.open(EXTRACTIONS_HeII)
DATA_HeII = extractions_HeII[1].data

# possible targets:
iden_str = ['88', '204', '435', '22429'] # string version
iden_int = [88, 204, 435, 22429] # integer version
REDSHIFT = [2.9541607, 3.1357558, 3.7247474, 2.9297342]
Ly_alpha_rest = 1215.67
HeII = 1640.42
z_min = 2.86
z_max = 4.7
N = 10 # 1000

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
rest_wavelen_HeII = [rest_wavelen_HeII_88, rest_wavelen_HeII_204, rest_wavelen_HeII_435, rest_wavelen_HeII_22429]
flux_HeII = [flux_HeII_88, flux_HeII_204, flux_HeII_435, flux_HeII_22429]
noise_HeII = [noise_HeII_88, noise_HeII_204, noise_HeII_435, noise_HeII_22429]

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
rest_wavelen_Lya = [rest_wavelen_Lya_88, rest_wavelen_Lya_204, rest_wavelen_Lya_435, rest_wavelen_Lya_22429]
flux_Lya = [flux_Lya_88, flux_Lya_204, flux_Lya_435, flux_Lya_22429]
noise_Lya = [noise_Lya_88, noise_Lya_204, noise_Lya_435, noise_Lya_22429]


def Gauss(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))

def objective88(params, x, data):
    #amp = params['amp'].value
    #cen = params['cen'].value
    #sigma = params['sigma'].value
    #model = Gauss(x, amp, cen, sigma)
    amp = params['amp_%s' % (iden_str[0])].value
    cen = params['cen_%s' % (iden_str[0])].value
    sigma = params['sigma_%s' % (iden_str[0])].value
    model = Gauss(x, amp, cen, sigma)
    return model - data

def objective204(params, x, data):
    amp = params['amp_%s' % (iden_str[1])].value
    cen = params['cen_%s' % (iden_str[1])].value
    sigma = params['sigma_%s' % (iden_str[1])].value
    model = Gauss(x, amp, cen, sigma)
    return model - data

def objective435(params, x, data):
    amp = params['amp_%s' % (iden_str[2])].value
    cen = params['cen_%s' % (iden_str[2])].value
    sigma = params['sigma_%s' % (iden_str[2])].value
    model = Gauss(x, amp, cen, sigma)
    return model - data

def objective22429(params, x, data):
    amp = params['amp_%s' % (iden_str[3])].value
    cen = params['cen_%s' % (iden_str[3])].value
    sigma = params['sigma_%s' % (iden_str[3])].value
    model = Gauss(x, amp, cen, sigma)
    return model - data

Gauss_model = Model(Gauss, nan_policy='propagate')


# TARGET 88: wavelengths 1638.3:1640.1
##################################################################################################################
FWHM_HeII_88 = 1.
Delta_wavelen_HeII_88 = 2  # from plots
Delta_v_HeII_88 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_88) / HeII  # in km/s
x = rest_wavelen_HeII_88
data = flux_HeII_88

'''
class GaussianModel(lmfit.Model):
    def __init__(self, *args, **kwargs):
        def Gauss(x, amp, cen, sigma):
            return amp * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))
        super(GaussianModel, self).__init__(Gauss, *args, **kwargs)

    def guess(self, data, **kwargs):
        params = self.make_params()
        def pset(param, value):
            params["%s%s" % (self.prefix, param)].set(value=value)
        pset('amp', np.max(data) - np.min(data)) #('amp_%s' % (iden_str[0]), np.max(data) - np.min(data))
        pset('sigma', 3) #('sigma_%s' % (iden_str[0]), 3)
        pset('cen', 1640.42) #('cen_%s' % (iden_str[0]), 1640.42)
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)
model = GaussianModel()
params_He88 = model.guess(data, x=x)
'''

# HE -II ANALYSIS
'''
params_He88 = Parameters()
params_He88.add('amp_%s' % (iden_str[0]), value=max(flux_HeII_88), min=15)
params_He88.add('cen_%s' % (iden_str[0]), value=1639.5, min=HeII - Delta_wavelen_HeII_88, max=HeII)
params_He88.add('sigma_%s' % (iden_str[0]), value=3, min=0.01, max=FWHM_HeII_88 / (2 * np.sqrt(2 * np.log(2))))
minner88 = Minimizer(objective88, params_He88, fcn_args=(x, data))
#print(dir(minner88)) # associated variables
result_HeII88 = minner88.minimize()
final_HeII_88 = flux_HeII[0] + result_HeII88.residual
n0_final_HeII_88 = final_HeII_88[final_HeII_88 != 0] # no zeros
n0_x = x[13:29]
n0_noise_HeII_88 = noise_HeII_88[13:29]
integrate_88 = simps(n0_final_HeII_88, n0_x) # = 21.259823639760725
print('integrated flux of He-II of 88 is: ', integrate_88)

index_max_flux_88 = np.argmax(flux_HeII_88)
SNR_HeII_88 = max(flux_HeII_88) / noise_HeII_88[index_max_flux_88]
print('The He-II SNR for target 88 is: ', SNR_HeII_88) # = 2.5999868

for key in result_HeII88.params:
    # prints the parameters amp_88, cen_88 and sigma_88 with standard deviations
    print(key, "=", result_HeII88.params[key].value, "+/-", result_HeII88.params[key].stderr)

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

# BOOTSTRAP EXPERIMENT:
x_centre_88 = []
integrate_new_88 = []
#sigma_new_88 = []
#FWHM_new_88 =[]
perr_gauss_88 = []
for n in range(N):
    y_new_88 = np.random.normal(n0_final_HeII_88, n0_noise_HeII_88)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_88, p0=[max(y_new_88), HeII, 3])
    perr_gauss_88.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_88.append(simps(y_new_88, n0_x))
    #sigma_new_88.append(1)
    #FWHM_new_88.append(2 * np.sqrt(2 * np.log(2)) * sigma_new_88)
    this_x_fit_88 = np.random.normal(0, 1.)
    x_centre_88.append(this_x_fit_88)
median_x_centre_88 = np.nanmedian(x_centre_88)
std_x_centre_88 = np.nanstd(x_centre_88)
print('std-errors gaussian of target 88 (amp, cen, sigma): ', perr_gauss_88)
print('median x_centre of target 88: ', median_x_centre_88)
print('standard deviation from x0=1640.42A of target 88: ', std_x_centre_88)

# LYA ANALYSIS:
index_max_Lyaflux_88 = np.argmax(flux_Lya_88)
SNR_Lya_88 = max(flux_Lya_88) / noise_Lya_88[index_max_Lyaflux_88]
print('The Lya SNR for target 88 is: ', SNR_Lya_88) # =4.717403
FWHM_Lya_88 = 2.
Delta_wavelen_Lya_88 = 1  # from plots
Delta_v_Lya_88 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_88) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_88
dataLya = flux_Lya_88

params_Lya88 = Parameters()
params_Lya88.add('amp_%s' % (iden_str[0]), value=max(flux_Lya_88), min=15)
params_Lya88.add('cen_%s' % (iden_str[0]), value=1215)#, min=Ly_alpha_rest - Delta_wavelen_Lya_88, max=Ly_alpha_rest)
params_Lya88.add('sigma_%s' % (iden_str[0]), value=3)#, min=0.01, max=FWHM_Lya_88 / (2 * np.sqrt(2 * np.log(2))))
minner88 = Minimizer(objective88, params_Lya88, fcn_args=(xLya, dataLya))
#print(dir(minner88)) # associated variables
result_Lya88 = minner88.minimize()
final_Lya_88 = flux_Lya[0] + result_Lya88.residual
n0_final_Lya_88 = final_Lya_88[final_Lya_88 != 0] # no zeros
n0_x = xLya[7:40]
n0_noise_Lya_88 = noise_Lya_88[7:40]
integrate_Lya_88 = simps(n0_final_Lya_88, n0_x) # = 54.429844
print('integrated flux of Lya of 88 is: ', integrate_Lya_88)
'''



# TARGET 204:  wavelengths 1638:1641.5
##################################################################################################################
FWHM_HeII_204 = 0.6
Delta_wavelen_HeII_204 = 1  # from plots
Delta_v_HeII_204 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_204) / HeII  # in km/s
ind_max_flux_204 = np.argmax(flux_HeII_204)
reduced_wavelen_HeII_204 = np.arange(1638., 1641.5, 0.01)
y_reduced_204 = Gauss(reduced_wavelen_HeII_204, 1, np.argmax(flux_HeII_204), 1)
integrate_flux_204 = simps(y_reduced_204, reduced_wavelen_HeII_204, 0.01)
'''
params_He204 = Parameters()
params_He204.add('amp_%s' % (iden_str[1]), value=max(flux_HeII_204), min=2.5, max=3)
params_He204.add('cen_%s' % (iden_str[1]), value=rest_wavelen_HeII_204[ind_max_flux_204], min=HeII - Delta_wavelen_HeII_204, max=HeII)
params_He204.add('sigma_%s' % (iden_str[1]), value=1, min=0.01, max=FWHM_HeII_204 / (2 * np.sqrt(2 * np.log(2))))
x = rest_wavelen_HeII_204
data = flux_HeII_204
minner204 = Minimizer(objective204, params_He204, fcn_args=(x, data))
result_HeII204 = minner204.minimize()
final_HeII_204 = flux_HeII[1] + result_HeII204.residual
n0_final_HeII_204 = final_HeII_204[final_HeII_204 != 0] # no zeros
n0_x = x[17:27]
n0_noise_HeII_204 = noise_HeII_204[17:27]
integrate_204 = simps(n0_final_HeII_204, n0_x) # = 1.753795421625341
print('integrated flux of He-II of 204 is: ', integrate_204)

index_max_flux_204 = np.argmax(flux_HeII_204)
SNR_HeII_204 = max(flux_HeII_204) / noise_HeII_204[index_max_flux_204]
print('The He-II SNR for target 204 is: ', SNR_HeII_204) # ~3.3

for key in result_HeII204.params:
    # prints the parameters amp_204, cen_204 and sigma_204 with standard deviations
    print(key, "=", result_HeII204.params[key].value, "+/-", result_HeII204.params[key].stderr)

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

# BOOTSTRAP EXPERIMENT:
x_centre_204 = []
integrate_new_204 = []
perr_gauss_204 = []
for n in range(N):
    y_new_204 = np.random.normal(n0_final_HeII_204, n0_noise_HeII_204)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_204, p0=[max(y_new_204), HeII, 3])
    perr_gauss_204.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_204.append(simps(y_new_204, n0_x))
    this_x_fit_204 = np.random.normal(0, 1.)
    x_centre_204.append(this_x_fit_204)
median_x_centre_204 = np.nanmedian(x_centre_204)
std_x_centre_204 = np.nanstd(x_centre_204)
print('std-errors gaussian of target 204 (amp, cen, sigma): ', perr_gauss_204)
print('median x_centre of target 204: ', median_x_centre_204)
print('standard deviation from x0=1640.42A of target 204: ', std_x_centre_204)
'''
'''
# LYA ANALYSIS:
index_max_Lyaflux_204 = np.argmax(flux_Lya_204)
SNR_Lya_204 = max(flux_Lya_204) / noise_Lya_204[index_max_Lyaflux_204]
print('The Lya SNR for target 204 is: ', SNR_Lya_204) # = 16.23129
FWHM_Lya_204 = 2.
Delta_wavelen_Lya_204 = 1  # from plots
Delta_v_Lya_204 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_204) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_204
dataLya = flux_Lya_204

params_Lya204 = Parameters()
params_Lya204.add('amp_%s' % (iden_str[1]), value=max(flux_Lya_204), min=15)
params_Lya204.add('cen_%s' % (iden_str[1]), value=1215)#, min=Ly_alpha_rest - Delta_wavelen_Lya_88, max=Ly_alpha_rest)
params_Lya204.add('sigma_%s' % (iden_str[1]), value=3)#, min=0.01, max=FWHM_Lya_88 / (2 * np.sqrt(2 * np.log(2))))
minner204 = Minimizer(objective204, params_Lya204, fcn_args=(xLya, dataLya))
#print(dir(minner88)) # associated variables
result_Lya204 = minner204.minimize()
final_Lya_204 = flux_Lya[1] + result_Lya204.residual
n0_final_Lya_204 = final_Lya_204[final_Lya_204 != 0] # no zeros
n0_x = xLya[3:47]
n0_noise_Lya_204 = noise_Lya_204[3:47]
integrate_Lya_204 = simps(n0_final_Lya_204, n0_x) # = 41.66130142482871
print('integrated flux of Lya of 204 is: ', integrate_Lya_204)
'''


'''
# TARGET 435:
##################################################################################################################
FWHM_HeII_435 = 1.5
Delta_wavelen_HeII_435 = 2.8  # from plots
Delta_v_HeII_435 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_435) / HeII  # in km/s

params_He435 = Parameters()
params_He435.add('amp_%s' % (iden_str[2]), value=flux_HeII_435[17])
params_He435.add('cen_%s' % (iden_str[2]), value=rest_wavelen_HeII_435[17])
params_He435.add('sigma_%s' % (iden_str[2]), value=1, min=0.01, max=FWHM_HeII_435 / (2 * np.sqrt(2 * np.log(2))))
x = rest_wavelen_HeII_435
data = flux_HeII_435
minner435 = Minimizer(objective435, params_He435, fcn_args=(x, data))
result_HeII435 = minner435.minimize()
final_HeII_435 = flux_HeII[2] + result_HeII435.residual # np.ndarray
n0_final_HeII_435 = final_HeII_435[final_HeII_435 != 0] # no zeros
n0_x = x[3:32]
n0_noise_HeII_435 = noise_HeII_435[3:32]
ModelResult435 = ModelResult(Gauss_model, params_He435, weights=True, nan_policy='propagate')
integrate_435 = simps(n0_final_HeII_435, n0_x) # = 7.9270864
print('integrated flux of He-II of 435 is: ', integrate_435)

index_max_flux_435 = 17 #np.argmax(flux_HeII_435)
SNR_HeII_435 = max(flux_HeII_435) / noise_HeII_435[index_max_flux_435]
print('The He-II SNR for target 435 is: ', SNR_HeII_435) # ~4.9

for key in result_HeII435.params:
    # prints the parameters amp_435, cen_435 and sigma_435 with standard deviations
    print(key, "=", result_HeII435.params[key].value, "+/-", result_HeII435.params[key].stderr)

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
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_435 = []
integrate_new_435 = []
perr_gauss_435 = []
for n in range(N):
    y_new_435 = np.random.normal(n0_final_HeII_435, n0_noise_HeII_435)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_435, p0=[max(y_new_435), HeII, 3])
    perr_gauss_435.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_435.append(simps(y_new_435, n0_x))
    this_x_fit_435 = np.random.normal(0, 1.)
    x_centre_435.append(this_x_fit_435)
median_x_centre_435 = np.nanmedian(x_centre_435)
std_x_centre_435 = np.nanstd(x_centre_435)
print('std-errors gaussian of target 435 (amp, cen, sigma): ', perr_gauss_435)
print('median x_centre of target 435: ', median_x_centre_435)
print('standard deviation from x0=1640.42A of target 435: ', std_x_centre_435)
'''
# LYA ANALYSIS:
'''
index_max_Lyaflux_435 = np.argmax(flux_Lya_435)
SNR_Lya_435 = max(flux_Lya_435) / noise_Lya_435[index_max_Lyaflux_435]
print('The Lya SNR for target 435 is: ', SNR_Lya_435) # = 84.764755
FWHM_Lya_435 = 2.
Delta_wavelen_Lya_435 = 2  # from plots
Delta_v_Lya_435 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_435) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_435
dataLya = flux_Lya_435

params_Lya435 = Parameters()
params_Lya435.add('amp_%s' % (iden_str[2]), value=max(flux_Lya_435), min=15)
params_Lya435.add('cen_%s' % (iden_str[2]), value=rest_wavelen_Lya_435[index_max_Lyaflux_435])#, min=Ly_alpha_rest - Delta_wavelen_Lya_88, max=Ly_alpha_rest)
params_Lya435.add('sigma_%s' % (iden_str[2]), value=3)#, min=0.01, max=FWHM_Lya_88 / (2 * np.sqrt(2 * np.log(2))))
minner435 = Minimizer(objective435, params_Lya435, fcn_args=(xLya, dataLya))
#print(dir(minner88)) # associated variables
result_Lya435 = minner435.minimize()
final_Lya_435 = flux_Lya[2] + result_Lya435.residual
n0_final_Lya_435 = final_Lya_435[final_Lya_435 != 0] # no zeros
n0_x = xLya[16:35]
n0_noise_Lya_435 = noise_Lya_435[16:35]
integrate_Lya_435 = simps(n0_final_Lya_435, n0_x) # = 125.522385
print('integrated flux of Lya of 435 is: ', integrate_Lya_435)
'''

# TARGET 22429:
##################################################################################################################
'''
FWHM_HeII_22429 = 1.5
Delta_wavelen_HeII_22429 = 2.8  # from plots
Delta_v_HeII_22429 = (const.speed_of_light / 1000) * (-Delta_wavelen_HeII_22429) / HeII  # in km/s

params_He22429 = Parameters()
params_He22429.add('amp_%s' % (iden_str[3]), value=flux_HeII_22429[12])
params_He22429.add('cen_%s' % (iden_str[3]), value=rest_wavelen_HeII_22429[12])
params_He22429.add('sigma_%s' % (iden_str[3]), value=1, min=0.01, max=FWHM_HeII_22429 / (2 * np.sqrt(2 * np.log(2))))
x = rest_wavelen_HeII_22429
data = flux_HeII_22429
minner22429 = Minimizer(objective22429, params_He22429, fcn_args=(x, data))
result_HeII22429 = minner22429.minimize()
final_HeII_22429 = flux_HeII[3] + result_HeII22429.residual # np.ndarray
n0_final_HeII_22429 = final_HeII_22429[final_HeII_22429 != 0] # no zeros
n0_x = x[0:23]
n0_noise_HeII_22429 = noise_HeII_22429[0:23]
ModelResult22429 = ModelResult(Gauss_model, params_He22429, weights=True, nan_policy='propagate')
integrate_22429 = simps(n0_final_HeII_22429, n0_x) # =
print('integrated flux of He-II of 22429 is: ', integrate_22429)

index_max_flux_22429 = 12 #np.argmax(flux_HeII_22429)
SNR_HeII_22429 = max(flux_HeII_22429) / noise_HeII_22429[index_max_flux_22429]
print('The He-II SNR for target 22429 is: ', SNR_HeII_22429) # = 2.6729734

for key in result_HeII22429.params:
    # prints the parameters amp_22429, cen_22429 and sigma_22429 with standard deviations
    print(key, "=", result_HeII22429.params[key].value, "+/-", result_HeII22429.params[key].stderr)

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
plt.clf()

# BOOTSTRAP EXPERIMENT:
x_centre_22429 = []
integrate_new_22429 = []
perr_gauss_22429 = []
for n in range(N):
    y_new_22429 = np.random.normal(n0_final_HeII_22429, n0_noise_HeII_22429)
    popt_gauss, pcov_gauss = curve_fit(Gauss, n0_x, y_new_22429, p0=[max(y_new_22429), HeII, 3])
    perr_gauss_22429.append(np.sqrt(np.diag(pcov_gauss)))
    integrate_new_22429.append(simps(y_new_22429, n0_x))
    this_x_fit_22429 = np.random.normal(0, 1.)
    x_centre_22429.append(this_x_fit_22429)
median_x_centre_22429 = np.nanmedian(x_centre_22429)
std_x_centre_22429 = np.nanstd(x_centre_22429)
print('std-errors gaussian of target 22429 (amp, cen, sigma): ', perr_gauss_22429)
print('median x_centre of target 22429: ', median_x_centre_22429)
print('standard deviation from x0=1640.42A of target 22429: ', std_x_centre_22429)
'''

# LYA ANALYSIS:
index_max_Lyaflux_22429 = np.argmax(flux_Lya_22429)
SNR_Lya_22429 = max(flux_Lya_22429) / noise_Lya_22429[index_max_Lyaflux_22429]
print('The Lya SNR for target 22429 is: ', SNR_Lya_22429) # = 13.586418

FWHM_Lya_22429 = 2.
Delta_wavelen_Lya_22429 = 2  # from plots
Delta_v_Lya_22429 = (const.speed_of_light / 1000) * (-Delta_wavelen_Lya_22429) / Ly_alpha_rest  # in km/s
xLya = rest_wavelen_Lya_22429
dataLya = flux_Lya_22429

params_Lya22429 = Parameters()
params_Lya22429.add('amp_%s' % (iden_str[3]), value=max(flux_Lya_22429), min=15)
params_Lya22429.add('cen_%s' % (iden_str[3]), value=rest_wavelen_Lya_22429[index_max_Lyaflux_22429])#, min=Ly_alpha_rest - Delta_wavelen_Lya_88, max=Ly_alpha_rest)
params_Lya22429.add('sigma_%s' % (iden_str[3]), value=3)#, min=0.01, max=FWHM_Lya_88 / (2 * np.sqrt(2 * np.log(2))))
minner22429 = Minimizer(objective22429, params_Lya22429, fcn_args=(xLya, dataLya))
#print(dir(minner88)) # associated variables
result_Lya22429 = minner22429.minimize()
final_Lya_22429 = flux_Lya[3] + result_Lya22429.residual
n0_final_Lya_22429 = final_Lya_22429[final_Lya_22429 != 0] # no zeros
n0_x = xLya[14:36]
n0_noise_Lya_22429 = noise_Lya_22429[14:36]
integrate_Lya_22429 = simps(n0_final_Lya_22429, n0_x) # = 36.138626431686134
print('integrated flux of Lya of 22429 is: ', integrate_Lya_22429)


# END
########################
print('finished part 3')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
