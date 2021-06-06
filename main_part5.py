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
#######################################################################################################################
# P A R T   5  : CREATE C-IV, C-III], OIII] and SiIII] PEAKs OF TARGETS [...] AND DETERMINE THE SNR  ###############
    # 48 88 118 131 204 435 538 5199 22429 23124
    # C-III] peak       :  arXiv:1902.05960v1 [astro-ph.GA]; Nanayakkara
    # C-IV peak         :  arXiv:1911.09999 [astro-ph.GA]; Saxena
# NOTE: IGNORE 54891 AND 53: HAVE NO HE-II 1640 PEAK

NAME = 'EXTRACTION'
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
IDEN = np.array([88, 204, 435, 22429, 5199, 48, 131, 118, 538, 23124, 53, 218, 54891, 7876])
iden_str = ['88', '204', '435', '22429', '5199', '48', '131', '118', '538', '23124', '53', '218', '54891', '7876']
z = [2.9541607, 3.1357558, 3.7247474, 2.9297342, 3.063, 2.9101489, 3.0191996, 3.0024831, 4.1764603,
     3.59353921008408, 2.9148405, 2.865628, 2.9374344, 2.993115]

C_IV = 1548.19
CIII_sf = 1908.73
O_III_sf = 1666.15
SiIII_sf = 1882.71
rest_vac_wavelen = [[] for j in range(len(IDEN))]
flux_dens_tot = [[] for j in range(len(IDEN))]
flux_dens_err = [[] for j in range(len(IDEN))]

C_IV_indices = []
rest_wavelen_C_IV = [[] for j in range(len(IDEN))]
flux_C_IV = [[] for j in range(len(IDEN))]
noise_C_IV = [[] for j in range(len(IDEN))]
C_IV_SNR = []

CIII_sf_indices = []
rest_wavelen_CIII_sf = [[] for j in range(len(IDEN))]
flux_CIII_sf = [[] for j in range(len(IDEN))]
noise_CIII_sf = [[] for j in range(len(IDEN))]
CIII_sf_SNR = []

O_III_sf_indices = []
rest_wavelen_O_III_sf = [[] for j in range(len(IDEN))]
flux_O_III_sf = [[] for j in range(len(IDEN))]
noise_O_III_sf = [[] for j in range(len(IDEN))]
O_III_sf_SNR = []

SiIII_sf_indices = []
rest_wavelen_SiIII_sf = [[] for j in range(len(IDEN))]
flux_SiIII_sf = [[] for j in range(len(IDEN))]
noise_SiIII_sf = [[] for j in range(len(IDEN))]
SiIII_sf_SNR = []

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
def objective5199(params, x, data):
    amp = params['amp_%s' % (iden_str[4])].value
    cen = params['cen_%s' % (iden_str[4])].value
    sigma = params['sigma_%s' % (iden_str[4])].value
    continuum = params['continuum_%s' % (iden_str[4])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective48(params, x, data):
    amp = params['amp_%s' % (iden_str[5])].value
    cen = params['cen_%s' % (iden_str[5])].value
    sigma = params['sigma_%s' % (iden_str[5])].value
    continuum = params['continuum_%s' % (iden_str[5])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective131(params, x, data):
    amp = params['amp_%s' % (iden_str[6])].value
    cen = params['cen_%s' % (iden_str[6])].value
    sigma = params['sigma_%s' % (iden_str[6])].value
    continuum = params['continuum_%s' % (iden_str[6])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective118(params, x, data):
    amp = params['amp_%s' % (iden_str[7])].value
    cen = params['cen_%s' % (iden_str[7])].value
    sigma = params['sigma_%s' % (iden_str[7])].value
    continuum = params['continuum_%s' % (iden_str[7])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective538(params, x, data):
    amp = params['amp_%s' % (iden_str[8])].value
    cen = params['cen_%s' % (iden_str[8])].value
    sigma = params['sigma_%s' % (iden_str[8])].value
    continuum = params['continuum_%s' % (iden_str[8])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective23124(params, x, data):
    amp = params['amp_%s' % (iden_str[9])].value
    cen = params['cen_%s' % (iden_str[9])].value
    sigma = params['sigma_%s' % (iden_str[9])].value
    continuum = params['continuum_%s' % (iden_str[9])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective53(params, x, data):
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
def objective54891(params, x, data):
    amp = params['amp_%s' % (iden_str[12])].value
    cen = params['cen_%s' % (iden_str[12])].value
    sigma = params['sigma_%s' % (iden_str[12])].value
    continuum = params['continuum_%s' % (iden_str[12])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
def objective7876(params, x, data):
    amp = params['amp_%s' % (iden_str[13])].value
    cen = params['cen_%s' % (iden_str[13])].value
    sigma = params['sigma_%s' % (iden_str[13])].value
    continuum = params['continuum_%s' % (iden_str[13])].value
    model = Gauss(x, amp, cen, sigma, continuum)
    return model - data
Gauss_model = Model(Gauss, nan_policy='propagate')

figNr = 0
for i in range(len(IDEN)):
    target_nr = str(IDEN[i])
    flux_dens_tot[i] = DATA.field('fluxdensity_total_%s' % (target_nr))
    flux_dens_err[i] = DATA.field('fluxdensity_total_ERR_%s' % (target_nr))
    rest_vac_wavelen[i] = DATA.field('rest_vac_wavelen_%s' % (target_nr))

    C_IV_indices.append(np.where((rest_vac_wavelen[i] > 1541) & (rest_vac_wavelen[i] < 1555)))
    flux_C_IV[i] = np.array(flux_dens_tot[i])[C_IV_indices[i]]
    noise_C_IV[i] = np.array(flux_dens_err[i])[C_IV_indices[i]]
    rest_wavelen_C_IV[i] = np.array(rest_vac_wavelen[i])[C_IV_indices[i]]

    CIII_sf_indices.append(np.where((rest_vac_wavelen[i] > 1902) & (rest_vac_wavelen[i] < 1916)))
    flux_CIII_sf[i] = np.array(flux_dens_tot[i])[CIII_sf_indices[i]]
    noise_CIII_sf[i] = np.array(flux_dens_err[i])[CIII_sf_indices[i]]
    rest_wavelen_CIII_sf[i] = np.array(rest_vac_wavelen[i])[CIII_sf_indices[i]]

    O_III_sf_indices.append(np.where((rest_vac_wavelen[i] > 1659) & (rest_vac_wavelen[i] < 1673)))
    flux_O_III_sf[i] = np.array(flux_dens_tot[i])[O_III_sf_indices[i]]
    noise_O_III_sf[i] = np.array(flux_dens_err[i])[O_III_sf_indices[i]]
    rest_wavelen_O_III_sf[i] = np.array(rest_vac_wavelen[i])[O_III_sf_indices[i]]

    SiIII_sf_indices.append(np.where((rest_vac_wavelen[i] > 1876) & (rest_vac_wavelen[i] < 1890)))
    flux_SiIII_sf[i] = np.array(flux_dens_tot[i])[SiIII_sf_indices[i]]
    noise_SiIII_sf[i] = np.array(flux_dens_err[i])[SiIII_sf_indices[i]]
    rest_wavelen_SiIII_sf[i] = np.array(rest_vac_wavelen[i])[SiIII_sf_indices[i]]

# for target 118, need 1896-1910 A
CIII_sf_indices[7] = np.where((rest_vac_wavelen[7] > 1896) & (rest_vac_wavelen[7] < 1910))
flux_CIII_sf[7] = np.array(flux_dens_tot[7])[CIII_sf_indices[7]]
noise_CIII_sf[7] = np.array(flux_dens_err[7])[CIII_sf_indices[7]]
rest_wavelen_CIII_sf[7] = np.array(rest_vac_wavelen[7])[CIII_sf_indices[7]]

# SIGNAL TO NOISE RATIO:

index_max_flux_C_IV = [16, np.argmax(flux_C_IV[1]), 20, np.argmax(flux_C_IV[3]), np.argmax(flux_C_IV[4]), 13,
                       np.argmax(flux_C_IV[6]), np.argmax(flux_C_IV[7]), np.argmax(flux_C_IV[8]),
                       np.argmax(flux_C_IV[9]), 21, 8, 0, 0]
# Peak doesnt exist for 88, 22429, 5199, 48, 131, 538, 53, (218), 54891, 7876  : append a "0"
C_IV_SNR.append(0)
C_IV_SNR.append(flux_C_IV[1][index_max_flux_C_IV[1]] / noise_C_IV[1][index_max_flux_C_IV[1]])
C_IV_SNR.append(flux_C_IV[2][index_max_flux_C_IV[2]] / noise_C_IV[2][index_max_flux_C_IV[2]])
C_IV_SNR.append(0)
C_IV_SNR.append(0)
C_IV_SNR.append(0)
C_IV_SNR.append(0)
C_IV_SNR.append(flux_C_IV[7][index_max_flux_C_IV[7]] / noise_C_IV[7][index_max_flux_C_IV[7]])
C_IV_SNR.append(0)
C_IV_SNR.append(flux_C_IV[9][index_max_flux_C_IV[9]] / noise_C_IV[9][index_max_flux_C_IV[9]])
C_IV_SNR.append(0)
C_IV_SNR.append(flux_C_IV[11][index_max_flux_C_IV[11]] / noise_C_IV[11][index_max_flux_C_IV[11]])
C_IV_SNR.append(0)
C_IV_SNR.append(0)
print('The C-IV SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, '
      '23124, 53, 218, 54891, 7876 are: ', C_IV_SNR)

index_max_flux_CIII_sf = [13, 10, np.argmax(flux_CIII_sf[2]), np.argmax(flux_CIII_sf[3]), np.argmax(flux_CIII_sf[4]), 2,
                          np.argmax(flux_CIII_sf[6]), np.argmax(flux_CIII_sf[7]), 0, 10, np.argmax(flux_CIII_sf[10]),
                          20, 11, np.argmax(flux_CIII_sf[13])]
# Peak doesnt exist for 204, 22429, 5199, 48, 538, 54891, 23124: append a "0"
CIII_sf_SNR.append(flux_CIII_sf[0][index_max_flux_CIII_sf[0]] / noise_CIII_sf[0][index_max_flux_CIII_sf[0]])
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(flux_CIII_sf[2][index_max_flux_CIII_sf[2]] / noise_CIII_sf[2][index_max_flux_CIII_sf[2]])
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(flux_CIII_sf[6][index_max_flux_CIII_sf[6]] / noise_CIII_sf[6][index_max_flux_CIII_sf[6]])
CIII_sf_SNR.append(flux_CIII_sf[7][index_max_flux_CIII_sf[7]] / noise_CIII_sf[7][index_max_flux_CIII_sf[7]])
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(flux_CIII_sf[10][index_max_flux_CIII_sf[10]] / noise_CIII_sf[10][index_max_flux_CIII_sf[10]])
CIII_sf_SNR.append(flux_CIII_sf[11][index_max_flux_CIII_sf[11]] / noise_CIII_sf[11][index_max_flux_CIII_sf[11]])
CIII_sf_SNR.append(0)
CIII_sf_SNR.append(flux_CIII_sf[13][index_max_flux_CIII_sf[13]] / noise_CIII_sf[13][index_max_flux_CIII_sf[13]])
print('The C-III] SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, '
      '23124, 53, 218, 54891, 7876: ', CIII_sf_SNR)

index_max_flux_OIII_sf = [8, 8, np.argmax(flux_O_III_sf[2]), 13, np.argmax(flux_O_III_sf[4]),
                          np.argmax(flux_O_III_sf[5]), np.argmax(flux_O_III_sf[6]), np.argmax(flux_O_III_sf[7]),
                          np.argmax(flux_O_III_sf[8]), np.argmax(flux_O_III_sf[9]), 12, np.argmax(flux_O_III_sf[11]),
                          0, np.argmax(flux_O_III_sf[13])]
# Peak doesnt exist for 204, 22429, 48, 538, 23124, 54891   : append a "0"
O_III_sf_SNR.append(flux_O_III_sf[0][index_max_flux_OIII_sf[0]] / noise_O_III_sf[0][index_max_flux_OIII_sf[0]])
O_III_sf_SNR.append(0)
O_III_sf_SNR.append(flux_O_III_sf[2][index_max_flux_OIII_sf[2]] / noise_O_III_sf[2][index_max_flux_OIII_sf[2]])
O_III_sf_SNR.append(0)
O_III_sf_SNR.append(flux_O_III_sf[4][index_max_flux_OIII_sf[4]] / noise_O_III_sf[4][index_max_flux_OIII_sf[4]])
O_III_sf_SNR.append(0)
O_III_sf_SNR.append(flux_O_III_sf[6][index_max_flux_OIII_sf[6]] / noise_O_III_sf[6][index_max_flux_OIII_sf[6]])
O_III_sf_SNR.append(flux_O_III_sf[7][index_max_flux_OIII_sf[7]] / noise_O_III_sf[7][index_max_flux_OIII_sf[7]])
O_III_sf_SNR.append(0)
O_III_sf_SNR.append(0)
O_III_sf_SNR.append(flux_O_III_sf[10][index_max_flux_OIII_sf[10]] / noise_O_III_sf[10][index_max_flux_OIII_sf[10]])
O_III_sf_SNR.append(flux_O_III_sf[11][index_max_flux_OIII_sf[11]] / noise_O_III_sf[11][index_max_flux_OIII_sf[11]])
O_III_sf_SNR.append(0)
O_III_sf_SNR.append(flux_O_III_sf[13][index_max_flux_OIII_sf[13]] / noise_O_III_sf[13][index_max_flux_OIII_sf[13]])
print('The O-III] SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, '
      '23124, 53, 218, 54891, 7876: ', O_III_sf_SNR)

index_max_flux_SiIII_sf = [13, 0, 14, 20, 0, 0, np.argmax(flux_SiIII_sf[6]),
                           np.argmax(flux_SiIII_sf[7]), 0, 0, 0, 0, np.argmax(flux_SiIII_sf[12]), 0]
# Peak doesnt exist for 88, 204, 435, 5199, 48, 538, 23124, 53, 218, 7876
SiIII_sf_SNR.append(0)
SiIII_sf_SNR.append(0)
SiIII_sf_SNR.append(flux_SiIII_sf[2][index_max_flux_SiIII_sf[2]] / noise_SiIII_sf[2][index_max_flux_SiIII_sf[2]])
SiIII_sf_SNR.append(flux_SiIII_sf[3][index_max_flux_SiIII_sf[3]] / noise_SiIII_sf[3][index_max_flux_SiIII_sf[3]])
SiIII_sf_SNR.append(0)
SiIII_sf_SNR.append(0)
SiIII_sf_SNR.append(flux_SiIII_sf[6][index_max_flux_SiIII_sf[6]] / noise_SiIII_sf[6][index_max_flux_SiIII_sf[6]])
SiIII_sf_SNR.append(flux_SiIII_sf[7][index_max_flux_SiIII_sf[7]] / noise_SiIII_sf[7][index_max_flux_SiIII_sf[7]])
SiIII_sf_SNR.append(0)  #
SiIII_sf_SNR.append(0)  #
SiIII_sf_SNR.append(0)
SiIII_sf_SNR.append(0)
SiIII_sf_SNR.append(flux_SiIII_sf[12][index_max_flux_SiIII_sf[12]] / noise_SiIII_sf[12][index_max_flux_SiIII_sf[12]])
SiIII_sf_SNR.append(0)
print('The Si-III] SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, '
      '23124, 53, 218, 54891, 7876: ', SiIII_sf_SNR)

print('#############################################################################################################')
figNr = 0

# C -III ANALYSIS   Peak doesnt exist for 204, 22429, 5199, 48, 538, 54891, 23124
##################################################################################################################
FWHM_CIII = [1., 1., 0.5, 1., 1., 1., 1., 1., 1., 1., 1., 0.5, 1., 0.5]
Delta_wavelen_CIII = [3, 4, 6, 5, 3, 7, 4, 7, 0, 4, 3, .5, 4, 4] # from plots

params_CIII_88 = Parameters()
params_CIII_88.add('amp_%s' % (iden_str[0]), value=flux_CIII_sf[0][index_max_flux_CIII_sf[0]], min=0)
params_CIII_88.add('cen_%s' % (iden_str[0]), value=1906, min=CIII_sf - Delta_wavelen_CIII[0], max=CIII_sf)
params_CIII_88.add('sigma_%s' % (iden_str[0]), value=0.5, min=0.01, max=FWHM_CIII[0])
params_CIII_88.add('continuum_%s' % (iden_str[0]), value=np.nanmedian(flux_CIII_sf[0]))
minner88 = Minimizer(objective88, params_CIII_88, fcn_args=(rest_wavelen_CIII_sf[0], flux_CIII_sf[0]))
result_CIII88 = minner88.minimize()
final_CIII_88 = flux_CIII_sf[0] + result_CIII88.residual
n0_final_CIII_88 = final_CIII_88[final_CIII_88 != 0]
indexs = np.array(np.argwhere(n0_final_CIII_88 != 0))
Index = np.concatenate(indexs)
rest_wavelen_CIII_sf[0] = rest_wavelen_CIII_sf[0][Index]
n0_x = rest_wavelen_CIII_sf[0][rest_wavelen_CIII_sf[0] != 0]
n0_final_CIII_88 = n0_final_CIII_88[0:len(n0_x)]
n0_noise_CIII_88 = noise_CIII_sf[0][0:len(n0_x)]
ModelResult88 = ModelResult(Gauss_model, params_CIII_88, weights=True, nan_policy='propagate')
n0_final_CIII_int_flux_88 = n0_final_CIII_88[n0_final_CIII_88 > 6.2]
index_int_flux = np.array(np.argwhere(n0_final_CIII_88 > 6.2))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_88 = simps(n0_final_CIII_int_flux_88, n0_x_int_flux)
print('INTEGRATED FLUX OF CIII] of 88 is: ', integrate_88)
Sigma_88 = result_CIII88.params['sigma_88'].value
FWHM_CIII_Gauss_88 = 2 * np.sqrt(2*np.log(2)) * Sigma_88
print('THE CIII] FWHM FOR TARGET 88 IS: ', FWHM_CIII_Gauss_88)
Continuum_88 = result_CIII88.params['continuum_88'].value
dlambda_CIII_88 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIII_88 = integrate_88 / (Continuum_88 * (1 + z[0]))
print('THE CIII] EW FOR TARGET 88 IS: ', EW_CIII_88)
figure = plt.figure(figNr)
plt.step(rest_wavelen_CIII_sf[0], flux_CIII_sf[0], 'b', label='flux')
plt.step(rest_wavelen_CIII_sf[0], noise_CIII_sf[0], 'k', label='noise')
plt.plot(rest_wavelen_CIII_sf[0], final_CIII_88, 'r', label='fit')
plt.axhline(y=Continuum_88, color='r')
plt.axvline(x=CIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C III] rest-peak of target 88')
plt.savefig('plots/ALL_PART5/CIII_Gauss_target_88.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIII_Gauss_target_88.pdf')
plt.clf()
figNr += 1

params_CIII_435 = Parameters()
params_CIII_435.add('amp_%s' % (iden_str[2]), value=max(flux_CIII_sf[2]), min=0)
params_CIII_435.add('cen_%s' % (iden_str[2]), value=1903, min=CIII_sf - Delta_wavelen_CIII[2], max=CIII_sf)
params_CIII_435.add('sigma_%s' % (iden_str[2]), value=0.5, min=0.01, max=FWHM_CIII[2])
params_CIII_435.add('continuum_%s' % (iden_str[2]), value=np.nanmedian(flux_CIII_sf[2]))
minner435 = Minimizer(objective435, params_CIII_435, fcn_args=(rest_wavelen_CIII_sf[2], flux_CIII_sf[2]))
result_CIII435 = minner435.minimize()
final_CIII_435 = flux_CIII_sf[2] + result_CIII435.residual
n0_final_CIII_435 = final_CIII_435[final_CIII_435 != 0]
indexs = np.array(np.argwhere(n0_final_CIII_435 != 0))
Index = np.concatenate(indexs)
rest_wavelen_CIII_sf[2] = rest_wavelen_CIII_sf[2][Index]
n0_x = rest_wavelen_CIII_sf[2][rest_wavelen_CIII_sf[2] != 0]
n0_final_CIII_435 = n0_final_CIII_435[0:len(n0_x)]
n0_noise_CIII_435 = noise_CIII_sf[2][0:len(n0_x)]
ModelResult435 = ModelResult(Gauss_model, params_CIII_435, weights=True, nan_policy='propagate')
n0_final_CIII_int_flux_435 = n0_final_CIII_435[n0_final_CIII_435 > 1.35]
index_int_flux = np.array(np.argwhere(n0_final_CIII_435 > 1.35))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_435 = result_CIII435.params['continuum_435'].value
integrate_435 = simps(n0_final_CIII_int_flux_435 - Continuum_435, n0_x_int_flux)
print('INTEGRATED FLUX OF CIII] of 435 is: ', integrate_435)
Sigma_435 = result_CIII435.params['sigma_435'].value
FWHM_CIII_Gauss_435 = 2 * np.sqrt(2*np.log(2)) * Sigma_435
print('THE CIII] FWHM FOR TARGET 435 IS: ', FWHM_CIII_Gauss_435)
dlambda_CIII_435 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIII_435 = integrate_435 / (Continuum_435 * (1 + z[2]))
print('THE CIII] EW FOR TARGET 435 IS: ', EW_CIII_435)
figure = plt.figure(figNr)
plt.step(rest_wavelen_CIII_sf[2], flux_CIII_sf[2], 'b', label='flux')
plt.step(rest_wavelen_CIII_sf[2], noise_CIII_sf[2], 'k', label='noise')
plt.plot(rest_wavelen_CIII_sf[2], final_CIII_435, 'r', label='fit')
plt.axhline(y=Continuum_435, color='r')
plt.axvline(x=CIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C III] rest-peak of target 435')
plt.savefig('plots/ALL_PART5/CIII_Gauss_target_435.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIII_Gauss_target_435.pdf')
plt.clf()
figNr += 1

params_CIII_131 = Parameters()
params_CIII_131.add('amp_%s' % (iden_str[6]), value=max(flux_CIII_sf[6]), min=0)
params_CIII_131.add('cen_%s' % (iden_str[6]), value=1906, min=CIII_sf - Delta_wavelen_CIII[6], max=CIII_sf)
params_CIII_131.add('sigma_%s' % (iden_str[6]), value=0.5, min=0.01, max=FWHM_CIII[6])
params_CIII_131.add('continuum_%s' % (iden_str[6]), value=np.nanmedian(flux_CIII_sf[6]))
minner131 = Minimizer(objective131, params_CIII_131, fcn_args=(rest_wavelen_CIII_sf[6], flux_CIII_sf[6]))
result_CIII131 = minner131.minimize()
final_CIII_131 = flux_CIII_sf[6] + result_CIII131.residual
n0_final_CIII_131 = final_CIII_131[final_CIII_131 != 0]
indexs = np.array(np.argwhere(n0_final_CIII_131 != 0))
Index = np.concatenate(indexs)
rest_wavelen_CIII_sf[6] = rest_wavelen_CIII_sf[6][Index]
n0_x = rest_wavelen_CIII_sf[6][rest_wavelen_CIII_sf[6] != 0]
n0_final_CIII_131 = n0_final_CIII_131[0:len(n0_x)]
n0_noise_CIII_131 = noise_CIII_sf[6][0:len(n0_x)]
ModelResult131 = ModelResult(Gauss_model, params_CIII_131, weights=True, nan_policy='propagate')
n0_final_CIII_int_flux_131 = n0_final_CIII_131[n0_final_CIII_131 > 2.73]
index_int_flux = np.array(np.argwhere(n0_final_CIII_131 > 2.73))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_131 = result_CIII131.params['continuum_131'].value
integrate_131 = simps(n0_final_CIII_int_flux_131 - Continuum_131, n0_x_int_flux)
print('INTEGRATED FLUX OF CIII] of 131 is: ', integrate_131)
Sigma_131 = result_CIII131.params['sigma_131'].value
FWHM_CIII_Gauss_131 = 2 * np.sqrt(2*np.log(2)) * Sigma_131
print('THE CIII] FWHM FOR TARGET 131 IS: ', FWHM_CIII_Gauss_131)
dlambda_CIII_131 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIII_131 = integrate_131 / (Continuum_131 * (1 + z[6]))
print('THE CIII] EW FOR TARGET 131 IS: ', EW_CIII_131)
figure = plt.figure(figNr)
plt.step(rest_wavelen_CIII_sf[6], flux_CIII_sf[6], 'b', label='flux')
plt.step(rest_wavelen_CIII_sf[6], noise_CIII_sf[6], 'k', label='noise')
plt.plot(rest_wavelen_CIII_sf[6], final_CIII_131, 'r', label='fit')
plt.axhline(y=Continuum_131, color='r')
plt.axvline(x=CIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C III] rest-peak of target 131')
plt.savefig('plots/ALL_PART5/CIII_Gauss_target_131.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIII_Gauss_target_131.pdf')
plt.clf()
figNr += 1

params_CIII_118 = Parameters()
params_CIII_118.add('amp_%s' % (iden_str[7]), value=max(flux_CIII_sf[7]), min=0)
params_CIII_118.add('cen_%s' % (iden_str[7]), value=1906, min=CIII_sf - Delta_wavelen_CIII[7], max=CIII_sf)
params_CIII_118.add('sigma_%s' % (iden_str[7]), value=0.5, min=0.01, max=FWHM_CIII[7])
params_CIII_118.add('continuum_%s' % (iden_str[7]), value=np.nanmedian(flux_CIII_sf[7]))
minner118 = Minimizer(objective118, params_CIII_118, fcn_args=(rest_wavelen_CIII_sf[7], flux_CIII_sf[7]))
result_CIII118 = minner118.minimize()
final_CIII_118 = flux_CIII_sf[7] + result_CIII118.residual
n0_final_CIII_118 = final_CIII_118[final_CIII_118 != 0]
indexs = np.array(np.argwhere(n0_final_CIII_118 != 0))
Index = np.concatenate(indexs)
rest_wavelen_CIII_sf[7] = rest_wavelen_CIII_sf[7][Index]
n0_x = rest_wavelen_CIII_sf[7][rest_wavelen_CIII_sf[7] != 0]
n0_final_CIII_118 = n0_final_CIII_118[0:len(n0_x)]
n0_noise_CIII_118 = noise_CIII_sf[7][0:len(n0_x)]
ModelResult118 = ModelResult(Gauss_model, params_CIII_118, weights=True, nan_policy='propagate')
n0_final_CIII_int_flux_118 = n0_final_CIII_118[n0_final_CIII_118 > 3.669]
index_int_flux = np.array(np.argwhere(n0_final_CIII_118 > 3.669))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_118 = result_CIII118.params['continuum_118'].value
integrate_118 = simps(n0_final_CIII_int_flux_118 - Continuum_118, n0_x_int_flux)
print('INTEGRATED FLUX OF CIII] of 118 is: ', integrate_118)
Sigma_118 = result_CIII118.params['sigma_118'].value
FWHM_CIII_Gauss_118 = 2 * np.sqrt(2*np.log(2)) * Sigma_118
print('THE CIII] FWHM FOR TARGET 118 IS: ', FWHM_CIII_Gauss_118)
dlambda_CIII_118 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIII_118 = integrate_118 / (Continuum_118 * (1 + z[7]))
print('THE CIII] EW FOR TARGET 118 IS: ', EW_CIII_118)
figure = plt.figure(figNr)
plt.step(rest_wavelen_CIII_sf[7], flux_CIII_sf[7], 'b', label='flux')
plt.step(rest_wavelen_CIII_sf[7], noise_CIII_sf[7], 'k', label='noise')
plt.plot(rest_wavelen_CIII_sf[7], final_CIII_118, 'r', label='fit')
plt.axhline(y=Continuum_118, color='r')
plt.axvline(x=CIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C III] rest-peak of target 118')
plt.savefig('plots/ALL_PART5/CIII_Gauss_target_118.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIII_Gauss_target_118.pdf')
plt.clf()
figNr += 1

params_CIII_218 = Parameters()
params_CIII_218.add('amp_%s' % (iden_str[11]), value=flux_CIII_sf[11][index_max_flux_CIII_sf[11]])
params_CIII_218.add('cen_%s' % (iden_str[11]), value=1908, min=CIII_sf - Delta_wavelen_CIII[11])
params_CIII_218.add('sigma_%s' % (iden_str[11]), value=3, min=0.01, max=FWHM_CIII[11])
params_CIII_218.add('continuum_%s' % (iden_str[11]), value=np.nanmedian(flux_CIII_sf[11]), vary=False)
minner218 = Minimizer(objective218, params_CIII_218, fcn_args=(rest_wavelen_CIII_sf[11], flux_CIII_sf[11]))
result_CIII218 = minner218.minimize()
final_CIII_218 = flux_CIII_sf[11] + result_CIII218.residual
n0_final_CIII_218 = final_CIII_218[final_CIII_218 != 0]
indexs = np.array(np.argwhere(n0_final_CIII_218 != 0))
Index = np.concatenate(indexs)
rest_wavelen_CIII_sf[11] = rest_wavelen_CIII_sf[11][Index]
n0_x = rest_wavelen_CIII_sf[11][rest_wavelen_CIII_sf[11] != 0]
n0_final_CIII_218 = n0_final_CIII_218[0:len(n0_x)]
n0_noise_CIII_218 = noise_CIII_sf[11][0:len(n0_x)]
ModelResult218 = ModelResult(Gauss_model, params_CIII_218, weights=True, nan_policy='propagate')
n0_final_CIII_int_flux_218 = n0_final_CIII_218[n0_final_CIII_218 > 2.038]
index_int_flux = np.array(np.argwhere(n0_final_CIII_218 > 2.038))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_218 = result_CIII218.params['continuum_218'].value
integrate_218 = simps(n0_final_CIII_int_flux_218 - Continuum_218, n0_x_int_flux)
print('INTEGRATED FLUX OF CIII] of 218 is: ', integrate_218)
Sigma_218 = result_CIII218.params['sigma_218'].value
FWHM_CIII_Gauss_218 = 2 * np.sqrt(2*np.log(2)) * Sigma_218
print('THE CIII] FWHM FOR TARGET 218 IS: ', FWHM_CIII_Gauss_218)
dlambda_CIII_218 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIII_218 = integrate_218 / (Continuum_218 * (1 + z[11]))
print('THE CIII] EW FOR TARGET 218 IS: ', EW_CIII_218)
figure = plt.figure(figNr)
plt.step(rest_wavelen_CIII_sf[11], flux_CIII_sf[11], 'b', label='flux')
plt.step(rest_wavelen_CIII_sf[11], noise_CIII_sf[11], 'k', label='noise')
plt.plot(rest_wavelen_CIII_sf[11], final_CIII_218, 'r', label='fit')
plt.axhline(y=Continuum_218, color='r')
plt.axvline(x=CIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C III] rest-peak of target 218')
plt.savefig('plots/ALL_PART5/CIII_Gauss_target_218.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIII_Gauss_target_218.pdf')
plt.clf()
figNr += 1

params_CIII_7876 = Parameters()
params_CIII_7876.add('amp_%s' % (iden_str[13]), value=flux_CIII_sf[13][index_max_flux_CIII_sf[13]], min=5.9)
params_CIII_7876.add('cen_%s' % (iden_str[13]), value=1906, min=CIII_sf - Delta_wavelen_CIII[13], max=CIII_sf)
params_CIII_7876.add('sigma_%s' % (iden_str[13]), value=3, min=0.45, max=FWHM_CIII[13])
params_CIII_7876.add('continuum_%s' % (iden_str[13]), value=np.nanmedian(flux_CIII_sf[13]))
minner7876 = Minimizer(objective7876, params_CIII_7876, fcn_args=(rest_wavelen_CIII_sf[13], flux_CIII_sf[13]))
result_CIII7876 = minner7876.minimize()
final_CIII_7876 = flux_CIII_sf[13] + result_CIII7876.residual
n0_final_CIII_7876 = final_CIII_7876[final_CIII_7876 != 0]
indexs = np.array(np.argwhere(n0_final_CIII_7876 != 0))
Index = np.concatenate(indexs)
rest_wavelen_CIII_sf[13] = rest_wavelen_CIII_sf[13][Index]
n0_x = rest_wavelen_CIII_sf[13][rest_wavelen_CIII_sf[13] != 0]
n0_final_CIII_7876 = n0_final_CIII_7876[0:len(n0_x)]
n0_noise_CIII_7876 = noise_CIII_sf[0][0:len(n0_x)]
ModelResult7876 = ModelResult(Gauss_model, params_CIII_7876, weights=True, nan_policy='propagate')
n0_final_CIII_int_flux_7876 = n0_final_CIII_7876[n0_final_CIII_7876 > 1.52]
index_int_flux = np.array(np.argwhere(n0_final_CIII_7876 > 1.52))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_7876 = result_CIII7876.params['continuum_7876'].value
integrate_7876 = simps(n0_final_CIII_int_flux_7876 - Continuum_7876, n0_x_int_flux)
print('INTEGRATED FLUX OF CIII] of 7876 is: ', integrate_7876)
Sigma_7876 = result_CIII7876.params['sigma_7876'].value
FWHM_CIII_Gauss_7876 = 2 * np.sqrt(2*np.log(2)) * Sigma_7876
print('THE CIII] FWHM FOR TARGET 7876 IS: ', FWHM_CIII_Gauss_7876)
dlambda_CIII_88 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIII_7876 = integrate_7876 / (Continuum_7876 * (1 + z[13]))
print('THE CIII] EW FOR TARGET 7876 IS: ', EW_CIII_7876)
figure = plt.figure(figNr)
plt.step(rest_wavelen_CIII_sf[13], flux_CIII_sf[13], 'b', label='flux')
plt.step(rest_wavelen_CIII_sf[13], noise_CIII_sf[13], 'k', label='noise')
plt.plot(rest_wavelen_CIII_sf[13], final_CIII_7876, 'r', label='fit')
plt.axhline(y=Continuum_7876, color='r')
plt.axvline(x=CIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C III] rest-peak of target 7876')
plt.savefig('plots/ALL_PART5/CIII_Gauss_target_7876.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIII_Gauss_target_7876.pdf')
plt.clf()
figNr += 1

print('#############################################################################################################')

# C -IV ANALYSIS   Peak doesnt exist for 22429, 5199, 48, 131, 538, 53, 54891
##################################################################################################################
#                     0    1     2     3    4    5    6    7     8    9     10   11   12   13
FWHM_CIV =          [0.5,  0.6,  0.5,  0.5, 0.5, 0.5, 0.5, 0.5,  3.,  0.1,  0.5, 1.0, 0.5, 0.5]
Delta_wavelen_CIV = [2.5,  1.5,  2.,   4.5, 6.0, 4.5, 1.0, 0.2,  4.5, 3.0,  1.0, 4.0, 4.0, 4.0] # from plots

params_CIV_204 = Parameters()
params_CIV_204.add('amp_%s' % (iden_str[1]), value=max(flux_C_IV[1]), min=2.4)
params_CIV_204.add('cen_%s' % (iden_str[1]), value=1547, min=C_IV - Delta_wavelen_CIV[1], max=C_IV)
params_CIV_204.add('sigma_%s' % (iden_str[1]), value=5, min=0.5, max=FWHM_CIV[1])
params_CIV_204.add('continuum_%s' % (iden_str[1]), value=np.nanmedian(flux_C_IV[1]))
minner204 = Minimizer(objective204, params_CIV_204, fcn_args=(rest_wavelen_C_IV[1], flux_C_IV[1]))
result_CIV204 = minner204.minimize()
final_CIV_204 = flux_C_IV[1] + result_CIV204.residual
n0_final_CIV_204 = final_CIV_204[final_CIV_204 != 0]
indexs = np.array(np.argwhere(n0_final_CIV_204 != 0))
Index = np.concatenate(indexs)
rest_wavelen_C_IV[1] = rest_wavelen_C_IV[1][Index]
n0_x = rest_wavelen_C_IV[1][rest_wavelen_C_IV[1] != 0]
n0_final_CIV_204 = n0_final_CIV_204[0:len(n0_x)]
n0_noise_CIV_204 = noise_C_IV[1][0:len(n0_x)]
ModelResult204 = ModelResult(Gauss_model, params_CIV_204, weights=True, nan_policy='propagate')
n0_final_CIV_int_flux_204 = n0_final_CIV_204[n0_final_CIV_204 > 0.91]
index_int_flux = np.array(np.argwhere(n0_final_CIV_204 > 0.91))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_204 = result_CIV204.params['continuum_204'].value
integrate_204 = simps(n0_final_CIV_int_flux_204 - Continuum_204, n0_x_int_flux)
print('INTEGRATED FLUX OF CIV of 204 is: ', integrate_204)
Sigma_204 = result_CIV204.params['sigma_204'].value
FWHM_CIV_Gauss_204 = 2 * np.sqrt(2*np.log(2)) * Sigma_204
print('THE CIV FWHM FOR TARGET 204 IS: ', FWHM_CIV_Gauss_204)
dlambda_CIV_204 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIV_204 = integrate_204 / (Continuum_204 * (1 + z[1]))
print('THE C IV EW FOR TARGET 204 IS: ', EW_CIV_204)
figure = plt.figure(figNr)
plt.step(rest_wavelen_C_IV[1], flux_C_IV[1], 'b', label='flux')
plt.step(rest_wavelen_C_IV[1], noise_C_IV[1], 'k', label='noise')
plt.plot(rest_wavelen_C_IV[1], final_CIV_204, 'r', label='fit')
plt.axhline(y=Continuum_204, color='r')
plt.axvline(x=C_IV, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C-IV rest-peak of target 204')
plt.savefig('plots/ALL_PART5/CIV_Gauss_target_204.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIV_Gauss_target_204.pdf')
plt.clf()
figNr += 1

params_CIV_435 = Parameters()
params_CIV_435.add('amp_%s' % (iden_str[2]), value=flux_C_IV[2][index_max_flux_C_IV[2]], min=0)
params_CIV_435.add('cen_%s' % (iden_str[2]), value=1545, min=C_IV - Delta_wavelen_CIV[2], max=C_IV)
params_CIV_435.add('sigma_%s' % (iden_str[2]), value=3, min=0.2, max=FWHM_CIV[2])
params_CIV_435.add('continuum_%s' % (iden_str[2]), value=np.nanmedian(flux_C_IV[2]))
minner435 = Minimizer(objective435, params_CIV_435, fcn_args=(rest_wavelen_C_IV[2], flux_C_IV[2]))
result_CIV435 = minner435.minimize()
final_CIV_435 = flux_C_IV[2] + result_CIV435.residual
n0_final_CIV_435 = final_CIV_435[final_CIV_435 != 0]
indexs = np.array(np.argwhere(n0_final_CIV_435 != 0))
Index = np.concatenate(indexs)
rest_wavelen_C_IV[2] = rest_wavelen_C_IV[2][Index]
n0_x = rest_wavelen_C_IV[2][rest_wavelen_C_IV[2] != 0]
n0_final_CIV_435 = n0_final_CIV_435[0:len(n0_x)]
n0_noise_CIV_435 = noise_C_IV[2][0:len(n0_x)]
ModelResult435 = ModelResult(Gauss_model, params_CIV_435, weights=True, nan_policy='propagate')
n0_final_CIV_int_flux_435 = n0_final_CIV_435[n0_final_CIV_435 > 1.45]
index_int_flux = np.array(np.argwhere(n0_final_CIV_435 > 1.45))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_435 = result_CIV435.params['continuum_435'].value
integrate_435 = simps(n0_final_CIV_int_flux_435 - Continuum_435, n0_x_int_flux)
print('INTEGRATED FLUX OF CIV of 435 is: ', integrate_435)
Sigma_435 = result_CIV435.params['sigma_435'].value
FWHM_CIV_Gauss_435 = 2 * np.sqrt(2*np.log(2)) * Sigma_435
print('THE CIV FWHM FOR TARGET 435 IS: ', FWHM_CIV_Gauss_435)
dlambda_CIV_435 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIV_435 = integrate_435 / (Continuum_435 * (1 + z[2]))
print('THE C IV EW FOR TARGET 435 IS: ', EW_CIV_435)
figure = plt.figure(figNr)
plt.step(rest_wavelen_C_IV[2], flux_C_IV[2], 'b', label='flux')
plt.step(rest_wavelen_C_IV[2], noise_C_IV[2], 'k', label='noise')
plt.plot(rest_wavelen_C_IV[2], final_CIV_435, 'r', label='fit')
plt.axhline(y=Continuum_435, color='r')
plt.axvline(x=C_IV, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C-IV rest-peak of target 435')
plt.savefig('plots/ALL_PART5/CIV_Gauss_target_435.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIV_Gauss_target_435.pdf')
plt.clf()
figNr += 1

params_CIV_118 = Parameters()
params_CIV_118.add('amp_%s' % (iden_str[7]), value=max(flux_C_IV[7]), min=0)
params_CIV_118.add('cen_%s' % (iden_str[7]), value=1547, min=C_IV - Delta_wavelen_CIV[7], max=C_IV)
params_CIV_118.add('sigma_%s' % (iden_str[7]), value=0.5, min=0.01, max=FWHM_CIV[7])
params_CIV_118.add('continuum_%s' % (iden_str[7]), value=np.nanmedian(flux_C_IV[7]))
minner118 = Minimizer(objective118, params_CIV_118, fcn_args=(rest_wavelen_C_IV[7], flux_C_IV[7]))
result_CIV118 = minner118.minimize()
final_CIV_118 = flux_C_IV[7] + result_CIV118.residual
n0_final_CIV_118 = final_CIV_118[final_CIV_118 != 0]
indexs = np.array(np.argwhere(n0_final_CIV_118 != 0))
Index = np.concatenate(indexs)
rest_wavelen_C_IV[7] = rest_wavelen_C_IV[7][Index]
n0_x = rest_wavelen_C_IV[7][rest_wavelen_C_IV[7] != 0]
n0_final_CIV_118 = n0_final_CIV_118[0:len(n0_x)]
n0_noise_CIV_118 = noise_C_IV[7][0:len(n0_x)]
ModelResult118 = ModelResult(Gauss_model, params_CIV_118, weights=True, nan_policy='propagate')
n0_final_CIV_int_flux_118 = n0_final_CIV_118[n0_final_CIV_118 > 3.93]
index_int_flux = np.array(np.argwhere(n0_final_CIV_118 > 3.93))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_118 = result_CIV118.params['continuum_118'].value
integrate_118 = simps(n0_final_CIV_int_flux_118 - Continuum_118, n0_x_int_flux)
print('INTEGRATED FLUX OF CIV of 118 is: ', integrate_118)
Sigma_118 = result_CIV118.params['sigma_118'].value
FWHM_CIV_Gauss_118 = 2 * np.sqrt(2*np.log(2)) * Sigma_118
print('THE CIV FWHM FOR TARGET 118 IS: ', FWHM_CIV_Gauss_118)
dlambda_CIV_118 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIV_118 = integrate_118 / (Continuum_118 * (1 + z[7]))
print('THE C IV EW FOR TARGET 118 IS: ', EW_CIV_118)
figure = plt.figure(figNr)
plt.step(rest_wavelen_C_IV[7], flux_C_IV[7], 'b', label='flux')
plt.step(rest_wavelen_C_IV[7], noise_C_IV[7], 'k', label='noise')
plt.plot(rest_wavelen_C_IV[7], final_CIV_118, 'r', label='fit')
plt.axhline(y=Continuum_118, color='r')
plt.axvline(x=C_IV, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C-IV rest-peak of target 118')
plt.savefig('plots/ALL_PART5/CIV_Gauss_target_118.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIV_Gauss_target_118.pdf')
plt.clf()
figNr += 1

params_CIV_23124 = Parameters()
params_CIV_23124.add('amp_%s' % (iden_str[9]), value=max(flux_C_IV[9]), min=2.2)
params_CIV_23124.add('cen_%s' % (iden_str[9]), value=1545)
params_CIV_23124.add('sigma_%s' % (iden_str[9]), value=1)
params_CIV_23124.add('continuum_%s' % (iden_str[9]), value=np.nanmedian(flux_C_IV[9]))
minner23124 = Minimizer(objective23124, params_CIV_23124, fcn_args=(rest_wavelen_C_IV[9], flux_C_IV[9]))
result_CIV23124 = minner23124.minimize()
final_CIV_23124 = flux_C_IV[9] + result_CIV23124.residual
n0_final_CIV_23124 = final_CIV_23124[final_CIV_23124 != 0]
indexs = np.array(np.argwhere(n0_final_CIV_23124 != 0))
Index = np.concatenate(indexs)
rest_wavelen_C_IV[9] = rest_wavelen_C_IV[9][Index]
n0_x = rest_wavelen_C_IV[9][rest_wavelen_C_IV[9] != 0]
n0_final_CIV_23124 = n0_final_CIV_23124[0:len(n0_x)]
n0_noise_CIV_23124 = noise_C_IV[9][0:len(n0_x)]
ModelResult23124 = ModelResult(Gauss_model, params_CIV_23124, weights=True, nan_policy='propagate')
n0_final_CIV_int_flux_23124 = n0_final_CIV_23124[n0_final_CIV_23124 > 2.72]
index_int_flux = np.array(np.argwhere(n0_final_CIV_23124 > 2.72))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_23124 = result_CIV23124.params['continuum_23124'].value
integrate_23124 = simps(n0_final_CIV_int_flux_23124 - Continuum_23124, n0_x_int_flux)
print('INTEGRATED FLUX OF CIV of 23124 is: ', integrate_23124)
Sigma_23124 = result_CIV23124.params['sigma_23124'].value
FWHM_CIV_Gauss_23124 = 2 * np.sqrt(2*np.log(2)) * Sigma_23124
print('THE CIV FWHM FOR TARGET 23124 IS: ', FWHM_CIV_Gauss_23124)
dlambda_CIV_23124 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIV_23124 = integrate_23124 / (Continuum_23124 * (1 + z[9]))
print('THE C IV EW FOR TARGET 23124 IS: ', EW_CIV_23124)
figure = plt.figure(figNr)
plt.step(rest_wavelen_C_IV[9], flux_C_IV[9], 'b', label='flux')
plt.step(rest_wavelen_C_IV[9], noise_C_IV[9], 'k', label='noise')
plt.plot(rest_wavelen_C_IV[9], final_CIV_23124, 'r', label='fit')
plt.axhline(y=Continuum_23124, color='r')
plt.axvline(x=C_IV, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C-IV rest-peak of target 23124')
plt.savefig('plots/ALL_PART5/CIV_Gauss_target_23124.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIV_Gauss_target_23124.pdf')
plt.clf()
figNr += 1

params_CIV_218 = Parameters()
params_CIV_218.add('amp_%s' % (iden_str[11]), value=flux_C_IV[11][index_max_flux_C_IV[11]], min=2.1)
params_CIV_218.add('cen_%s' % (iden_str[11]), value=1545, min=C_IV - Delta_wavelen_CIV[11], max=C_IV)
params_CIV_218.add('sigma_%s' % (iden_str[11]), value=3, min=0.3, max=FWHM_CIV[11])
params_CIV_218.add('continuum_%s' % (iden_str[11]), value=np.nanmedian(flux_C_IV[11]))
minner218 = Minimizer(objective218, params_CIV_218, fcn_args=(rest_wavelen_C_IV[11], flux_C_IV[11]))
result_CIV218 = minner218.minimize()
final_CIV_218 = flux_C_IV[11] + result_CIV218.residual
n0_final_CIV_218 = final_CIV_218[final_CIV_218 != 0]
indexs = np.array(np.argwhere(n0_final_CIV_218 != 0))
Index = np.concatenate(indexs)
rest_wavelen_C_IV[11] = rest_wavelen_C_IV[11][Index]
n0_x = rest_wavelen_C_IV[11][rest_wavelen_C_IV[11] != 0]
n0_final_CIV_218 = n0_final_CIV_218[0:len(n0_x)]
n0_noise_CIV_218 = noise_C_IV[11][0:len(n0_x)]
ModelResult218 = ModelResult(Gauss_model, params_CIV_218, weights=True, nan_policy='propagate')
n0_final_CIV_int_flux_218 = n0_final_CIV_218[n0_final_CIV_218 > 2.193]
index_int_flux = np.array(np.argwhere(n0_final_CIV_218 > 2.193))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_218 = result_CIV218.params['continuum_218'].value
integrate_218 = simps(n0_final_CIV_int_flux_218 - Continuum_218, n0_x_int_flux)
print('INTEGRATED FLUX OF CIV of 218 is: ', integrate_218)
Sigma_218 = result_CIV218.params['sigma_218'].value
FWHM_CIV_Gauss_218 = 2 * np.sqrt(2*np.log(2)) * Sigma_218
print('THE CIV FWHM FOR TARGET 218 IS: ', FWHM_CIV_Gauss_218)
dlambda_CIV_218 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_CIV_218 = integrate_218 / (Continuum_218 * (1 + z[11]))
print('THE C IV EW FOR TARGET 218 IS: ', EW_CIV_218)
figure = plt.figure(figNr)
plt.step(rest_wavelen_C_IV[11], flux_C_IV[11], 'b', label='flux')
plt.step(rest_wavelen_C_IV[11], noise_C_IV[11], 'k', label='noise')
plt.plot(rest_wavelen_C_IV[11], final_CIV_218, 'r', label='fit')
plt.axhline(y=Continuum_218, color='r')
plt.axvline(x=C_IV, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('C-IV rest-peak of target 218')
plt.savefig('plots/ALL_PART5/CIV_Gauss_target_218.pdf')
plt.savefig('MAIN_LATEX/PLOTS/CIV_Gauss_target_218.pdf')
plt.clf()
figNr += 1

print('#############################################################################################################')

# O -III ANALYSIS   Peak doesnt exist for 204, 22429, 48, 538, 23124, 54891
##################################################################################################################
#                     0   1   2    3   4    5   6   7   8   9   10  11  12  13
FWHM_OIII =          [1., 3., 0.5, 3., 1.,  3., 3., 3., 3., 3., 1., 3., 3., 1.]
Delta_wavelen_OIII = [5,  0,  3,   3,  5,   0,  2,  4,  0,  0,  3,  1,  0,  0.5]  # from plots

params_OIII_88 = Parameters()
params_OIII_88.add('amp_%s' % (iden_str[0]), value=flux_O_III_sf[0][index_max_flux_OIII_sf[0]])
params_OIII_88.add('cen_%s' % (iden_str[0]), value=1661, min=O_III_sf - Delta_wavelen_OIII[0], max=1662)
params_OIII_88.add('sigma_%s' % (iden_str[0]), value=0.3, min=0.2, max=FWHM_OIII[0])
params_OIII_88.add('continuum_%s' % (iden_str[0]), value=np.nanmedian(flux_O_III_sf[0]))
minner88 = Minimizer(objective88, params_OIII_88, fcn_args=(rest_wavelen_O_III_sf[0], flux_O_III_sf[0]))
result_OIII88 = minner88.minimize()
final_OIII_88 = flux_O_III_sf[0] + result_OIII88.residual
n0_final_OIII_88 = final_OIII_88[final_OIII_88 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_88 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[0] = rest_wavelen_O_III_sf[0][Index]
n0_x = rest_wavelen_O_III_sf[0][rest_wavelen_O_III_sf[0] != 0]
n0_final_OIII_88 = n0_final_OIII_88[0:len(n0_x)]
n0_noise_OIII_88 = noise_O_III_sf[0][0:len(n0_x)]
ModelResult88 = ModelResult(Gauss_model, params_OIII_88, weights=True, nan_policy='propagate')
n0_final_OIII_int_flux_88 = n0_final_OIII_88[n0_final_OIII_88 > 6.21]
index_int_flux = np.array(np.argwhere(n0_final_OIII_88 > 6.21))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_88 = result_OIII88.params['continuum_88'].value
integrate_88 = simps(n0_final_OIII_int_flux_88 - Continuum_88, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 88 is: ', integrate_88)
Sigma_88 = result_OIII88.params['sigma_88'].value
FWHM_OIII_Gauss_88 = 2 * np.sqrt(2*np.log(2)) * Sigma_88
print('THE OIII FWHM FOR TARGET 88 IS: ', FWHM_OIII_Gauss_88)
dlambda_OIII_88 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_88 = integrate_88 / (Continuum_88 * (1 + z[0]))
print('THE O III] EW FOR TARGET 88 IS: ', EW_OIII_88)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[0], flux_O_III_sf[0], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[0], noise_O_III_sf[0], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[0], final_OIII_88, 'r', label='fit')
plt.axhline(y=Continuum_88, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 88')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_88.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_88.pdf')
plt.clf()
figNr += 1

params_OIII_435 = Parameters()
params_OIII_435.add('amp_%s' % (iden_str[2]), value=max(flux_O_III_sf[2]), min=6.)
params_OIII_435.add('cen_%s' % (iden_str[2]), value=1663, min=O_III_sf - Delta_wavelen_OIII[2], max=O_III_sf)
params_OIII_435.add('sigma_%s' % (iden_str[2]), value=3, min=0.1, max=FWHM_OIII[2])
params_OIII_435.add('continuum_%s' % (iden_str[2]), value=np.nanmedian(flux_O_III_sf[2]))
minner435 = Minimizer(objective435, params_OIII_435, fcn_args=(rest_wavelen_O_III_sf[2], flux_O_III_sf[2]))
result_OIII435 = minner435.minimize()
final_OIII_435 = flux_O_III_sf[2] + result_OIII435.residual
n0_final_OIII_435 = final_OIII_435[final_OIII_435 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_435 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[2] = rest_wavelen_O_III_sf[2][Index]
n0_x = rest_wavelen_O_III_sf[2][rest_wavelen_O_III_sf[2] != 0]
n0_final_OIII_435 = n0_final_OIII_435[0:len(n0_x)]
n0_noise_OIII_435 = noise_O_III_sf[2][0:len(n0_x)]
ModelResult435 = ModelResult(Gauss_model, params_OIII_435, weights=True, nan_policy='propagate')
n0_final_OIII_int_flux_435 = n0_final_OIII_435[n0_final_OIII_435 > 1.048]
index_int_flux = np.array(np.argwhere(n0_final_OIII_435 > 1.048))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_435 = result_OIII435.params['continuum_435'].value
integrate_435 = simps(n0_final_OIII_int_flux_435 - Continuum_435, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 435 is: ', integrate_435)
Sigma_435 = result_OIII435.params['sigma_435'].value
FWHM_OIII_Gauss_435 = 2 * np.sqrt(2*np.log(2)) * Sigma_435
print('THE OIII FWHM FOR TARGET 435 IS: ', FWHM_OIII_Gauss_435)
dlambda_OIII_435 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_435 = integrate_435 / (Continuum_435 * (1 + z[2]))
print('THE O III] EW FOR TARGET 435 IS: ', EW_OIII_435)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[2], flux_O_III_sf[2], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[2], noise_O_III_sf[2], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[2], final_OIII_435, 'r', label='fit')
plt.axhline(y=Continuum_435, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 435')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_435.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_435.pdf')
plt.clf()
figNr += 1

params_OIII_5199 = Parameters()
params_OIII_5199.add('amp_%s' % (iden_str[4]), value=max(flux_O_III_sf[4]), min=2.)
params_OIII_5199.add('cen_%s' % (iden_str[4]), value=rest_wavelen_O_III_sf[4][4])
params_OIII_5199.add('sigma_%s' % (iden_str[4]), value=3, min=0.5, max=FWHM_OIII[4])
params_OIII_5199.add('continuum_%s' % (iden_str[4]), value=np.nanmedian(flux_O_III_sf[4]))
minner5199 = Minimizer(objective5199, params_OIII_5199, fcn_args=(rest_wavelen_O_III_sf[4], flux_O_III_sf[4]))
result_OIII5199 = minner5199.minimize()
final_OIII_5199 = flux_O_III_sf[4] + result_OIII5199.residual
n0_final_OIII_5199 = final_OIII_5199[final_OIII_5199 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_5199 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[4] = rest_wavelen_O_III_sf[4][Index]
n0_x = rest_wavelen_O_III_sf[4][rest_wavelen_O_III_sf[4] != 0]
n0_final_OIII_5199 = n0_final_OIII_5199[0:len(n0_x)]
n0_noise_OIII_5199 = noise_O_III_sf[4][0:len(n0_x)]
ModelResult5199 = ModelResult(Gauss_model, params_OIII_5199, weights=True, nan_policy='propagate')
n0_final_OIII_int_flux_5199 = n0_final_OIII_5199[n0_final_OIII_5199 > 0.6304]
index_int_flux = np.array(np.argwhere(n0_final_OIII_5199 > 0.6304))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_5199 = result_OIII5199.params['continuum_5199'].value
integrate_5199 = simps(n0_final_OIII_int_flux_5199 - Continuum_5199, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 5199 is: ', integrate_5199)
Sigma_5199 = result_OIII5199.params['sigma_5199'].value
FWHM_OIII_Gauss_5199 = 2 * np.sqrt(2*np.log(2)) * Sigma_5199
print('THE OIII FWHM FOR TARGET 5199 IS: ', FWHM_OIII_Gauss_5199)
dlambda_OIII_5199 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_5199 = integrate_5199 / (Continuum_5199 * (1 + z[4]))
print('THE O III] EW FOR TARGET 5199 IS: ', EW_OIII_5199)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[4], flux_O_III_sf[4], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[4], noise_O_III_sf[4], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[4], final_OIII_5199, 'r', label='fit')
plt.axhline(y=Continuum_5199, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 5199')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_5199.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_5199.pdf')
plt.clf()
figNr += 1

params_OIII_131 = Parameters()
params_OIII_131.add('amp_%s' % (iden_str[6]), value=max(flux_O_III_sf[6]), min=4)
params_OIII_131.add('cen_%s' % (iden_str[6]), value=1665, min=O_III_sf - Delta_wavelen_OIII[6], max=O_III_sf)
params_OIII_131.add('sigma_%s' % (iden_str[6]), value=0.5, min=0.01, max=FWHM_OIII[6])
params_OIII_131.add('continuum_%s' % (iden_str[6]), value=np.nanmedian(flux_O_III_sf[6]))
minner131 = Minimizer(objective131, params_OIII_131, fcn_args=(rest_wavelen_O_III_sf[6], flux_O_III_sf[6]))
result_OIII131 = minner131.minimize()
final_OIII_131 = flux_O_III_sf[6] + result_OIII131.residual
n0_final_OIII_131 = final_OIII_131[final_OIII_131 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_131 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[6] = rest_wavelen_O_III_sf[6][Index]
n0_x = rest_wavelen_O_III_sf[6][rest_wavelen_O_III_sf[6] != 0]
n0_final_OIII_131 = n0_final_OIII_131[0:len(n0_x)]
n0_noise_OIII_131 = noise_O_III_sf[6][0:len(n0_x)]
ModelResult131 = ModelResult(Gauss_model, params_OIII_131, weights=True, nan_policy='propagate')
n0_final_OIII_int_flux_131 = n0_final_OIII_131[n0_final_OIII_131 > 2.7543]
index_int_flux = np.array(np.argwhere(n0_final_OIII_131 > 2.7543))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_131 = result_OIII131.params['continuum_131'].value
integrate_131 = simps(n0_final_OIII_int_flux_131 - Continuum_131, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 131 is: ', integrate_131)
Sigma_131 = result_OIII131.params['sigma_131'].value
FWHM_OIII_Gauss_131 = 2 * np.sqrt(2*np.log(2)) * Sigma_131
print('THE OIII FWHM FOR TARGET 131 IS: ', FWHM_OIII_Gauss_131)
dlambda_OIII_131 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_131 = integrate_131 / (Continuum_131 * (1 + z[6]))
print('THE O III] EW FOR TARGET 131 IS: ', EW_OIII_131)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[6], flux_O_III_sf[6], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[6], noise_O_III_sf[6], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[6], final_OIII_131, 'r', label='fit')
plt.axhline(y=Continuum_131, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 131')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_131.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_131.pdf')
plt.clf()
figNr += 1

params_OIII_118 = Parameters()
params_OIII_118.add('amp_%s' % (iden_str[7]), value=max(flux_O_III_sf[7]), min=4.)
params_OIII_118.add('cen_%s' % (iden_str[7]), value=1662.5, min=O_III_sf - Delta_wavelen_OIII[7], max=O_III_sf)
params_OIII_118.add('sigma_%s' % (iden_str[7]), value=3, min=0.5, max=FWHM_OIII[7])
params_OIII_118.add('continuum_%s' % (iden_str[7]), value=np.nanmedian(flux_O_III_sf[7]))
minner118 = Minimizer(objective118, params_OIII_118, fcn_args=(rest_wavelen_O_III_sf[7], flux_O_III_sf[7]))
result_OIII118 = minner118.minimize()
final_OIII_118 = flux_O_III_sf[7] + result_OIII118.residual
n0_final_OIII_118 = final_OIII_118[final_OIII_118 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_118 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[7] = rest_wavelen_O_III_sf[7][Index]
n0_x = rest_wavelen_O_III_sf[7][rest_wavelen_O_III_sf[7] != 0]
n0_final_OIII_118 = n0_final_OIII_118[0:len(n0_x)]
n0_noise_OIII_118 = noise_O_III_sf[7][0:len(n0_x)]
ModelResult118 = ModelResult(Gauss_model, params_OIII_118, weights=True, nan_policy='propagate')
Continuum_118 = result_OIII118.params['continuum_118'].value
n0_final_OIII_int_flux_118 = n0_final_OIII_118[n0_final_OIII_118 > 3.89]
index_int_flux = np.array(np.argwhere(n0_final_OIII_118 > 3.89))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
integrate_118 = simps(n0_final_OIII_int_flux_118 - Continuum_118, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 118 is: ', integrate_118)
Sigma_118 = result_OIII118.params['sigma_118'].value
FWHM_OIII_Gauss_118 = 2 * np.sqrt(2*np.log(2)) * Sigma_118
print('THE OIII FWHM FOR TARGET 118 IS: ', FWHM_OIII_Gauss_118)
dlambda_OIII_118 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_118 = integrate_118 / (Continuum_118 * (1 + z[7]))
print('THE O III] EW FOR TARGET 118 IS: ', EW_OIII_118)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[7], flux_O_III_sf[7], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[7], noise_O_III_sf[7], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[7], final_OIII_118, 'r', label='fit')
plt.axhline(y=Continuum_118, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 118')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_118.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_118.pdf')
plt.clf()
figNr += 1

params_OIII_218 = Parameters()
params_OIII_218.add('amp_%s' % (iden_str[11]), value=max(flux_O_III_sf[11]), min=0)
params_OIII_218.add('cen_%s' % (iden_str[11]), value=1665, min=O_III_sf - Delta_wavelen_OIII[11], max=O_III_sf)
params_OIII_218.add('sigma_%s' % (iden_str[11]), value=0.5, min=0.01, max=FWHM_OIII[11])
params_OIII_218.add('continuum_%s' % (iden_str[11]), value=np.nanmedian(flux_O_III_sf[11]))
minner218 = Minimizer(objective218, params_OIII_218, fcn_args=(rest_wavelen_O_III_sf[11], flux_O_III_sf[11]))
result_OIII218 = minner218.minimize()
final_OIII_218 = flux_O_III_sf[11] + result_OIII218.residual
n0_final_OIII_218 = final_OIII_218[final_OIII_218 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_218 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[11] = rest_wavelen_O_III_sf[11][Index]
n0_x = rest_wavelen_O_III_sf[11][rest_wavelen_O_III_sf[11] != 0]
n0_final_OIII_218 = n0_final_OIII_218[0:len(n0_x)]
n0_noise_OIII_218 = noise_O_III_sf[11][0:len(n0_x)]
ModelResult218 = ModelResult(Gauss_model, params_OIII_218, weights=True, nan_policy='propagate')
n0_final_OIII_int_flux_218 = n0_final_OIII_218[n0_final_OIII_218 > 2.103]
index_int_flux = np.array(np.argwhere(n0_final_OIII_218 > 2.103))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_218 = result_OIII218.params['continuum_218'].value
integrate_218 = simps(n0_final_OIII_int_flux_218 - Continuum_218, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 218 is: ', integrate_218)
Sigma_218 = result_OIII218.params['sigma_218'].value
FWHM_OIII_Gauss_218 = 2 * np.sqrt(2*np.log(2)) * Sigma_218
print('THE OIII FWHM FOR TARGET 218 IS: ', FWHM_OIII_Gauss_218)
dlambda_OIII_218 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_218 = integrate_218 / (Continuum_218 * (1 + z[11]))
print('THE O III] EW FOR TARGET 218 IS: ', EW_OIII_218)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[11], flux_O_III_sf[11], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[11], noise_O_III_sf[11], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[11], final_OIII_218, 'r', label='fit')
plt.axhline(y=Continuum_218, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 218')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_218.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_218.pdf')
plt.clf()
figNr += 1

params_OIII_7876 = Parameters()
params_OIII_7876.add('amp_%s' % (iden_str[13]), value=flux_O_III_sf[13][index_max_flux_OIII_sf[13]], min=0)
params_OIII_7876.add('cen_%s' % (iden_str[13]), value=1665.9)
params_OIII_7876.add('sigma_%s' % (iden_str[13]), value=0.5, min=0.01, max=FWHM_OIII[13])
params_OIII_7876.add('continuum_%s' % (iden_str[13]), value=np.nanmedian(flux_O_III_sf[13]))
minner7876 = Minimizer(objective7876, params_OIII_7876, fcn_args=(rest_wavelen_O_III_sf[13], flux_O_III_sf[13]))
result_OIII7876 = minner7876.minimize()
final_OIII_7876 = flux_O_III_sf[13] + result_OIII7876.residual
n0_final_OIII_7876 = final_OIII_7876[final_OIII_7876 != 0]
indexs = np.array(np.argwhere(n0_final_OIII_7876 != 0))
Index = np.concatenate(indexs)
rest_wavelen_O_III_sf[13] = rest_wavelen_O_III_sf[13][Index]
n0_x = rest_wavelen_O_III_sf[13][rest_wavelen_O_III_sf[13] != 0]
n0_final_OIII_7876 = n0_final_OIII_7876[0:len(n0_x)]
n0_noise_OIII_7876 = noise_O_III_sf[13][0:len(n0_x)]
ModelResult7876 = ModelResult(Gauss_model, params_OIII_7876, weights=True, nan_policy='propagate')
n0_final_OIII_int_flux_7876 = n0_final_OIII_7876[n0_final_OIII_7876 > 2.006]
index_int_flux = np.array(np.argwhere(n0_final_OIII_7876 > 2.006))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_7876 = result_OIII7876.params['continuum_7876'].value
integrate_7876 = simps(n0_final_OIII_int_flux_7876 - Continuum_7876, n0_x_int_flux)
print('INTEGRATED FLUX OF OIII] of 7876 is: ', integrate_7876)
Sigma_7876 = result_OIII7876.params['sigma_7876'].value
FWHM_OIII_Gauss_7876 = 2 * np.sqrt(2*np.log(2)) * Sigma_7876
print('THE OIII FWHM FOR TARGET 7876 IS: ', FWHM_OIII_Gauss_7876)
dlambda_OIII_7876 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_OIII_7876 = integrate_7876 / (Continuum_7876 * (1 + z[13]))
print('THE O III] EW FOR TARGET 7876 IS: ', EW_OIII_7876)
figure = plt.figure(figNr)
plt.step(rest_wavelen_O_III_sf[13], flux_O_III_sf[13], 'b', label='flux')
plt.step(rest_wavelen_O_III_sf[13], noise_O_III_sf[13], 'k', label='noise')
plt.plot(rest_wavelen_O_III_sf[13], final_OIII_7876, 'r', label='fit')
plt.axhline(y=Continuum_7876, color='r')
plt.axvline(x=O_III_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('O-III] rest-peak of target 7876')
plt.savefig('plots/ALL_PART5/OIII_sf_Gauss_target_7876.pdf')
plt.savefig('MAIN_LATEX/PLOTS/OIII_Gauss_target_7876.pdf')
plt.clf()
figNr += 1

print('#############################################################################################################')
# Si -III ANALYSIS   Peak doesnt exist for  88, 204, 435, 5199, 48, 538, 23124, 53, 218
##################################################################################################################
#                      0   1   2   3   4   5   6   7    8   9   10  11  12  13
FWHM_SiIII =          [3., 3., 3., 1., 3., 3., 3., 1.5, 3., 3., 3., 3., 3., 1.]
Delta_wavelen_SiIII = [5, 0,   3,  1,  5,  0,  2,  5,   0,  0,  3,  1,  0,  4.]  # from plots

params_SiIII_22429 = Parameters()
params_SiIII_22429.add('amp_%s' % (iden_str[3]), value=max(flux_SiIII_sf[3]), min=0)
params_SiIII_22429.add('cen_%s' % (iden_str[3]), value=1882, min=SiIII_sf - Delta_wavelen_SiIII[3], max=SiIII_sf)
params_SiIII_22429.add('sigma_%s' % (iden_str[3]), value=0.5, min=0.01, max=FWHM_SiIII[3])
params_SiIII_22429.add('continuum_%s' % (iden_str[3]), value=np.nanmedian(flux_SiIII_sf[3]))
minner22429 = Minimizer(objective22429, params_SiIII_22429, fcn_args=(rest_wavelen_SiIII_sf[3], flux_SiIII_sf[3]))
result_SiIII22429 = minner22429.minimize()
final_SiIII_22429 = flux_SiIII_sf[3] + result_SiIII22429.residual
n0_final_SiIII_22429 = final_SiIII_22429[final_SiIII_22429 != 0]
indexs = np.array(np.argwhere(n0_final_SiIII_22429 != 0))
Index = np.concatenate(indexs)
rest_wavelen_SiIII_sf[3] = rest_wavelen_SiIII_sf[3][Index]
n0_x = rest_wavelen_SiIII_sf[3][rest_wavelen_SiIII_sf[3] != 0]
n0_final_SiIII_22429 = n0_final_SiIII_22429[0:len(n0_x)]
n0_noise_SiIII_22429 = noise_SiIII_sf[3][0:len(n0_x)]
ModelResult22429 = ModelResult(Gauss_model, params_SiIII_22429, weights=True, nan_policy='propagate')
n0_final_SiIII_int_flux_22429 = n0_final_SiIII_22429[n0_final_SiIII_22429 > 1.073]
index_int_flux = np.array(np.argwhere(n0_final_SiIII_22429 > 1.073))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_22429 = result_SiIII22429.params['continuum_22429'].value
integrate_22429 = simps(n0_final_SiIII_int_flux_22429 - Continuum_22429, n0_x_int_flux)
print('INTEGRATED FLUX OF SiIII] of 22429 is: ', integrate_22429)
Sigma_22429 = result_SiIII22429.params['sigma_22429'].value
FWHM_SiIII_Gauss_22429 = 2 * np.sqrt(2*np.log(2)) * Sigma_22429
print('THE SiIII] FWHM FOR TARGET 22429 IS: ', FWHM_SiIII_Gauss_22429)
dlambda_SiIII_22429 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_SiIII_22429 = integrate_22429 / (Continuum_22429 * (1 + z[3]))
print('THE Si III] EW FOR TARGET 22429 IS: ', EW_SiIII_22429)
figure = plt.figure(figNr)
plt.step(rest_wavelen_SiIII_sf[3], flux_SiIII_sf[3], 'b', label='flux')
plt.step(rest_wavelen_SiIII_sf[3], noise_SiIII_sf[3], 'k', label='noise')
plt.plot(rest_wavelen_SiIII_sf[3], final_SiIII_22429, 'r', label='fit')
plt.axhline(y=Continuum_22429, color='r')
plt.axvline(x=SiIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Si-III] rest-peak of target 22429')
plt.savefig('plots/ALL_PART5/SiIII_sf_Gauss_target_22429.pdf')
plt.savefig('MAIN_LATEX/PLOTS/SiIII_Gauss_target_22429.pdf')
plt.clf()
figNr += 1

params_SiIII_131 = Parameters()
params_SiIII_131.add('amp_%s' % (iden_str[6]), value=max(flux_SiIII_sf[6]), min=0)
params_SiIII_131.add('cen_%s' % (iden_str[6]), value=1881.5)
params_SiIII_131.add('sigma_%s' % (iden_str[6]), value=0.5, min=0.1, max=FWHM_SiIII[6])
params_SiIII_131.add('continuum_%s' % (iden_str[6]), value=np.nanmedian(flux_SiIII_sf[6]))
minner131 = Minimizer(objective131, params_SiIII_131, fcn_args=(rest_wavelen_SiIII_sf[6], flux_SiIII_sf[6]))
result_SiIII131 = minner131.minimize()
final_SiIII_131 = flux_SiIII_sf[6] + result_SiIII131.residual
n0_final_SiIII_131 = final_SiIII_131[final_SiIII_131 != 0]
indexs = np.array(np.argwhere(n0_final_SiIII_131 != 0))
Index = np.concatenate(indexs)
rest_wavelen_SiIII_sf[6] = rest_wavelen_SiIII_sf[6][Index]
n0_x = rest_wavelen_SiIII_sf[6][rest_wavelen_SiIII_sf[6] != 0]
n0_final_SiIII_131 = n0_final_SiIII_131[0:len(n0_x)]
n0_noise_SiIII_131 = noise_SiIII_sf[6][0:len(n0_x)]
ModelResult131 = ModelResult(Gauss_model, params_SiIII_131, weights=True, nan_policy='propagate')
n0_final_SiIII_int_flux_131 = n0_final_SiIII_131[n0_final_SiIII_131 > 2.27]
index_int_flux = np.array(np.argwhere(n0_final_SiIII_131 > 2.27))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_131 = result_SiIII131.params['continuum_131'].value
integrate_131 = simps(n0_final_SiIII_int_flux_131 - Continuum_131, n0_x_int_flux)
print('INTEGRATED FLUX OF SiIII] of 131 is: ', integrate_131)
Sigma_131 = result_SiIII131.params['sigma_131'].value
FWHM_SiIII_Gauss_131 = 2 * np.sqrt(2*np.log(2)) * Sigma_131
print('THE SiIII] FWHM FOR TARGET 131 IS: ', FWHM_SiIII_Gauss_131)
dlambda_SiIII_131 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_SiIII_131 = integrate_131 / (Continuum_131 * (1 + z[6]))
print('THE Si III] EW FOR TARGET 131 IS: ', EW_SiIII_131)
figure = plt.figure(figNr)
plt.step(rest_wavelen_SiIII_sf[6], flux_SiIII_sf[6], 'b', label='flux')
plt.step(rest_wavelen_SiIII_sf[6], noise_SiIII_sf[6], 'k', label='noise')
plt.plot(rest_wavelen_SiIII_sf[6], final_SiIII_131, 'r', label='fit')
plt.axhline(y=Continuum_131, color='r')
plt.axvline(x=SiIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Si-III] rest-peak of target 131')
plt.savefig('plots/ALL_PART5/SiIII_sf_Gauss_target_131.pdf')
plt.savefig('MAIN_LATEX/PLOTS/SiIII_Gauss_target_131.pdf')
plt.clf()
figNr += 1

params_SiIII_118 = Parameters()
params_SiIII_118.add('amp_%s' % (iden_str[7]), value=max(flux_SiIII_sf[7]), min=0)
params_SiIII_118.add('cen_%s' % (iden_str[7]), value=1878.6)
params_SiIII_118.add('sigma_%s' % (iden_str[7]), value=0.5, min=0.01, max=FWHM_SiIII[7])
params_SiIII_118.add('continuum_%s' % (iden_str[7]), value=np.nanmedian(flux_SiIII_sf[7]))
minner118 = Minimizer(objective118, params_SiIII_118, fcn_args=(rest_wavelen_SiIII_sf[7], flux_SiIII_sf[7]))
result_SiIII118 = minner118.minimize()
final_SiIII_118 = flux_SiIII_sf[7] + result_SiIII118.residual
n0_final_SiIII_118 = final_SiIII_118[final_SiIII_118 != 0]
indexs = np.array(np.argwhere(n0_final_SiIII_118 != 0))
Index = np.concatenate(indexs)
rest_wavelen_SiIII_sf[7] = rest_wavelen_SiIII_sf[7][Index]
n0_x = rest_wavelen_SiIII_sf[7][rest_wavelen_SiIII_sf[7] != 0]
n0_final_SiIII_118 = n0_final_SiIII_118[0:len(n0_x)]
n0_noise_SiIII_118 = noise_SiIII_sf[7][0:len(n0_x)]
ModelResult118 = ModelResult(Gauss_model, params_SiIII_118, weights=True, nan_policy='propagate')
n0_final_SiIII_int_flux_118 = n0_final_SiIII_118[n0_final_SiIII_118 > 2.91]
index_int_flux = np.array(np.argwhere(n0_final_SiIII_118 > 2.91))
Index_int_flux = np.concatenate(index_int_flux)
n0_x_int_flux = n0_x[Index_int_flux]
Continuum_118 = result_SiIII118.params['continuum_118'].value
integrate_118 = simps(n0_final_SiIII_int_flux_118 - Continuum_118, n0_x_int_flux)
print('INTEGRATED FLUX OF SiIII] of 118 is: ', integrate_118)
Sigma_118 = result_SiIII118.params['sigma_118'].value
FWHM_SiIII_Gauss_118 = 2 * np.sqrt(2*np.log(2)) * Sigma_118
print('THE SiIII] FWHM FOR TARGET 118 IS: ', FWHM_SiIII_Gauss_118)
dlambda_SiIII_118 = n0_x_int_flux[-1] - n0_x_int_flux[0]
EW_SiIII_118 = integrate_118 / (Continuum_118 * (1 + z[7]))
print('THE Si III] EW FOR TARGET 118 IS: ', EW_SiIII_118)
figure = plt.figure(figNr)
plt.step(rest_wavelen_SiIII_sf[7], flux_SiIII_sf[7], 'b', label='flux')
plt.step(rest_wavelen_SiIII_sf[7], noise_SiIII_sf[7], 'k', label='noise')
plt.plot(rest_wavelen_SiIII_sf[7], final_SiIII_118, 'r', label='fit')
plt.axhline(y=Continuum_118, color='r')
plt.axvline(x=SiIII_sf, color='c')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Si-III] rest-peak of target 118')
plt.savefig('plots/ALL_PART5/SiIII_sf_Gauss_target_118.pdf')
plt.savefig('MAIN_LATEX/PLOTS/SiIII_Gauss_target_118.pdf')
plt.clf()
figNr += 1


# END
########################
print('finished part 5')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))

# RESULTS:
########################
'''
The C-IV SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, 23124, 53, 218, 54891, 7876 are:  
        [0, 3.508653, 5.809946, 0, 0, 0, 0, 7.2703605, 0, 4.352294, 0, 2.2833753, 0, 0]
The C-III] SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, 23124, 53, 218, 54891, 7876:  
        [3.4629018, 0, 5.9962206, 0, 0, 0, 6.8833375, 6.0870404, 0, 0, 2.541053, 4.3899655, 0, 3.726547]
The O-III] SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, 23124, 53, 218, 54891, 7876:  
        [2.509681, 0, 3.6446757, 0, 2.896738, 0, 6.321117, 8.924425, 0, 0, 2.9740508, 4.391429, 0, 3.7683911]
The Si-III] SNR for targets 88, 204, 435, 22429, 5199, 48, 131, 118, 538, 23124, 53, 218, 54891, 7876:  
        [0, 0, 2.9633286, 2.979976, 0, 0, 4.646365, 5.8141975, 0, 0, 0, 0, 3.3325238, 0]
#############################################################################################################
INTEGRATED FLUX OF CIII] of 88 is:  3.0738483706954867
THE CIII] FWHM FOR TARGET 88 IS:  0.2246756215362471
THE CIII] EW FOR TARGET 88 IS:  0.12738843296580868
INTEGRATED FLUX OF CIII] of 435 is:  5.488889978241787
THE CIII] FWHM FOR TARGET 435 IS:  0.6551067339059153
THE CIII] EW FOR TARGET 435 IS:  0.8695840677041756
INTEGRATED FLUX OF CIII] of 131 is:  7.81263520749053
THE CIII] FWHM FOR TARGET 131 IS:  1.2392770887128595
THE CIII] EW FOR TARGET 131 IS:  0.7165053045855062
INTEGRATED FLUX OF CIII] of 118 is:  6.0262175
THE CIII] FWHM FOR TARGET 118 IS:  0.8816588501777426
THE CIII] EW FOR TARGET 118 IS:  0.4104166298746188
INTEGRATED FLUX OF CIII] of 218 is:  1.4374028
THE CIII] FWHM FOR TARGET 218 IS:  0.9890709524673857
THE CIII] EW FOR TARGET 218 IS:  0.18308474500637
INTEGRATED FLUX OF CIII] of 7876 is:  7.393275350357726
THE CIII] FWHM FOR TARGET 7876 IS:  1.1774100225154747
THE CIII] EW FOR TARGET 7876 IS:  1.2217432222458742
#############################################################################################################
INTEGRATED FLUX OF CIV of 204 is:  3.0104167
THE CIV FWHM FOR TARGET 204 IS:  1.178780217887707
THE C IV EW FOR TARGET 204 IS:  0.8013008517587199
INTEGRATED FLUX OF CIV of 435 is:  4.0595827
THE CIV FWHM FOR TARGET 435 IS:  0.9345906544072262
THE C IV EW FOR TARGET 435 IS:  0.5947029707069094
INTEGRATED FLUX OF CIV of 118 is:  3.5508342675457243
THE CIV FWHM FOR TARGET 118 IS:  0.8226736572500797
THE C IV EW FOR TARGET 118 IS:  0.22610377276604154
INTEGRATED FLUX OF CIV of 23124 is:  2.0157748520796304
THE CIV FWHM FOR TARGET 23124 IS:  0.8506694628799198
THE C IV EW FOR TARGET 23124 IS:  0.16135767415372462
INTEGRATED FLUX OF CIV of 218 is:  1.5835547
THE CIV FWHM FOR TARGET 218 IS:  0.7077965240658466
THE C IV EW FOR TARGET 218 IS:  0.18698445666116614
#############################################################################################################
INTEGRATED FLUX OF OIII] of 88 is:  2.6109940384340007
THE OIII FWHM FOR TARGET 88 IS:  0.8137471333084931
THE O III] EW FOR TARGET 88 IS:  0.1063993371210082
INTEGRATED FLUX OF OIII] of 435 is:  6.221867171210761
THE OIII FWHM FOR TARGET 435 IS:  0.9739876243307101
THE O III] EW FOR TARGET 435 IS:  1.2573044693217317
INTEGRATED FLUX OF OIII] of 5199 is:  3.2920132
THE OIII FWHM FOR TARGET 5199 IS:  1.5617558987894153
THE O III] EW FOR TARGET 5199 IS:  1.2858283794072023
INTEGRATED FLUX OF OIII] of 131 is:  5.373997450515162
THE OIII FWHM FOR TARGET 131 IS:  1.2620986367287665
THE O III] EW FOR TARGET 131 IS:  0.48548608533373977
INTEGRATED FLUX OF OIII] of 118 is:  6.5610447
THE OIII FWHM FOR TARGET 118 IS:  1.5422648194798183
THE O III] EW FOR TARGET 118 IS:  0.42233924961630215
INTEGRATED FLUX OF OIII] of 218 is:  4.6271873
THE OIII FWHM FOR TARGET 218 IS:  2.3768462802757977
THE O III] EW FOR TARGET 218 IS:  0.5697543203238464
INTEGRATED FLUX OF OIII] of 7876 is:  3.168704188094125
THE OIII FWHM FOR TARGET 7876 IS:  1.93863910939647
THE O III] EW FOR TARGET 7876 IS:  0.3957549083486352
#############################################################################################################
INTEGRATED FLUX OF SiIII] of 22429 is:  1.3908504
THE SiIII] FWHM FOR TARGET 22429 IS:  0.6702325693817042
THE Si III] EW FOR TARGET 22429 IS:  0.3303653594568976
INTEGRATED FLUX OF SiIII] of 131 is:  3.9384393320360687
THE SiIII] FWHM FOR TARGET 131 IS:  1.264816834643804
THE Si III] EW FOR TARGET 131 IS:  0.43325103244557617
INTEGRATED FLUX OF SiIII] of 118 is:  5.248531
THE SiIII] FWHM FOR TARGET 118 IS:  1.5187046783401965
THE Si III] EW FOR TARGET 118 IS:  0.45115052857890325
#############################################################################################################
finished part 5
0:00:08.797000

Process finished with exit code 0

'''
