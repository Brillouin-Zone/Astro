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
from lmfit.model import Model
from lmfit import Parameters, minimize, fit_report, Minimizer, report_fit
import pandas as pd
import seaborn as sns
import time
from datetime import timedelta
import os

#niceValue = os.nice(19)	#  nice: value in [-20 important, 19 unimportant]
start_time = time.monotonic()
##########################################################################################################################
########## P A R T    2  : SELECTION OF TARGET GALAXIES ACCORDING TO He-II PEAK GAUSSIAN FIT #############################
	# select and save in separate tables also the peaks for CIII], SiIII], OIII] and CIV for later
# https://lmfit.github.io/lmfit-py/
NAME = 'EXTRACTION'
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
# vac_wavelength = DATA.field('vacuum_wavelength')
REDSHIFT = DATA.field('z')
REDSHIFT = REDSHIFT[REDSHIFT != 0] # delets all "0" in the array; len(REDSHIFT) = 318 == nr of targets in '1D_SPECTRUM_ALL_%s.fits'%(NAME)
t = DATA.field('iden')
t = t[t != 0] # delets all "0" in the array
target_identification = []
for j in range(len(t)):
	target_identification.append(str(int(t[j]))) # convert the exponential form to strings
select_targets = len(target_identification) # 318 targets for 2.86 <= z <= 4.7
z_min = 2.86
z_max = 4.7
Ly_alpha_rest = 1215.67
HeII = 1640.42
CIII_f = 1906.68
CIII_sf = 1908.73
C_IV = 1548.19
O_III_sf_1 = 1660.81
O_III_sf_2 = 1666.15
O_III_f = 2321.66
SiIII_sf_1 = 1882.71
SiIII_sf_2 = 1892.03

# CHOOSE GALAXIES THAT SHOW A He-II PEAK:
rest_vac_wavelen = [[] for j in range(select_targets)]
flux_dens_tot = [[] for j in range(select_targets)]
flux_dens_err = [[] for j in range(select_targets)]

Lya_indices = [] # save for each galaxy the list-indices of the wavelengths corresponding to the Lya peak
rest_wavelen_Lya = [[] for i in range(select_targets)]
flux_Lya = [[] for i in range(select_targets)]
noise_Lya = [[] for i in range(select_targets)]

HeII_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the He-II peak
rest_wavelen_HeII = [[] for j in range(select_targets)]
flux_HeII = [[] for j in range(select_targets)]
noise_HeII = [[] for j in range(select_targets)]

CIII_sf_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the CIII] peak
rest_wavelen_CIII_sf = [[] for j in range(select_targets)]
flux_CIII_sf = [[] for j in range(select_targets)]
noise_CIII_sf = [[] for j in range(select_targets)]

O_III_sf_1_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the He-II peak
rest_wavelen_O_III_sf_1 = [[] for j in range(select_targets)]
flux_O_III_sf_1 = [[] for j in range(select_targets)]
noise_O_III_sf_1 = [[] for j in range(select_targets)]

C_IV_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the He-II peak
rest_wavelen_C_IV = [[] for j in range(select_targets)]
flux_C_IV = [[] for j in range(select_targets)]
noise_C_IV = [[] for j in range(select_targets)]

SiIII_sf_1_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the He-II peak
rest_wavelen_SiIII_sf_1 = [[] for j in range(select_targets)]
flux_SiIII_sf_1 = [[] for j in range(select_targets)]
noise_SiIII_sf_1 = [[] for j in range(select_targets)]

columnsLya = []
columnsHe = []
columnsCIII = []
columnsOIII = []
columnsCIV = []
columnsSiIII = []

figNr = 0
for i in range(select_targets):
	target_nr = str(target_identification[i]) # ok
	flux_dens_tot[i] = DATA.field('fluxdensity_total_%s' % (target_nr))  # extract the total-flux-density-column for each galaxy
	flux_dens_err[i] = DATA.field('fluxdensity_total_ERR_%s' % (target_nr))  # extract the total-flux-density-error-column for each galaxy
	rest_vac_wavelen[i] = DATA.field('rest_vac_wavelen_%s' % (target_nr))  # extract the rest-wavelength-column for each galaxy

	Lya_indices.append(np.where((rest_vac_wavelen[i] > 1208) & (rest_vac_wavelen[i] < 1222)))
	flux_Lya[i] = np.array(flux_dens_tot[i])[Lya_indices[i]]
	noise_Lya[i] = np.array(flux_dens_err[i])[Lya_indices[i]]
	rest_wavelen_Lya[i] = np.array(rest_vac_wavelen[i])[Lya_indices[i]]

	HeII_indices.append(np.where((rest_vac_wavelen[i] > 1633) & (rest_vac_wavelen[i] < 1647)))
	flux_HeII[i] = np.array(flux_dens_tot[i])[HeII_indices[i]]
	noise_HeII[i] = np.array(flux_dens_err[i])[HeII_indices[i]]
	rest_wavelen_HeII[i] = np.array(rest_vac_wavelen[i])[HeII_indices[i]]

	CIII_sf_indices.append(np.where((rest_vac_wavelen[i] > 1902) & (rest_vac_wavelen[i] < 1916)))
	flux_CIII_sf[i] = np.array(flux_dens_tot[i])[CIII_sf_indices[i]]
	noise_CIII_sf[i] = np.array(flux_dens_err[i])[CIII_sf_indices[i]]
	rest_wavelen_CIII_sf[i] = np.array(rest_vac_wavelen[i])[CIII_sf_indices[i]]

	O_III_sf_1_indices.append(np.where((rest_vac_wavelen[i] > 1654) & (rest_vac_wavelen[i] < 1668)))
	flux_O_III_sf_1[i] = np.array(flux_dens_tot[i])[O_III_sf_1_indices[i]]
	noise_O_III_sf_1[i] = np.array(flux_dens_err[i])[O_III_sf_1_indices[i]]
	rest_wavelen_O_III_sf_1[i] = np.array(rest_vac_wavelen[i])[O_III_sf_1_indices[i]]

	C_IV_indices.append(np.where((rest_vac_wavelen[i] > 1541) & (rest_vac_wavelen[i] < 1555)))
	flux_C_IV[i] = np.array(flux_dens_tot[i])[C_IV_indices[i]]
	noise_C_IV[i] = np.array(flux_dens_err[i])[C_IV_indices[i]]
	rest_wavelen_C_IV[i] = np.array(rest_vac_wavelen[i])[C_IV_indices[i]]

	SiIII_sf_1_indices.append(np.where((rest_vac_wavelen[i] > 1876) & (rest_vac_wavelen[i] < 1890)))
	flux_SiIII_sf_1[i] = np.array(flux_dens_tot[i])[SiIII_sf_1_indices[i]]
	noise_SiIII_sf_1[i] = np.array(flux_dens_err[i])[SiIII_sf_1_indices[i]]
	rest_wavelen_SiIII_sf_1[i] = np.array(rest_vac_wavelen[i])[SiIII_sf_1_indices[i]]

	# SAVE THE Lya- AND HeII-PEAK INTO A FITS TABLE
	mean_HeII = sum(rest_wavelen_HeII[i] * flux_HeII[i]) / sum(flux_HeII[i]) # e.g.: 1643.4745725917699 (last entry in array)
	Delta_v_HeII = const.speed_of_light * (mean_HeII - HeII) / (1000 * HeII)  # velocity offset ([km/s]) between HeII = 164.0nm and detected maximum; c in m/s
	if ((0 <= np.mean(flux_HeII[i])) and (Delta_v_HeII >= -800) and (Delta_v_HeII <= 500)):  # TODO: MAY CHANGE BOUDNARIES
		# I M P O R T A N T : FITS CAN ONLY HAVE MAX 999 COLUMNS !!!
		columnsLya.append(pyfits.Column(name='rest_wavelen_Lya_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_Lya[i]))
		columnsLya.append(pyfits.Column(name='flux_Lya_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_Lya[i]))
		columnsLya.append(pyfits.Column(name='noise_Lya_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_Lya[i]))
		columnsHe.append(pyfits.Column(name='rest_wavelen_He-II_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_HeII[i]))
		columnsHe.append(pyfits.Column(name='flux_He-II_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_HeII[i]))
		columnsHe.append(pyfits.Column(name='noise_He-II_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_HeII[i]))
		columnsCIII.append(pyfits.Column(name='rest_wavelen_C-III_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_CIII_sf[i]))
		columnsCIII.append(pyfits.Column(name='flux_C-III_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_CIII_sf[i]))
		columnsCIII.append(pyfits.Column(name='noise_C-III_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_CIII_sf[i]))
		columnsOIII.append(pyfits.Column(name='rest_wavelen_O-III_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_O_III_sf_1[i]))
		columnsOIII.append(pyfits.Column(name='flux_O-III_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_O_III_sf_1[i]))
		columnsOIII.append(pyfits.Column(name='noise_O-III_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_O_III_sf_1[i]))
		columnsCIV.append(pyfits.Column(name='rest_wavelen_C-IV_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_C_IV[i]))
		columnsCIV.append(pyfits.Column(name='flux_C-IV_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_C_IV[i]))
		columnsCIV.append(pyfits.Column(name='noise_C-IV_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_C_IV[i]))
		columnsSiIII.append(pyfits.Column(name='rest_wavelen_Si-III_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_SiIII_sf_1[i]))
		columnsSiIII.append(pyfits.Column(name='flux_Si-III_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_SiIII_sf_1[i]))
		columnsSiIII.append(pyfits.Column(name='noise_Si-III_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_SiIII_sf_1[i]))

		# GAUSSIAN FIT:

		def Gauss(x, amp, cen, sigma):
			return amp *(1/(sigma * (np.sqrt(2*np.pi)))) * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))

		def objective2(params, x, data):
			amp = params['amp_%s' % (str(target_identification[i]))]
			cen = params['cen_%s' % (str(target_identification[i]))]
			sigma = params['sigma_%s' % (str(target_identification[i]))]
			model = Gauss(x, amp, cen, sigma)
			return model - data

		gmodel = Model(Gauss, nan_policy='propagate')
		# He-II GAUSSIAN FIT:
		params_He = Parameters()  # gmodel.make_params()
		params_He.add('amp_%s' % (target_nr), value=6, min=0.0, max=25)
		params_He.add('cen_%s' % (target_nr), value=1640.42, min=1630.42, max=1650.42)
		params_He.add('sigma_%s' % (target_nr), value=3, min=0.01, max=15)
		x = rest_wavelen_HeII[i]
		data = flux_HeII[i]
		minner = Minimizer(objective2, params_He, fcn_args=(x, data))
		result_HeII = minner.minimize()
		final_HeII = flux_HeII[i] + result_HeII.residual

		figure = plt.figure(figNr)
		#plt.figure(figNr)
		plt.step(rest_wavelen_HeII[i], flux_HeII[i], 'b', label='flux')
		plt.step(rest_wavelen_HeII[i], noise_HeII[i], 'k', label='noise')
		plt.plot(rest_wavelen_HeII[i], final_HeII, 'r')
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.axvline(x=HeII, color='c')
		plt.grid(True)
		plt.legend(loc='best')
		plt.title('He-II rest-peak of target %s at z = %s' % (target_identification[i], str(REDSHIFT[i])))
		plt.savefig('plots/ALL_PART2_lmfit/HeII_Gauss_target_%s.pdf' % (target_nr))
		plt.clf()
		figNr += 1

		# Lya GAUSSIAN FIT:
		params_Lya = Parameters() # gmodel.make_params()
		params_Lya.add('amp_%s' % (target_nr), value=20, min=0.0, max=200)
		params_Lya.add('cen_%s' % (target_nr), value=1215.67, min=1210.67, max=1220.67)
		params_Lya.add('sigma_%s' % (target_nr), value=3, min=0.1, max=10.0)
		x = rest_wavelen_Lya[i]
		data = flux_Lya[i]
		minner = Minimizer(objective2, params_Lya, fcn_args=(x, data))
		result_Lya = minner.minimize()
		final_Lya = flux_Lya[i] + result_Lya.residual

		figure = plt.figure(figNr)
		#plt.figure(figNr)
		plt.step(rest_wavelen_Lya[i], flux_Lya[i], 'b', label='flux')
		plt.step(rest_wavelen_Lya[i], noise_Lya[i], 'k', label='noise')
		plt.plot(rest_wavelen_Lya[i], final_Lya, 'r')
		plt.axvline(x=Ly_alpha_rest, color='c')
		plt.grid(True)
		plt.legend(loc='best')
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.title('Lya rest-peak of target %s at z = %s' % (target_identification[i], str(REDSHIFT[i])))
		plt.savefig('plots/ALL_PART2_lmfit/Lya_Gauss_target_%s.pdf' % (target_nr))
		plt.clf()
		figNr += 1

ColsLya = pyfits.ColDefs(columnsLya)
ColsHe = pyfits.ColDefs(columnsHe)
ColsCIII = pyfits.ColDefs(columnsCIII)
ColsOIII = pyfits.ColDefs(columnsOIII)
ColsCIV = pyfits.ColDefs(columnsCIV)
ColsSiIII = pyfits.ColDefs(columnsSiIII)

hduLya = pyfits.BinTableHDU.from_columns(ColsLya)
hduHe = pyfits.BinTableHDU.from_columns(ColsHe)
hduCIII = pyfits.BinTableHDU.from_columns(ColsCIII)
hduOIII = pyfits.BinTableHDU.from_columns(ColsOIII)
hduCIV = pyfits.BinTableHDU.from_columns(ColsCIV)
hduSiIII = pyfits.BinTableHDU.from_columns(ColsSiIII)

# each of the following tables contains 183 target-peaks when using if-constraints in line 110 (-800 < Delta_v_HeII < 500)
hduLya.writeto('Lya_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduHe.writeto('HeII_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduCIII.writeto('CIII_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduOIII.writeto('OIII_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduCIV.writeto('CIV_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduSiIII.writeto('SiIII_selected_ALL_%s.fits' % (NAME), overwrite=True)

print('finished part 2')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
