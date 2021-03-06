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
import pandas as pd
import seaborn as sns
import time
from datetime import timedelta
import os

#niceValue = os.nice(19)	#  nice: value in [-20 important, 19 unimportant]
start_time = time.monotonic()

################################  P A R T    1:  SET UP, SAVE SPECTRA IN FITS-FILES, Lya-/He-II PLOTS    ####################################
# DON'T CHANGE THESE SETTINGS:
FILENAME_AIR_VACUUM = 'VACCUM_AIR_conversion.fits' # LIBRARY FOR WAVELENGTH CONVERSION
pixscale = 0.20
FILENAME = 'DATACUBE_MXDF_ZAP_COR.fits' # 3D DATACUBE
radius = 0.4/pixscale # arcsec; you can vary the aperture radius
NAME = 'EXTRACTION'

# WAVELENGTHS IN REST FRAME IN ANGSTROM
Ly_alpha_rest = 1215.67
HeII = 1640.42
CIII_f = 1906.68
CIII_sf = 1908.73
C_IV = 1548.19
O_III_sf_1 = 1660.81
O_III_sf_2 = 1666.15
O_III_f = 2321.66

# READ IN A FITS CATALOG and extract data:
CAT = 'MXDF_BETA_CATALOG.fits' # contains 724 targets
cat = pyfits.open(CAT)
data = cat[1].data
RA = data.field('ra')
DEC = data.field('dec')
REDSHIFT = data.field('z')
iden = data.field('iden')

# GET THE PIXEL POSITIONS OF THE TARGET GALAXY:
hdu = pyfits.open(FILENAME)[1]
wcs = WCS(hdu.header)
wcs = wcs.dropaxis(2)
xcenter, ycenter = wcs.wcs_world2pix(RA, DEC, 1)
# print('Pixel center of this object:', xcenter, ycenter)
data = pyfits.getdata(FILENAME, 1)
hd = pyfits.getheader(FILENAME, 1)
data_err = pyfits.getdata(FILENAME, 2)

# SET UP THE WAVELENGTH ARRAY:
wavelength = np.arange(float(len(data[:, :, 0])))
wavelength = wavelength*hd['CD3_3']+hd['CRVAL3']
catalog = pyfits.open(FILENAME_AIR_VACUUM)
dat = catalog[1].data
lamb_VACCUM = dat.field('VACCUM') # ANGSTROM
lamb_AIR = dat.field('AIR') # ANGSTROM
f = interp1d(lamb_AIR, lamb_VACCUM)
wav_vac = f(wavelength) # VACUUM WAVELENGTH ARRAY

# LOOPS AND MEASURE THE FLUX
flux_A = [[] for j in range(0, len(RA))]
noise_pipe = [[] for j in range(0, len(RA))]
x = [[] for j in range(0, len(RA))]
y = [[] for j in range(0, len(RA))]
circle_mask = [[] for j in range(0, len(RA))]
totflux = [[] for j in range(0, len(RA))]
errflux = [[] for j in range(0, len(RA))]
rest_wavelength = [[] for j in range(0, len(RA))]
target_identification = []
COLS = []
COLS.append(pyfits.Column(name='vacuum_wavelength', unit='Angstrom', format='E', array=wav_vac))

FigNr = 0
for i in range(0, len(RA)):		# loop through the number of target galaxies
	if (REDSHIFT[i] >= 3) and (REDSHIFT[i] <= 4): # select galaxies according to redshift interval; # TODO: change to [2, 5]
		for q in range(len(wavelength)):
			data_slice = data[q, :, :]  # LAYER for this wavelength
			data_slice_err = data_err[q, :, :]  # VARIANCE
			ivar = 1./data_slice_err

			# CREATE A CIRCULAR MASK AT THE POSITION
			X = len(data_slice[0, :])
			Y = len(data_slice[:, 0])
			y[i], x[i] = np.ogrid[-ycenter[i]:Y-ycenter[i], -xcenter[i]:X-xcenter[i]] # COORDINATE SYSTEM
			circle_mask[i] = x[i] * x[i] + y[i] * y[i] <= radius**2  # SELECT THE CIRCLE
			totflux[i] = np.nansum(data_slice[circle_mask[i]]) # SUMS THE FLUX
			errflux[i] = 1.27 * np.nansum(data_slice_err[circle_mask[i]])**0.5   # 1.27: accounts for correlated noise

			flux_A[i].append(totflux[i])
			noise_pipe[i].append(errflux[i])

		rest_wavelength[i] = np.array(wavelength) / (1 + REDSHIFT[i])	# TO REST-FRAME SHIFTED SPECTRA
		flux_A[i] = np.array(flux_A[i])  # flux_fraction
		noise_pipe[i] = np.array(noise_pipe[i])
		target_identification.append(iden[i]) # reduces the array of identification nrs to the potential ones

		# SAVE SPECTRA INTO A FITS TABLE
		target_nr = str(iden[i])
		COLS.append(pyfits.Column(name='rest_vac_wavelen_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelength[i]))
		COLS.append(pyfits.Column(name='fluxdensity_total_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_A[i]))
		COLS.append(pyfits.Column(name='fluxdensity_total_ERR_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_pipe[i]))

		# ??? GALAXIES, MAKE AN INDIVIDUAL SPECTRAL PLOT, zoom in to the Lya interval:

		# Lyman-alpha subplot at ~121.5nm
		plt.figure(FigNr)
		plt.step(rest_wavelength[i], flux_A[i], 'b', label='flux')
		plt.step(rest_wavelength[i], noise_pipe[i], 'k', label='noise')
		plt.axvline(x=Ly_alpha_rest, color='c')
		plt.xlim(1180, 1250)
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.title('Ly-$\u03B1$ rest-peak of target %s at z = %s' % (target_nr, REDSHIFT[i]))
		plt.grid(True)
		plt.legend(loc=1)
		plt.savefig('plots/ALL_PART1/Lya_rest_spectrum_target_%s.pdf' % (target_nr))
		plt.clf()

		FigNr += 1

		# He-II subplot at ~164.0nm
		# fit the peak with a gaussian ==> see  PART 2
		plt.figure(FigNr)
		plt.step(rest_wavelength[i], flux_A[i], 'b', label='flux')
		plt.step(rest_wavelength[i], noise_pipe[i], 'k', label='noise')
		plt.axvline(x=HeII, color='c')
		plt.xlim(1610, 1670)
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.title('He-II rest-peak of target %s at z = %s' % (target_nr, REDSHIFT[i]))
		plt.grid(True)
		plt.legend(loc=1)
		plt.savefig('plots/ALL_PART1/HeII_rest_spectrum_target_%s.pdf' % (target_nr))
		plt.clf()

		FigNr += 1


cols = pyfits.ColDefs(COLS)
hdu = pyfits.BinTableHDU.from_columns(cols)
# print(len(target_identification)) # expect: 171 == nr of targets if 3 <= z <= 4: OK
hdu.writeto('1D_SPECTRUM_ALL_%s.fits' % (NAME), overwrite=True) # saves 171 targets if 3 <= z <= 4

print('finished part 1')

########## P A R T    2  : SELECTION OF TARGET GALAXIES ACCORDING TO He-II PEAK GAUSSIAN FIT  / Bootstrap ################
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
# vac_wavelength = DATA.field('vacuum_wavelength')
select_targets = len(target_identification) # 171 targets if 3 <= z <= 4

# CHOOSE GALAXIES THAT SHOW A He-II PEAK:
rest_vac_wavelen = [[] for j in range(select_targets)] # extract the rest-frame-shifted wavelength-columns
flux_dens_tot = [[] for j in range(select_targets)]
flux_dens_err = [[] for j in range(select_targets)]
REDSHIFT_select = [item for item in REDSHIFT if item <= 4 and item >= 3]  # redshift of galaxies with 3 <= z <= 4
N = 10 # nr of bootstrap

Lya_indices = [] # save for each galaxy the list-indices of the wavelengths corresponding to the Lya peak
rest_wavelen_Lya = [[] for i in range(select_targets)]
flux_Lya = [[] for i in range(select_targets)]
noise_Lya = [[] for i in range(select_targets)]
residual_Lya = [[] for i in range(select_targets)]
bootstrap_flux_Lya = [[0 for m in range(N)] for i in range(select_targets)]
bootstrap_err_Lya = [[0 for m in range(N)] for i in range(select_targets)]
av_flux_estim_Lya = [[] for j in range(select_targets)]
av_err_estim_Lya = [[] for j in range(select_targets)]
tot_av_flux_Lya = [[] for j in range(select_targets)]
tot_av_err_Lya = [[] for j in range(select_targets)]

HeII_indices = [] # save here for each galaxy the list-indices of the wavelengths corresponding to the He-II peak
rest_wavelen_HeII = [[] for j in range(select_targets)]
flux_HeII = [[] for j in range(select_targets)]
noise_HeII = [[] for j in range(select_targets)]
residual_HeII = [[] for i in range(select_targets)]
bootstrap_flux_HeII = [[0 for m in range(N)] for i in range(select_targets)]
bootstrap_err_HeII = [[0 for m in range(N)] for i in range(select_targets)]
av_flux_estim_HeII = [[] for j in range(select_targets)]
av_err_estim_HeII = [[] for j in range(select_targets)]
tot_av_flux_HeII = [[] for j in range(select_targets)]
tot_av_err_HeII = [[] for j in range(select_targets)]

columnsHe = []
columnsLya = []
figNr = 0
for i in range(select_targets):
	target_nr = str(target_identification[i])
	flux_dens_tot[i] = DATA.field('fluxdensity_total_%s' % (target_nr))  # extract the total-flux-density-column for each galaxy
	flux_dens_err[i] = DATA.field('fluxdensity_total_ERR_%s' % (target_nr))  # extract the total-flux-density-error-column for each galaxy
	rest_vac_wavelen[i] = DATA.field('rest_vac_wavelen_%s' % (target_nr))  # extract the rest-wavelength-column for each galaxy

	Lya_indices.append(np.where((rest_vac_wavelen[i] > 1205) & (rest_vac_wavelen[i] < 1225)))
	flux_Lya[i] = np.array(flux_dens_tot[i])[Lya_indices[i]]
	noise_Lya[i] = np.array(flux_dens_err[i])[Lya_indices[i]]
	rest_wavelen_Lya[i] = np.array(rest_vac_wavelen[i])[Lya_indices[i]]

	HeII_indices.append(np.where((rest_vac_wavelen[i] > 1630) & (rest_vac_wavelen[i] < 1650)))
	flux_HeII[i] = np.array(flux_dens_tot[i])[HeII_indices[i]]
	noise_HeII[i] = np.array(flux_dens_err[i])[HeII_indices[i]]
	rest_wavelen_HeII[i] = np.array(rest_vac_wavelen[i])[HeII_indices[i]]

	# SAVE THE Lya- AND HeII-PEAK INTO A FITS TABLE
	# I M P O R T A N T : FITS CAN ONLY HAVE MAX 999 COLUMNS !!!
	columnsLya.append(pyfits.Column(name='rest_wavelen_Lya_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_Lya[i]))
	columnsLya.append(pyfits.Column(name='flux_Lya_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_Lya[i]))
	columnsLya.append(pyfits.Column(name='noise_Lya_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_Lya[i]))
	columnsHe.append(pyfits.Column(name='rest_wavelen_He-II_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_HeII[i]))
	columnsHe.append(pyfits.Column(name='flux_He-II_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_HeII[i]))
	columnsHe.append(pyfits.Column(name='noise_He-II_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_HeII[i]))

	# BOOTSTRAP EXPERIMENT:
	for n in range(N):
		bootstrap_flux_HeII[i] = np.random.choice(flux_HeII[i], size=len(flux_HeII[i]))
		bootstrap_err_HeII[i] = np.random.choice(noise_HeII[i], size=len(noise_HeII[i]))
		bootstrap_flux_Lya[i] = np.random.choice(flux_Lya[i], size=len(flux_Lya[i]))
		bootstrap_err_Lya[i] = np.random.choice(noise_Lya[i], size=len(noise_Lya[i]))
		av_flux_estim_HeII[i].append(np.mean(bootstrap_flux_HeII[i])) # saves average for each target for each bootstrap
		av_err_estim_HeII[i].append(np.mean(bootstrap_err_HeII[i]))	# "
		av_flux_estim_Lya[i].append(np.mean(bootstrap_flux_Lya[i]))	# "
		av_err_estim_Lya[i].append(np.mean(bootstrap_err_Lya[i])) # "
	tot_av_flux_HeII[i] = np.mean(av_flux_estim_HeII[i]) # saves average for each target
	tot_av_err_HeII[i] = np.mean(av_err_estim_HeII[i]) # "
	tot_av_flux_Lya[i] = np.mean(av_flux_estim_Lya[i]) # "
	tot_av_err_Lya[i] = np.mean(av_err_estim_Lya[i]) # "
			# above tot_av_... for SNR
	# GAUSSIAN FIT:
	def Gauss(x, a, x0, sigma):
		return a *(1/(sigma * (np.sqrt(2*np.pi)))) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

	# He-II GAUSSIAN FIT:
	mean_HeII = sum(rest_wavelen_HeII[i] * flux_HeII[i]) / sum(flux_HeII[i]) # np.mean(flux_HeII[i])
	# print("mean_HeII: ", mean_HeII)
	sigma_HeII = np.sqrt(sum(flux_HeII[i] * (rest_wavelen_HeII[i] - mean_HeII) ** 2) / sum(flux_HeII[i]))
	popt, pcov = curve_fit(Gauss, rest_wavelen_HeII[i], flux_HeII[i], p0=[max(flux_HeII[i]), mean_HeII, sigma_HeII], maxfev=1000000000)
	perr_gauss = np.sqrt(np.diag(pcov))
	residual_HeII[i] = flux_HeII[i] - (Gauss(rest_wavelen_HeII[i], *popt))
	Delta_v_HeII = const.speed_of_light * (mean_HeII - HeII) / (1000 * HeII) # velocity offset ([km/s]) between HeII = 164.0nm and detected maximum; c in m/s
	# print("Delta_v_HeII: ", Delta_v_HeII)

	if ((0 <= np.mean(flux_HeII[i])) and (Delta_v_HeII >= -800) and (Delta_v_HeII <= 500)):  # TODO: MAY CHANGE BOUDNARIES
		# plt.figure(figNr)
		figure = plt.figure(figNr)
		frame1 = figure.add_axes((0.1, 0.3, 0.8, 0.6))
		plt.plot(rest_wavelen_HeII[i], Gauss(rest_wavelen_HeII[i], *popt), 'r-', label='fit')
		plt.step(rest_wavelen_HeII[i], flux_HeII[i], 'b', label='flux')
		plt.step(rest_wavelen_HeII[i], noise_HeII[i], 'k', label='noise')
		frame1.legend(loc='best')
		plt.grid(True)
		plt.axvline(x=HeII, color='c')
		frame2 = figure.add_axes((0.1, 0.1, 0.8, 0.2))
		plt.plot(rest_wavelen_HeII[i], residual_HeII[i], '.g', label='residuals')
		frame2.legend(loc='best')
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.suptitle('He-II rest-peak of target %s at z = %s' % (target_nr, REDSHIFT_select[i]))
		plt.grid(True)
		frame1.get_shared_x_axes().join(frame1, frame2)
		frame1.set_xticklabels([])
		plt.savefig('plots/ALL_PART2_GAUSSIAN/HeII_Gauss_target_%s.pdf' % (target_nr))
		plt.clf()
		figNr += 1

		# Lya GAUSSIAN FIT:
		mean_Lya = np.mean(flux_Lya[i]) # sum(rest_wavelen_Lya[i] * flux_Lya[i]) / sum(flux_Lya[i])
		sigma_Lya = np.sqrt(sum(flux_Lya[i] * (rest_wavelen_Lya[i] - mean_Lya) ** 2) / sum(flux_Lya[i]))
		popt, pcov = curve_fit(Gauss, rest_wavelen_Lya[i], flux_Lya[i], p0=[max(flux_Lya[i]), mean_Lya, sigma_Lya], maxfev=1000000000)
		perr_gauss = np.sqrt(np.diag(pcov))
		residual_Lya[i] = flux_Lya[i] - (Gauss(rest_wavelen_Lya[i], *popt))

		# plt.figure(figNr)
		figure = plt.figure(figNr)
		frame1 = figure.add_axes((0.1, 0.3, 0.8, 0.6))
		plt.plot(rest_wavelen_Lya[i], Gauss(rest_wavelen_Lya[i], *popt), 'r-', label='fit')
		plt.step(rest_wavelen_Lya[i], flux_Lya[i], 'b', label='flux')
		plt.step(rest_wavelen_Lya[i], noise_Lya[i], 'k', label='noise')
		frame1.legend(loc='best')
		plt.grid(True)
		plt.axvline(x=Ly_alpha_rest, color='c')
		frame2 = figure.add_axes((0.1, 0.1, 0.8, 0.2))
		plt.plot(rest_wavelen_Lya[i], residual_Lya[i], '.g', label='residuals')
		frame2.legend(loc='best')
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.suptitle('Ly-$\u03B1$ rest-peak of target %s at z = %s' % (target_nr, REDSHIFT_select[i]))
		plt.grid(True)
		frame1.get_shared_x_axes().join(frame1, frame2) # plot and residual-plot share x-axis
		frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
		plt.savefig('plots/ALL_PART2_GAUSSIAN/Lya_Gauss_target_%s.pdf' % (target_nr))
		plt.clf()
		figNr += 1

ColsLya = pyfits.ColDefs(columnsLya)
ColsHe = pyfits.ColDefs(columnsHe)
hduLya = pyfits.BinTableHDU.from_columns(ColsLya)
hduHe = pyfits.BinTableHDU.from_columns(ColsHe)
hduLya.writeto('Lya_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduHe.writeto('HeII_selected_ALL_%s.fits' % (NAME), overwrite=True)
print('finished part 2')

################################  P A R T    3 : SNR   ####################################
# SNR = S / N with S the flux and N the nose from the boostrap

print('finished part 3')

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
