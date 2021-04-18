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
################################  P A R T    1:  SET UP, SAVE SPECTRA IN FITS-FILES, Lya-/He-II PLOTS    #################
# DON'T CHANGE THESE SETTINGS:
FILENAME_AIR_VACUUM = 'VACCUM_AIR_conversion.fits' # LIBRARY FOR WAVELENGTH CONVERSION
pixscale = 0.20
FILENAME = 'DATACUBE_MXDF_ZAP_COR.fits' # 3D DATACUBE
radius = 0.4/pixscale # in arcsec
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
CAT = 'MXDF_BETA_CATALOG.fits' # 724 targets
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
redshift = []
COLS = []
COLS.append(pyfits.Column(name='vacuum_wavelength', unit='Angstrom', format='E', array=wav_vac))

min_wavelength = 4700 # wavelength interval: 4700-9350 A
max_wavelength = 9350 # corresponding redshift interval:
z_min = 2.86 # 	for Lya: ~ 2.86 - 6.7  		 for HeII: ~ 1.86- 4.7
z_max = 3.0 #	combined interval: 2.86 - 4.7
FigNr = 0
for i in range(0, len(RA)): # TODO: potential targets: 204, 435, 88
	if (REDSHIFT[i] >= z_min) and (REDSHIFT[i] <= z_max):
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
		flux_A[i] = np.array(flux_A[i])
		noise_pipe[i] = np.array(noise_pipe[i])
		target_identification.append(iden[i]) # reduces the array of identification nrs to the potential ones
		redshift.append(REDSHIFT[i]) # reduces the array of REDSHIFTS to the potential ones

		# SAVE SPECTRA INTO A FITS TABLE
		target_nr = str(iden[i])
		COLS.append(pyfits.Column(name='rest_vac_wavelen_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelength[i]))
		COLS.append(pyfits.Column(name='fluxdensity_total_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_A[i]))
		COLS.append(pyfits.Column(name='fluxdensity_total_ERR_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_pipe[i]))

		# ∀ GALAXIES, MAKE AN INDIVIDUAL SPECTRAL PLOT, zoom in to the Lya interval:

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
		#plt.savefig('ALL_PART1/Lya_rest_spectrum_target_%s.pdf' % (target_nr))
		plt.savefig('plots/ALL_PART1/Lya_rest_spectrum_target_%s.pdf' % (target_nr))
		plt.clf()
		FigNr += 1

		# He-II subplot at ~164.0nm
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
		#plt.savefig('ALL_PART1/HeII_rest_spectrum_target_%s.pdf' % (target_nr))
		plt.savefig('plots/ALL_PART1/HeII_rest_spectrum_target_%s.pdf' % (target_nr))
		plt.clf()
		FigNr += 1

COLS.append(pyfits.Column(name='z', format='E', array=redshift))
COLS.append(pyfits.Column(name='iden', format='E', array=target_identification))
cols = pyfits.ColDefs(COLS)
hdu = pyfits.BinTableHDU.from_columns(cols)
hdu.writeto('1D_SPECTRUM_ALL_%s.fits' % (NAME), overwrite=True)
print('finished part 1')

##########################################################################################################################
########## P A R T    2  : SELECTION OF TARGET GALAXIES ACCORDING TO He-II PEAK GAUSSIAN FIT  / Bootstrap ################
NAME = 'EXTRACTION'
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
# vac_wavelength = DATA.field('vacuum_wavelength')
REDSHIFT = DATA.field('z')
REDSHIFT = REDSHIFT[REDSHIFT != 0] # delets all "0" in the array
t = DATA.field('iden')
t = t[t != 0] # delets all "0" in the array
target_identification = []
for j in range(len(t)):
	target_identification.append(str(int(t[j]))) # convert the exponential form to strings
select_targets = len(target_identification) # 318 targets for 2.86 <= z <= 4.7
z_min = 2.86
z_max = 3.0
Ly_alpha_rest = 1215.67
HeII = 1640.42

# CHOOSE GALAXIES THAT SHOW A He-II PEAK:
rest_vac_wavelen = [[] for j in range(select_targets)]
flux_dens_tot = [[] for j in range(select_targets)]
flux_dens_err = [[] for j in range(select_targets)]
REDSHIFT_select = [item for item in REDSHIFT if item <= z_min and item >= z_max]
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

	Lya_indices.append(np.where((rest_vac_wavelen[i] > 1208) & (rest_vac_wavelen[i] < 1222)))
	flux_Lya[i] = np.array(flux_dens_tot[i])[Lya_indices[i]]
	noise_Lya[i] = np.array(flux_dens_err[i])[Lya_indices[i]]
	rest_wavelen_Lya[i] = np.array(rest_vac_wavelen[i])[Lya_indices[i]]

	HeII_indices.append(np.where((rest_vac_wavelen[i] > 1633) & (rest_vac_wavelen[i] < 1647)))
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
	'''
	# BOOTSTRAP EXPERIMENT:
	for n in range(N):
		bootstrap_flux_HeII[i] = np.random.choice(flux_HeII[i], size=len(flux_HeII[i]))
		bootstrap_err_HeII[i] = np.random.choice(noise_HeII[i], size=len(noise_HeII[i]))
		bootstrap_flux_Lya[i] = np.random.choice(flux_Lya[i], size=len(flux_Lya[i]))
		bootstrap_err_Lya[i] = np.random.choice(noise_Lya[i], size=len(noise_Lya[i]))
		av_flux_estim_HeII[i].append(np.mean(bootstrap_flux_HeII[i])) # saves average for each target for each bootstrap
		av_err_estim_HeII[i].append(np.mean(bootstrap_err_HeII[i]))
		av_flux_estim_Lya[i].append(np.mean(bootstrap_flux_Lya[i]))
		av_err_estim_Lya[i].append(np.mean(bootstrap_err_Lya[i]))
	tot_av_flux_HeII[i] = np.mean(av_flux_estim_HeII[i]) # saves average for each target
	tot_av_err_HeII[i] = np.mean(av_err_estim_HeII[i])
	tot_av_flux_Lya[i] = np.mean(av_flux_estim_Lya[i])
	tot_av_err_Lya[i] = np.mean(av_err_estim_Lya[i])
			# above tot_av_... for SNR
	'''

	# GAUSSIAN FIT:
	def Gauss(x, amp, cen, sigma):
		return amp *(1/(sigma * (np.sqrt(2*np.pi)))) * np.exp(-(x - cen) ** 2 / (2. * sigma ** 2))

	#def Gauss_dataset(params, m, x): # params==params_...;  m==target_identification; x==rest_wavelen_...
		# Calculate Gaussian lineshape from parameters for data set.
	#	amp = params['amp_%s' % (str(m))]
	#	cen = params['cen_%s' % (str(m))]
	#	sigma = params['sigma_%s' % (str(m))]
	#	return Gauss(x, amp, cen, sigma)

	#def objective(params, x, data): # data==flux_...
		# Calculate total residual for fits of Gaussians to several data sets.
	#	ndata = len(data) # ndata, _ = len(data)
	#	resid = np.zeros(len(data))
		# make residual per data set
	#	for b in range(ndata):
	#		resid[b] = data[b] - Gauss_dataset(params, target_identification[b], x)
		# now flatten this to a 1D array, as minimize() needs
	#	return resid.flatten()

	def objective2(params, x, data):
		amp = params['amp_%s' % (str(target_identification[i]))]
		cen = params['cen_%s' % (str(target_identification[i]))]
		sigma = params['sigma_%s' % (str(target_identification[i]))]
		model = Gauss(x, amp, cen, sigma)
		return model - data


	gmodel = Model(Gauss, nan_policy='propagate')
	# params = gmodel.make_params()
	# He-II GAUSSIAN FIT:
	mean_HeII = sum(rest_wavelen_HeII[i] * flux_HeII[i]) / sum(flux_HeII[i]) # np.mean(flux_HeII[i])
	sigma_HeII = np.sqrt(sum(flux_HeII[i] * (rest_wavelen_HeII[i] - mean_HeII) ** 2) / sum(flux_HeII[i]))
	popt, pcov = curve_fit(Gauss, rest_wavelen_HeII[i], flux_HeII[i], p0=[max(flux_HeII[i]), mean_HeII, sigma_HeII], maxfev=1000000000)
	perr_gauss = np.sqrt(np.diag(pcov))
	residual_HeII[i] = flux_HeII[i] - (Gauss(rest_wavelen_HeII[i], *popt))
	index = np.argmax(flux_HeII[i])  # position==index of max. element in flux_HeII[i]
	#result_HeII = gmodel.fit(flux_HeII[i], x=rest_wavelen_HeII[i], amp=max(flux_HeII[i]), cen=rest_wavelen_HeII[i][index], sigma=sigma_HeII)
	Delta_v_HeII = const.speed_of_light * (mean_HeII - HeII) / (1000 * HeII) # velocity offset ([km/s]) between HeII = 164.0nm and detected maximum; c in m/s
	if ((0 <= np.mean(flux_HeII[i])) and (Delta_v_HeII >= -800) and (Delta_v_HeII <= 500)):  # TODO: MAY CHANGE BOUDNARIES (-800, 500)
		params_He = Parameters()  # gmodel.make_params()
		params_He.add('amp_%s' % (target_nr), value=6, min=0.0, max=25)
		params_He.add('cen_%s' % (target_nr), value=1640.42, min=1630.42, max=1650.42)
		params_He.add('sigma_%s' % (target_nr), value=3, min=0.01, max=15)
		x = rest_wavelen_HeII[i]
		data = flux_HeII[i]
		minner = Minimizer(objective2, params_He, fcn_args=(x, data))
		result_HeII = minner.minimize()
		final_HeII = flux_HeII[i] + result_HeII.residual
		# report_fit(result_HeII)
		# out_He = minimize(objective, params_He, args=(rest_wavelen_HeII[i], flux_HeII[i]), nan_policy='propagate')
		# fit_report(out_He.params_He)
		figure = plt.figure(figNr)
		# y_fit_He = Gauss_dataset(out_He.params_He, target_identification[i], rest_wavelen_HeII[i])
		# plt.plot(rest_wavelen_HeII[i], flux_HeII[i], 'o', rest_wavelen_HeII[i], y_fit_He, '-')
		plt.step(rest_wavelen_HeII[i], flux_HeII[i], 'b', label='flux')
		plt.step(rest_wavelen_HeII[i], noise_HeII[i], 'k', label='noise')
		plt.plot(rest_wavelen_HeII[i], final_HeII, 'r')
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		plt.axvline(x=HeII, color='c')
		plt.grid(True)
		plt.legend(loc='best')
		#plt.title('He-II rest-peak of target %s at z = %s' % (target_nr, REDSHIFT_select[i]))
		plt.savefig('plots/ALL_PART2_lmfit/HeII_Gauss_target_%s.pdf' % (target_nr))
		plt.clf()
		figNr += 1
		'''
		figure = plt.figure(figNr)
		frame1 = figure.add_axes((0.1, 0.3, 0.8, 0.6))
		#plt.plot(rest_wavelen_HeII[i], Gauss(rest_wavelen_HeII[i], *popt), 'r-', label='fit')
		plt.plot(rest_wavelen_HeII[i], result_HeII.best_fit, 'm-', label='best fit')
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
		#plt.legend(loc='best')
		plt.grid(True)
		frame1.get_shared_x_axes().join(frame1, frame2)
		frame1.set_xticklabels([])
		#plt.savefig('ALL_PART2_GAUSSIAN/HeII_Gauss_target_%s.pdf' % (target_nr))  # TODO: run on server
		plt.savefig('plots/ALL_PART2_GAUSSIAN/HeII_Gauss_target_%s.pdf' % (target_nr)) # TODO: run here
		plt.clf()
		figNr += 1 '''

		# Lya GAUSSIAN FIT:
		params_Lya = Parameters() # gmodel.make_params()
		params_Lya.add('amp_%s' % (target_nr), value=20, min=0.0, max=200)
		params_Lya.add('cen_%s' % (target_nr), value=1215.67, min=1210.67, max=1220.67)
		params_Lya.add('sigma_%s' % (target_nr), value=3, min=0.1, max=10.0)
		# out_Lya = minimize(objective, params_Lya, args=(rest_wavelen_Lya[i], flux_Lya[i]), nan_policy='propagate')
		# fit_report(out_Lya.params_Lya)
		x = rest_wavelen_Lya[i]
		data = flux_Lya[i]
		minner = Minimizer(objective2, params_Lya, fcn_args=(x, data))
		result_Lya = minner.minimize()
		final_Lya = flux_Lya[i] + result_Lya.residual
		# report_fit(result_Lya)
		# y_fit_Lya = Gauss_dataset(out_Lya.params_Lya, target_identification[i], rest_wavelen_Lya[i])
		figure = plt.figure(figNr)
		#plt.plot(rest_wavelen_Lya[i], flux_Lya[i], 'o', rest_wavelen_Lya[i], y_fit_Lya, '-')
		plt.step(rest_wavelen_Lya[i], flux_Lya[i], 'b', label='flux')
		plt.step(rest_wavelen_Lya[i], noise_Lya[i], 'k', label='noise')
		plt.plot(rest_wavelen_Lya[i], final_Lya, 'r')
		plt.axvline(x=Ly_alpha_rest, color='c')
		plt.grid(True)
		plt.legend(loc='best')
		plt.xlabel(r'wavelength in rest-frame $[\AA]$')
		plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
		# plt.title('Lya rest-peak of target %s at z = %s' % (target_nr, REDSHIFT_select[i])) # TODO: IndexError: list index out of range ?????
		plt.savefig('plots/ALL_PART2_lmfit/Lya_Gauss_target_%s.pdf' % (target_nr))
		plt.clf()
		figNr += 1

		mean_Lya = sum(rest_wavelen_Lya[i] * flux_Lya[i]) / sum(flux_Lya[i]) # np.mean(flux_Lya[i])
		sigma_Lya = np.sqrt(sum(flux_Lya[i] * (rest_wavelen_Lya[i] - mean_Lya) ** 2) / sum(flux_Lya[i]))
		popt, pcov = curve_fit(Gauss, rest_wavelen_Lya[i], flux_Lya[i], p0=[max(flux_Lya[i]), mean_Lya, sigma_Lya], maxfev=1000000000)
		perr_gauss = np.sqrt(np.diag(pcov))
		residual_Lya[i] = flux_Lya[i] - (Gauss(rest_wavelen_Lya[i], *popt))
		index = np.argmax(flux_Lya[i])
		result_Lya = gmodel.fit(flux_Lya[i], x=rest_wavelen_Lya[i], amp=max(flux_Lya[i]), cen=rest_wavelen_Lya[i][index], sigma=sigma_Lya)
		Delta_v_Lya = const.speed_of_light * (mean_Lya - Ly_alpha_rest) / (Ly_alpha_rest * 1000)
		'''		 
		figure = plt.figure(figNr)
		frame1 = figure.add_axes((0.1, 0.3, 0.8, 0.6))
		#plt.plot(rest_wavelen_Lya[i], Gauss(rest_wavelen_Lya[i], *popt), 'r-', label='fit')
		plt.plot(rest_wavelen_Lya[i], result_Lya.best_fit, 'm-', label='best fit')
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
		#plt.legend(loc='best')
		plt.grid(True)
		frame1.get_shared_x_axes().join(frame1, frame2)
		frame1.set_xticklabels([])
		#plt.savefig('ALL_PART2_GAUSSIAN/Lya_Gauss_target_%s.pdf' % (target_nr)) # TODO: run on server
		plt.savefig('plots/ALL_PART2_GAUSSIAN/Lya_Gauss_target_%s.pdf' % (target_nr)) # TODO: run here
		plt.clf()
		figNr += 1  '''

ColsLya = pyfits.ColDefs(columnsLya)
ColsHe = pyfits.ColDefs(columnsHe)
hduLya = pyfits.BinTableHDU.from_columns(ColsLya)
hduHe = pyfits.BinTableHDU.from_columns(ColsHe)
hduLya.writeto('Lya_selected_ALL_%s.fits' % (NAME), overwrite=True)
hduHe.writeto('HeII_selected_ALL_%s.fits' % (NAME), overwrite=True)
print('finished part 2')
##########################################################################################################################
################################  P A R T    3 : SNR   ###################################################################
# SNR = S / N with S the flux and N the nose from the boostrap

print('finished part 3')

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
