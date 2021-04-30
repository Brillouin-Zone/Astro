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
SiIII_sf_1 = 1882.71

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
z_max = 4.7 #	combined interval: 2.86 - 4.7
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
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
