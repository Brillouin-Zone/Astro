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
########## P A R T   4  : CREATE FULL SPECTRUM PLOT OF TARGETS 88, 204, 435, 22429 FOR PAPER #############################
NAME = 'EXTRACTION'
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
# vac_wavelength = DATA.field('vacuum_wavelength')
Ly_alpha_rest = 1215.67
HeII = 1640.42
C_IV = 1548.19
O_III_sf_1 = 1660.81

flux_dens_tot_88 = DATA.field('fluxdensity_total_88')
flux_dens_tot_204 = DATA.field('fluxdensity_total_204')
flux_dens_tot_435 = DATA.field('fluxdensity_total_435')
flux_dens_tot_22429 = DATA.field('fluxdensity_total_22429')
flux_dens_err_88 = DATA.field('fluxdensity_total_ERR_88')
flux_dens_err_204 = DATA.field('fluxdensity_total_ERR_204')
flux_dens_err_435 = DATA.field('fluxdensity_total_ERR_435')
flux_dens_err_22429 = DATA.field('fluxdensity_total_ERR_22429')
rest_vac_wavelen_88 = DATA.field('rest_vac_wavelen_88')
rest_vac_wavelen_204 = DATA.field('rest_vac_wavelen_204')
rest_vac_wavelen_435 = DATA.field('rest_vac_wavelen_435')
rest_vac_wavelen_22429 = DATA.field('rest_vac_wavelen_22429')

figNr = 0
figure = plt.figure(figNr, figsize=(10, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_88, flux_dens_tot_88, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_88, flux_dens_err_88, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_1, color='c', linewidth=0.7)
plt.text(1225, 60, r'$Ly\alpha$')
plt.text(1644, 70, r'$He-II$')
plt.text(1684, 60, r'$O-III$')
plt.text(1557, 60, r'$C-IV$')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 88 at z = 2.9541607')
plt.savefig('MAIN_LATEX/target_88.pdf')
plt.savefig('plots/ALL_PART4/target_88.pdf')
plt.clf()
figNr += 1

figure = plt.figure(figNr, figsize=(10, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_204, flux_dens_tot_204, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_204, flux_dens_err_204, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_1, color='c', linewidth=0.7)
plt.text(1225, 15, r'$Ly\alpha$')
plt.text(1644, 20, r'$He-II$')
plt.text(1684, 15, r'$O-III$')
plt.text(1557, 15, r'$C-IV$')
plt.grid(True)
plt.legend(loc='upper right')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 204 at z = 3.1357558')
plt.savefig('MAIN_LATEX/target_204.pdf')
plt.savefig('plots/ALL_PART4/target_204.pdf')
plt.clf()
figNr += 1

figure = plt.figure(figNr, figsize=(10, 5))
plt.step(rest_vac_wavelen_435, flux_dens_tot_435, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_435, flux_dens_err_435, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_1, color='c', linewidth=0.7)
plt.text(1225, 60, r'$Ly\alpha$')
plt.text(1644, 70, r'$He-II$')
plt.text(1684, 60, r'$O-III$')
plt.text(1557, 60, r'$C-IV$')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 435 at z = 3.7247474')
plt.savefig('MAIN_LATEX/target_435.pdf')
plt.savefig('plots/ALL_PART4/target_435.pdf')
plt.clf()
figNr += 1

figure = plt.figure(figNr, figsize=(10, 5))
plt.step(rest_vac_wavelen_22429, flux_dens_tot_22429, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_22429, flux_dens_err_22429, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_1, color='c', linewidth=0.7)
plt.text(1225, 20, r'$Ly\alpha$')
plt.text(1644, 15, r'$He-II$')
plt.text(1684, 20, r'$O-III$')
plt.text(1557, 20, r'$C-IV$')
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 22429 at z = 2.9297342')
plt.savefig('MAIN_LATEX/target_22429.pdf')
plt.savefig('plots/ALL_PART4/target_22429.pdf')
plt.clf()
figNr += 1

# END
########################
print('finished part 4')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
