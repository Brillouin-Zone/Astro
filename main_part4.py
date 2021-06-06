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
########## P A R T   4  : CREATE FULL SPECTRUM PLOT OF TARGETS 88, 204, 435, 22429 FOR PAPER ##########################
NAME = 'EXTRACTION'
EXTRACTIONS = '1D_SPECTRUM_ALL_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
# vac_wavelength = DATA.field('vacuum_wavelength')
Ly_alpha_rest = 1215.67
HeII = 1640.42
C_IV = 1548.19
O_III_sf_1 = 1660.81
O_III_sf_2 = 1666.15
CIII_sf = 1908.73
SiIII_sf_1 = 1882.71

flux_dens_tot_48 = DATA.field('fluxdensity_total_48')
flux_dens_tot_88 = DATA.field('fluxdensity_total_88')
flux_dens_tot_118 = DATA.field('fluxdensity_total_118')
flux_dens_tot_131 = DATA.field('fluxdensity_total_131')
flux_dens_tot_204 = DATA.field('fluxdensity_total_204')
flux_dens_tot_218 = DATA.field('fluxdensity_total_218')
flux_dens_tot_435 = DATA.field('fluxdensity_total_435')
flux_dens_tot_538 = DATA.field('fluxdensity_total_538')
flux_dens_tot_5199 = DATA.field('fluxdensity_total_5199')
flux_dens_tot_7876 = DATA.field('fluxdensity_total_7876')
flux_dens_tot_23124 = DATA.field('fluxdensity_total_23124')
flux_dens_tot_22429 = DATA.field('fluxdensity_total_22429')
flux_dens_err_48 = DATA.field('fluxdensity_total_ERR_48')
flux_dens_err_88 = DATA.field('fluxdensity_total_ERR_88')
flux_dens_err_118 = DATA.field('fluxdensity_total_ERR_118')
flux_dens_err_131 = DATA.field('fluxdensity_total_ERR_131')
flux_dens_err_204 = DATA.field('fluxdensity_total_ERR_204')
flux_dens_err_218 = DATA.field('fluxdensity_total_ERR_218')
flux_dens_err_435 = DATA.field('fluxdensity_total_ERR_435')
flux_dens_err_538 = DATA.field('fluxdensity_total_ERR_538')
flux_dens_err_5199 = DATA.field('fluxdensity_total_ERR_5199')
flux_dens_err_7876 = DATA.field('fluxdensity_total_ERR_7876')
flux_dens_err_23124 = DATA.field('fluxdensity_total_ERR_23124')
flux_dens_err_22429 = DATA.field('fluxdensity_total_ERR_22429')
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

figNr = 0
# SPECTRUM OF TARGET 48
figure = plt.figure(figNr, figsize=(15, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_48, flux_dens_tot_48, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_48, flux_dens_err_48, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 15, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 15, r'$He II$', rotation=90.)
plt.text(1684, 15, r'$O III]$', rotation=90.)
plt.text(1557, 15, r'$C IV$', rotation=90.)
plt.text(1890, 15, r'$Si III]$', rotation=90.)
plt.text(1915, 15, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 48 at z = 2.9101489')
plt.savefig('MAIN_LATEX/PLOTS/target_48.pdf')
plt.savefig('plots/ALL_PART4/target_48.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 88
figure = plt.figure(figNr, figsize=(15, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_88, flux_dens_tot_88, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_88, flux_dens_err_88, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 60, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 60, r'$He II$', rotation=90.)
plt.text(1684, 60, r'$O III]$', rotation=90.)
plt.text(1557, 60, r'$C IV$', rotation=90.)
plt.text(1890, 60, r'$Si III]$', rotation=90.)
plt.text(1915, 60, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 88 at z = 2.9541607')
plt.savefig('MAIN_LATEX/PLOTS/target_88.pdf')
plt.savefig('plots/ALL_PART4/target_88.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 118
figure = plt.figure(figNr, figsize=(15, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_118, flux_dens_tot_118, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_118, flux_dens_err_118, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 60, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 60, r'$He II$', rotation=90.)
plt.text(1684, 60, r'$O III]$', rotation=90.)
plt.text(1557, 60, r'$C IV$', rotation=90.)
plt.text(1890, 60, r'$Si III]$', rotation=90.)
plt.text(1915, 60, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 118 at z = 3.0024831')
plt.savefig('MAIN_LATEX/PLOTS/target_118.pdf')
plt.savefig('plots/ALL_PART4/target_118.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 131
figure = plt.figure(figNr, figsize=(15, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_131, flux_dens_tot_131, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_131, flux_dens_err_131, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 20, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 20, r'$He II$', rotation=90.)
plt.text(1684, 20, r'$O III]$', rotation=90.)
plt.text(1557, 20, r'$C IV$', rotation=90.)
plt.text(1890, 20, r'$Si III]$', rotation=90.)
plt.text(1915, 20, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 131 at z = 3.0191996')
plt.savefig('MAIN_LATEX/PLOTS/target_131.pdf')
plt.savefig('plots/ALL_PART4/target_131.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 204
figure = plt.figure(figNr, figsize=(15, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_204, flux_dens_tot_204, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_204, flux_dens_err_204, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 15, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 15, r'$He II$', rotation=90.)
plt.text(1684, 15, r'$O III]$', rotation=90.)
plt.text(1557, 15, r'$C IV$', rotation=90.)
plt.text(1890, 15, r'$Si III]$', rotation=90.)
plt.text(1915, 15, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='upper right')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 204 at z = 3.1357558')
plt.savefig('MAIN_LATEX/PLOTS/target_204.pdf')
plt.savefig('plots/ALL_PART4/target_204.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 218
figure = plt.figure(figNr, figsize=(15, 5))
#plt.figure(figNr)
plt.step(rest_vac_wavelen_218, flux_dens_tot_218, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_218, flux_dens_err_218, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 10, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 10, r'$He II$', rotation=90.)
plt.text(1684, 10, r'$O III]$', rotation=90.)
plt.text(1557, 10, r'$C IV$', rotation=90.)
plt.text(1890, 10, r'$Si III]$', rotation=90.)
plt.text(1915, 10, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='upper right')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 218 at z = 2.865628')
plt.savefig('MAIN_LATEX/PLOTS/target_218.pdf')
plt.savefig('plots/ALL_PART4/target_218.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 435
figure = plt.figure(figNr, figsize=(15, 5))
plt.step(rest_vac_wavelen_435, flux_dens_tot_435, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_435, flux_dens_err_435, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 60, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 60, r'$He II$', rotation=90.)
plt.text(1684, 60, r'$O III]$', rotation=90.)
plt.text(1557, 60, r'$C IV$', rotation=90.)
plt.text(1890, 60, r'$Si III]$', rotation=90.)
plt.text(1915, 60, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 435 at z = 3.7247474')
plt.savefig('MAIN_LATEX/PLOTS/target_435.pdf')
plt.savefig('plots/ALL_PART4/target_435.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 538
figure = plt.figure(figNr, figsize=(15, 5))
plt.step(rest_vac_wavelen_538, flux_dens_tot_538, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_538, flux_dens_err_538, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 15, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 15, r'$He II$', rotation=90.)
plt.text(1684, 15, r'$O III]$', rotation=90.)
plt.text(1557, 15, r'$C IV$', rotation=90.)
plt.text(1890, 15, r'$Si III]$', rotation=90.)
plt.text(1915, 15, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 538 at z = 4.1764603')
plt.savefig('MAIN_LATEX/PLOTS/target_538.pdf')
plt.savefig('plots/ALL_PART4/target_538.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 5199
figure = plt.figure(figNr, figsize=(15, 5))
plt.step(rest_vac_wavelen_5199, flux_dens_tot_5199, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_5199, flux_dens_err_5199, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 5, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 5, r'$He II$', rotation=90.)
plt.text(1684, 5, r'$O III]$', rotation=90.)
plt.text(1557, 5, r'$C IV$', rotation=90.)
plt.text(1890, 5, r'$Si III]$', rotation=90.)
plt.text(1915, 5, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 5199 at z = 3.063')
plt.savefig('MAIN_LATEX/PLOTS/target_5199.pdf')
plt.savefig('plots/ALL_PART4/target_5199.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 7876
figure = plt.figure(figNr, figsize=(15, 5))
plt.step(rest_vac_wavelen_7876, flux_dens_tot_7876, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_7876, flux_dens_err_7876, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 10, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 10, r'$He II$', rotation=90.)
plt.text(1684, 10, r'$O III]$', rotation=90.)
plt.text(1557, 10, r'$C IV$', rotation=90.)
plt.text(1890, 10, r'$Si III]$', rotation=90.)
plt.text(1915, 10, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 7876 at z = 2.993115')
plt.savefig('MAIN_LATEX/PLOTS/target_7876.pdf')
plt.savefig('plots/ALL_PART4/target_7876.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 22429
figure = plt.figure(figNr, figsize=(15, 5))
plt.step(rest_vac_wavelen_22429, flux_dens_tot_22429, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_22429, flux_dens_err_22429, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 15, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 15, r'$He II$', rotation=90.)
plt.text(1684, 15, r'$O III]$', rotation=90.)
plt.text(1557, 15, r'$C IV$', rotation=90.)
plt.text(1890, 15, r'$Si III]$', rotation=90.)
plt.text(1915, 15, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 22429 at z = 2.9297342')
plt.savefig('MAIN_LATEX/PLOTS/target_22429.pdf')
plt.savefig('plots/ALL_PART4/target_22429.pdf')
plt.clf()
figNr += 1

# SPECTRUM OF TARGET 23124
figure = plt.figure(figNr, figsize=(15, 5))
plt.step(rest_vac_wavelen_23124, flux_dens_tot_23124, 'b', label='flux', linewidth=0.3)
plt.step(rest_vac_wavelen_23124, flux_dens_err_23124, 'k', label='noise', linewidth=0.3)
plt.axvline(x=Ly_alpha_rest, color='c', linewidth=0.7)
plt.axvline(x=HeII, color='c', linewidth=0.7)
plt.axvline(x=C_IV, color='c', linewidth=0.7)
plt.axvline(x=CIII_sf, color='c', linewidth=0.7)
plt.axvline(x=SiIII_sf_1, color='c', linewidth=0.7)
plt.axvline(x=O_III_sf_2, color='c', linewidth=0.7)
plt.text(1225, 20, r'$Ly\alpha$', rotation=90.)
plt.text(1644, 20, r'$He II$', rotation=90.)
plt.text(1684, 20, r'$O III]$', rotation=90.)
plt.text(1557, 20, r'$C IV$', rotation=90.)
plt.text(1890, 20, r'$Si III]$', rotation=90.)
plt.text(1915, 20, r'$C III]$', rotation=90.)
plt.grid(True)
plt.legend(loc='best')
plt.xlabel(r'wavelength in rest-frame $[\AA]$')
plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
plt.title('Lya rest-peak of target 23124 at z = 3.5935392')
plt.savefig('MAIN_LATEX/PLOTS/target_23124.pdf')
plt.savefig('plots/ALL_PART4/target_23124.pdf')
plt.clf()
figNr += 1

# END
########################
print('finished part 4')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
