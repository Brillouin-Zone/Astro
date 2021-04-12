import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize as sci
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from decimal import Decimal
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
import scipy.constants as const
from lmfit.model import Model

import time
from datetime import timedelta

################################  P A R T    1:  SET UP, SAVE SPECTRA IN FITS-FILES, Lya-/He-II PLOTS    ####################################

# DON'T CHANGE THESE SETTINGS:
start_time = time.monotonic()
FILENAME_AIR_VACUUM = 'VACCUM_AIR_conversion.fits' # THIS IS THE FILENAME OF A LIBRARY FOR WAVELENGTH CONVERSION
pixscale = 0.20  # Don't change this
FILENAME = 'DATACUBE_MXDF_ZAP_COR.fits' # THIS IS THE FILENAME of the 3D DATACUBE


# WAVELENGTHS IN REST FRAME IN ANGSTROM
Ly_alpha_rest = 1215.67
HeII = 1640.42
CIII_f = 1906.68
CIII_sf = 1908.73
C_IV = 1548.19
O_III_sf_1 = 1660.81
O_III_sf_2 = 1666.15
O_III_f = 2321.66

# YOU CAN CHANGE THESE SETTINGS:
NAME = 'TEST_EXTRACTION' # This name is used for saving
radius = 0.4/pixscale # arcsec; you can vary the aperture radius
RA = np.array([0, 0, 0], dtype=float)
DEC = np.array([0, 0, 0], dtype=float)

# TARGETS:
RA[0] = 53.1628441950602 # This is the Right Ascension of the target
DEC[0] = -27.7765927418624 # This is the Declination of the target

RA[1] = 53.1682775257735
DEC[1] = -27.7810365663019

RA[2] = 53.1580101003638
DEC[2] = -27.7870931133547
######################
z = [3.189, 3.0, 3.192]

# THE CODE STARTS HERE:
# GET THE PIXEL POSITIONS OF THE TARGET GALAXY:
hdu = pyfits.open(FILENAME)[1]
wcs = WCS(hdu.header)
wcs = wcs.dropaxis(2)
xcenter, ycenter = wcs.wcs_world2pix(RA, DEC, 1)
print('Pixel center of this object:', xcenter, ycenter)
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
wav_vac = f(wavelength) # THIS IS THE VACUUM WAVELENGTH ARRAY


# LOOP THROUGH THE WAVELENGTH LAYERS AND MEASURE THE FLUX
flux_A = [[] for i in range(len(RA))]
noise_pipe = [[] for i in range(len(RA))]
x = [[] for i in range(len(RA))]
y = [[] for i in range(len(RA))]
circle_mask = [[] for i in range(len(RA))]
totflux = [[] for i in range(len(RA))]
errflux = [[] for i in range(len(RA))]
rest_wavelength = [[] for i in range(len(RA))] # np.array([np.zeros(len(wavelength))] * len(RA), dtype=float)
COLS = []
COLS.append(pyfits.Column(name='vacuum_wavelength', unit='Angstrom', format='E', array=wav_vac))

FigNr = 0
for j in range(0, len(RA)):
    if (z[j] >= 2) and (z[j] <= 5): # select galaxies according to redshift interval after looking at data manually
        for q in range(len(wavelength)):
            data_slice = data[q, :, :]  # THIS IS THE LAYER for this wavelength
            data_slice_err = data_err[q, :, :]  # THIS IS THE VARIANCE
            ivar = 1./data_slice_err

            # CREATE A CIRCULAR MASK AT THE POSITION
            X = len(data_slice[0, :])
            Y = len(data_slice[:, 0])
            y[j], x[j] = np.ogrid[-ycenter[j]:Y-ycenter[j], -xcenter[j]:X-xcenter[j]] # COORDINATE SYSTEM
            circle_mask[j] =  x[j] * x[j] + y[j] * y[j] <= radius**2  # SELECT THE CIRCLE
            totflux[j] = np.nansum(data_slice[circle_mask[j]]) # THIS SUMS THE FLUX
            errflux[j] = 1.27 * np.nansum(data_slice_err[circle_mask[j]])**0.5   # 1.27 is to account for correlated noise

            flux_A[j].append(totflux[j])
            noise_pipe[j].append(errflux[j])

        rest_wavelength[j] = np.array(wavelength) / (1 + z[j])
        flux_A[j] = np.array(flux_A[j])  # flux_fraction
        noise_pipe[j] = np.array(noise_pipe[j])

        # SAVE THE SPECTRUM INTO A FITS TABLE
        target_nr = str(j)
        COLS.append(pyfits.Column(name='rest_vac_wavelen_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelength[j]))
        COLS.append(pyfits.Column(name='fluxdensity_total_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_A[j]))
        COLS.append(pyfits.Column(name='fluxdensity_total_ERR_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_pipe[j]))


        # ∀ GALAXIES, MAKE AN INDIVIDUAL SPECTRAL PLOT, zoom in to the Lya & He-II intervals:
        # Lyman-alpha subplot at ~121.5nm
        plt.figure(FigNr)
        plt.step(rest_wavelength[j], flux_A[j], 'b', label='flux')
        plt.step(rest_wavelength[j], noise_pipe[j], 'k', label='noise')
        plt.axvline(x = Ly_alpha_rest, color='c')
        plt.xlim(1180, 1250)
        plt.xlabel(r'wavelength in rest-frame $[\AA]$')
        plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
        plt.title('Ly-$\u03B1$ rest-peak of target %s at z = %s' % (target_nr, z[j]))   # \u03B1 == alpha
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig('plots/Lya_rest_spectrum_target_%s.pdf' % (target_nr))
        plt.clf()

        FigNr += 1

        # He-II subplot at ~164.0nm
        # fit the peak with a gaussian ==> see  PART 2
        plt.figure(FigNr)
        plt.step(rest_wavelength[j], flux_A[j], 'b', label='flux')
        plt.step(rest_wavelength[j], noise_pipe[j], 'k', label='noise')
        plt.axvline(x = HeII, color='c')
        plt.xlim(1610, 1670)
        plt.ylim(-5, 15)
        plt.xlabel(r'wavelength in rest-frame $[\AA]$')
        plt.ylabel(r'total flux density $10^{-20} \frac{erg}{s\cdot cm^2 A}$')
        plt.title('He-II rest-peak of target %s at z = %s' % (target_nr, z[j]))
        plt.grid(True)
        plt.legend(loc='best')
        plt.savefig('plots/HeII_rest_spectrum_target_%s.pdf' % (target_nr))
        plt.clf()

        FigNr += 1

# TODO: \
#   (make one plot for each gal that zooms in on the lya and he2 line to check)
#   first manually check then coding (bootstrap method) to measure flux => median / significance
#   then look at O-III and C-IV use paper for wavelength

cols = pyfits.ColDefs(COLS)
hdu = pyfits.BinTableHDU.from_columns(cols)
hdu.writeto('1D_SPECTRUM_3targets_%s.fits' % (NAME), overwrite=True)

print('finished part 1')

################################  P A R T    2  : SELECTION OF TARGET GALAXIES ACCORDING TO He-II PEAK  ####################################
EXTRACTIONS = '1D_SPECTRUM_3targets_%s.fits'%(NAME)
extractions = pyfits.open(EXTRACTIONS)
DATA = extractions[1].data
# vac_wavelength = DATA.field('vacuum_wavelength')

# CHOOSE GALAXIES THAT SHOW A He-II PEAK:
rest_vac_wavelen = [[] for i in range(len(RA))]
flux_dens_tot = [[] for i in range(len(RA))]
flux_dens_err = [[] for i in range(len(RA))]

Lya_indices = [] # save for each galaxy the list-indices of the wavelengths corresponding to the Lya peak
rest_wavelen_Lya = [[] for i in range(len(RA))]
flux_Lya = [[] for i in range(len(RA))]
noise_Lya = [[] for i in range(len(RA))]
residual_Lya = [[] for i in range(len(RA))]

HeII_indices = [] # save for each galaxy the list-indices of the wavelengths corresponding to the He-II peak
rest_wavelen_HeII = [[] for i in range(len(RA))]
flux_HeII = [[] for i in range(len(RA))]
noise_HeII = [[] for i in range(len(RA))]
residual_HeII = [[] for i in range(len(RA))]

columns = []

figNr = 0
for i in range(0, len(RA)):
    target_nr = str(i)
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

    #if (np.mean(noise_HeII[i]) < np.mean(flux_HeII[i])):  # select galaxies according to their noise
    # SAVE THE HeII-PEAK INTO A FITS TABLE
    columns.append(pyfits.Column(name='rest_wavelen_He-II_%s' % (target_nr), unit='Angstrom', format='E', array=rest_wavelen_HeII[i]))
    columns.append(pyfits.Column(name='flux_He-II_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=flux_HeII[i]))
    columns.append(pyfits.Column(name='noise_He-II_%s' % (target_nr), unit='10**-20 erg s-1 cm-2 A-1', format='E', array=noise_HeII[i]))

    # GAUSSIAN FIT:
    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    # He-II GAUSSIAN FIT:
    mean_HeII = sum(rest_wavelen_HeII[i] * flux_HeII[i]) / sum(flux_HeII[i]) # np.mean(flux_HeII[i])
    print("mean_HeII: ", mean_HeII)
    #array_mean_HeII = np.array(mean_HeII)
    sigma_HeII = np.sqrt(sum(flux_HeII[i] * (rest_wavelen_HeII[i] - HeII) ** 2) / sum(flux_HeII[i]))
    #gmodel = Model(Gauss, nan_policy='propagate')
    #result_HeII = gmodel.fit(flux_HeII[i], x=rest_wavelen_HeII[i], amp=max(flux_HeII[i]), cen=HeII, wid=sigma_HeII)
    popt, pcov = curve_fit(Gauss, rest_wavelen_HeII[i], flux_HeII[i], p0=[max(flux_HeII[i]), HeII, sigma_HeII], maxfev=1000000000)
    perr_gauss = np.sqrt(np.diag(pcov))
    residual_HeII[i] = flux_HeII[i] - (Gauss(rest_wavelen_HeII[i], *popt))
    Delta_v_HeII = const.speed_of_light * (mean_HeII - HeII) / (HeII * 1000)  # velocity offset ([km/s]) between HeII = 164.0nm and detected maximum; c in m/s
    print("Delta_v_HeII: ", Delta_v_HeII)


    if ((0 <= np.mean(flux_HeII[i])) and (Delta_v_HeII >= -800) and (Delta_v_HeII <= 500)):# # TODO: MAY CHANGE BOUDNARIES
        # plt.figure(figNr)
        figure = plt.figure(figNr)
        frame1 = figure.add_axes((0.1, 0.3, 0.8, 0.6))
        plt.plot(rest_wavelen_HeII[i], Gauss(rest_wavelen_HeII[i], *popt), 'r-', label='fit')
        #plt.plot(rest_wavelen_HeII[i], result_HeII.best_fit, 'r-', label='best fit')
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
        plt.suptitle('He-II rest-peak of target %s at z = %s' % (target_nr, z[i]))
        plt.grid(True)
        frame1.get_shared_y_axes().join(frame1, frame2)
        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        plt.savefig('plots/HeII_Gauss_target_%s.pdf' % (target_nr))
        figure.clear()

        figNr += 1

        # Lya GAUSSIAN FIT:
        mean_Lya = sum(rest_wavelen_Lya[i] * flux_Lya[i]) / sum(flux_Lya[i]) # np.mean(flux_Lya[i])
        sigma_Lya = np.sqrt(sum(flux_Lya[i] * (rest_wavelen_Lya[i] - Ly_alpha_rest) ** 2) / sum(flux_Lya[i]))
        #gmodel = Model(Gauss, nan_policy='propagate')
        #result_Lya = gmodel.fit(flux_Lya[i], x=rest_wavelen_Lya[i], amp=max(flux_Lya[i]), cen=Ly_alpha_rest, wid=sigma_Lya)
        popt, pcov = curve_fit(Gauss, rest_wavelen_Lya[i], flux_Lya[i], p0=[max(flux_Lya[i]), mean_Lya, sigma_Lya], maxfev=1000000000)
        perr_gauss = np.sqrt(np.diag(pcov))
        residual_Lya[i] = flux_Lya[i] - (Gauss(rest_wavelen_Lya[i], *popt))

        #if (0 <= np.mean(flux_HeII[i])):
        # plt.figure(figNr)
        figure = plt.figure(figNr)
        frame1 = figure.add_axes((0.1, 0.3, 0.8, 0.6))
        plt.plot(rest_wavelen_Lya[i], Gauss(rest_wavelen_Lya[i], *popt), 'r-', label='fit')
        #plt.plot(rest_wavelen_Lya[i], result_Lya.best_fit, 'r-', label='best fit')
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
        plt.suptitle('Ly-$\u03B1$ rest-peak of target %s at z = %s' % (target_nr, z[i]))
        plt.grid(True)
        frame1.get_shared_y_axes().join(frame1, frame2)
        frame1.set_xticklabels([])  # Remove x-tic labels for the first frame
        plt.savefig('plots/Lya_Gauss_target_%s.pdf' % (target_nr))
        figure.clear()

        figNr += 1

Cols = pyfits.ColDefs(columns)
hdu = pyfits.BinTableHDU.from_columns(Cols)
hdu.writeto('HeII_selected_3targets_%s.fits' % (NAME), overwrite=True)
print('finished part 2')

################################  P A R T    3 : BOOTSTRAP EXPERIMENT   ####################################

# TODO:
#  1) BOOTSTRAP method to measure flux => median / significance
#  2) then look at O-III and C-IV

# TODO: now: use all of wavelength-array to determine the noise-average: use only the He-II interval 1630-1655

bootstrap_flux_input = flux_A[1] # choose the second target here, since He-II line most dominant
                            # from the targets that show a He-II peak; choose a random galaxy using 'bootstrap_input = random.choice(flux_A_HeII)'
bootstrap_error_input = noise_pipe[1]
# print(statistics.mean(bootstrap_flux_input))
N = 10 # nr of bootstrap; use later 1000
bootstrap_flux = []
bootstrap_error = []
average_flux_estimate = [np.mean(bootstrap_flux_input)]
average_error_estimate = [np.mean(bootstrap_error_input)]
for n in range(0, N):
    for q in range(0, len(wavelength)):
        bootstrap_flux.append(np.random.choice(bootstrap_flux_input))
        bootstrap_error.append(np.random.choice(bootstrap_error_input))
    average_flux_estimate.append(np.mean(bootstrap_flux))
    average_error_estimate.append(np.mean(bootstrap_error))

bootstrap_flux = np.array(bootstrap_flux, dtype=float)
bootstrap_error = np.array(bootstrap_error, dtype=float)
average_flux_estimate = np.array(average_flux_estimate, dtype=float)  # note: the first entry is the mean of bootstrap_input!!!
average_error_estimate = np.array(average_error_estimate, dtype=float)  # note: the first entry is the mean of bootstrap_input!!!

print(average_error_estimate) # THIS IS THE NOISE N IN THE NEXT SECTION; SNR
print(average_flux_estimate)
print('finished part 3')


################################  P A R T    4 : SNR   ####################################
# SNR = S / N with S the flux and N the nose from the boostrap

print('finished part 4')


end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))



