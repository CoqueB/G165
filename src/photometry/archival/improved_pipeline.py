# Second Draft Photometry Pipeline:
# =================================

# Kron Photometry:
# ================


# Importing libraries:
# -------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.table import Table
from astropy.table import MaskedColumn
from astropy.nddata import NDData
from astropy.modeling.models import Sersic2D
from astropy.nddata import Cutout2D
from astropy.table import Column
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian2DKernel

from photutils.background import Background2D, MedianBackground
from photutils.utils import calc_total_error
from photutils.segmentation import SourceFinder, SourceCatalog
from photutils.psf import extract_stars, EPSFBuilder



# Loading FITS Image:
# -------------------

fits_file = "/mnt/c/Users/Coque/Desktop/astronomy_research/G165/cutouts/cutout1.fits"  # Replace with the actual FITS file path
hdul = fits.open(fits_file)       # This is the HDUList object
hdu = hdul[0]                     # This is the PrimaryHDU
image_header = hdu.header
data = hdu.data.astype(float)
hdul.close()                      # Close the HDUList, not the PrimaryHDU

image_data = data * image_header['PHOTMJSR'] #Converts the image from surface brightness to flux in MJy

# Estimating and subtracting background:
# --------------------------------------

bkg_estimator = MedianBackground() 
bkg = Background2D(image_data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
image_sub = image_data - bkg.background

# Calculating Uncertainty:
# ------------------------

error = calc_total_error(data, bkg.background_rms, effective_gain=1.0)

# Convolving image with Gaussian kernel:
# --------------------------------------

kernel = kernel = Gaussian2DKernel(x_stddev=3.0)  # ** detection filter of 3.0 recommended for galaxies
threshold = 3.0 * bkg.background_rms         # 3-sigma threshold

# Using SourceFinder():
# ---------------------

finder = SourceFinder(npixels=5, deblend=True, nlevels=32, contrast=0.001)
smoothed_image = convolve_fft(image_sub, kernel)
segm = finder(smoothed_image, threshold)



# Saving egmentation map to a FITS file:
# --------------------------------------

segm_data = segm.data.astype(np.int32)  # Cast to int32 for DS9 compatibility
segm_hdu = fits.PrimaryHDU(data=segm_data, header=image_header)
segm_hdu.writeto('segmentation_map.fits', overwrite=True)
print("Segmentation map saved as 'segmentation_map.fits'")


# Extracting source properties using SourceCatalog():
# ---------------------------------------------------

catalog = SourceCatalog(image_sub, segm, error = error)
tbl = catalog.to_table()

# Computing AB magnitudes using Kron flux:
# --------------------------------------


kron_flux = tbl['kron_flux']
valid = kron_flux > 0

ab_kron_mag = MaskedColumn(np.zeros_like(kron_flux), mask=~valid)
ab_kron_mag[valid] = -2.5 * np.log10(kron_flux[valid] * image_header['PIXAR_SR']) - 6.10
tbl['ab_kron_mag'] = ab_kron_mag

# Visualizing segmentation map:
# -----------------------------

plt.imshow(segm.data, cmap='tab20b', origin='lower')
plt.title('Deblended Segmentation Map')
plt.colorbar(label='Source Label')
plt.show()


# PSF Photometry:
# ===============


# Selecting stars for and building the EPSF:
# ------------------------------------------


# Select a few bright, isolated from your source table
# For simplicity, here we manually filter small, round sources



a = tbl['semimajor_sigma']
b = tbl['semiminor_sigma']
ellipticity = 1 - (b / a)
tbl['ellipticity'] = ellipticity


bright_stars = tbl[(tbl['kron_flux'] > 1e-6) & (tbl['ellipticity'] < 0.2)]  # should a size limit be imposed?
stars_tbl = bright_stars[['xcentroid', 'ycentroid']].copy()
stars_tbl.rename_columns(['xcentroid', 'ycentroid'], ['x', 'y'])

# Creating cutouts of the sourses selected for EPSF:
# --------------------------------------------------

nddata = NDData(data=image_sub)
star_cutouts = extract_stars(nddata, stars_tbl, size=25)

# Building the EPSFs and storing them in an array:
# ------------------------------------------------

epsf_builder = EPSFBuilder(oversampling=4)  # oversampling=4 is the resolution of the EPSF
epsf, fitted_stars = epsf_builder.build_epsf(star_cutouts)  # builds a model of the PSF by stacking and averaging the star images
psf_array = epsf.data       # Final PSF

# Creating the Sersic model:
# --------------------------

sersic_params = []
sersic_flux = []

for i, row in enumerate(tbl):
    x0, y0 = row['xcentroid'], row['ycentroid']
    a = row['semimajor_sigma']
    b = row['semiminor_sigma']
    theta = row['orientation']

    # Making cutouts for all the galaxies 

    cutout_size = int((6 * a).value) 
    try:
        cutout = Cutout2D(image_sub, (x0, y0), (cutout_size, cutout_size))
    except Exception as e:
        print(f"[{i}] Cutout failed: {e}")
        sersic_params.append((np.nan,) * 7)
        sersic_flux.append(np.nan)
        continue


    # Builds coordinate grids for evaluating the model and centers on the middle of the cutouts
    y, x = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
    x_cen = cutout.shape[1] // 2
    y_cen = cutout.shape[0] // 2

    # Remove units before fitting
    amp = cutout.data.max()
    r_eff = a.to_value('pix')
    ellip = 1 - b.to_value('pix') / a.to_value('pix')
    theta_val = theta.to_value('rad')


    # Initial parameter guess: [amplitude, r_eff, n, x_0, y_0, ellip, theta]
    p0 = [amp, r_eff, 2.0, x_cen, y_cen, ellip, theta_val]



    # This function tells the fitting algorithm how to measure error between the observed
    #  cutout and the PSF-convolved Sérsic model
    def residual(params, x, y, data, psf):
        try:
            model = Sersic2D(*params)
            model_img = model(x, y)
            conv = convolve_fft(model_img, psf)
            return (conv - data).ravel()
        except Exception:
            return np.ones_like(data).ravel() * 1e9
        

    try:
        # optimizing the model parameters: changing the Sérsic
        #  parameters until the PSF-convolved model best matches the cutout
        result = least_squares(residual, p0, args=(x, y, cutout.data, psf_array), max_nfev=100)
        params = result.x
        sersic_params.append(params)

        # Compute total flux using the formula
        amp, r_eff, n = params[0:3]
        bn = 2 * n - 1/3
        flux = amp * (2 * np.pi * r_eff**2 * n * np.exp(bn)) / (bn**(2 * n))
        sersic_flux.append(flux)

        print(f"[{i}] Fitted flux: {flux:.3e}")

    except Exception as e:
        print(f"[{i}] Sérsic fitting failed: {e}")
        sersic_params.append((np.nan,) * 7)
        sersic_flux.append(np.nan)

params_arr = np.array(sersic_params)

# Galaxy structural parameters corrected for the PSF : amplitude, r_eff, n, x0, y0, ellip, theta

tbl['sersic_amp'] = Column(params_arr[:, 0])
tbl['sersic_r_eff'] = Column(params_arr[:, 1])
tbl['sersic_n'] = Column(params_arr[:, 2])
tbl['sersic_x0'] = Column(params_arr[:, 3] + tbl['xcentroid'] - cutout_size // 2)
tbl['sersic_y0'] = Column(params_arr[:, 4] + tbl['ycentroid'] - cutout_size // 2)
tbl['sersic_ellip'] = Column(params_arr[:, 5])
tbl['sersic_theta'] = Column(params_arr[:, 6])
tbl['sersic_flux'] = sersic_flux

# Convert to AB magnitudes:
# -------------------------

valid = np.array(sersic_flux) > 0
ab_sersic_mags = MaskedColumn(np.zeros_like(sersic_flux), mask=~valid)
ab_sersic_mags[valid] = -2.5 * np.log10(np.array(sersic_flux)[valid] * image_header['PIXAR_SR']) - 6.1

tbl['ab_sersic_mag'] = ab_sersic_mags


# Printing tbl:
# -------------

print(tbl)







