# First Draft Aperture Photometry Pipeline:
# ==========================================

# Importing libraries:
# -------------------

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.table import Column
from astropy.table import MaskedColumn
from astropy.nddata import Cutout2D
from astropy.nddata import NDData
from astropy.modeling.models import Sersic2D
from astropy.convolution import convolve_fft
from scipy.optimize import least_squares
import numpy as np
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.segmentation import SourceCatalog
from photutils.segmentation import source_properties
from photutils.aperture import EllipticalAperture, aperture_photometry
from photutils.aperture import EllipticalAnnulus
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.isophote import build_ellipse_model
from photutils.psf import extract_stars, EPSFBuilder


# Loading FITS Image:
# -------------------

fits_file = 'image.fits'  # Replace with the actual FITS file path
hdu = fits.open(fits_file)[0]
image_header = hdu.header
data = hdu.data.astype(float)
hdu.close()

image_data = data * image_header['PHOTMJSR'] #Converts the image from surface brightness to flux in MJy

# Estimating and subtracting background:
# --------------------------------------

bkg_estimator = MedianBackground() # Use mefian or mean?
bkg = Background2D(image_data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
image_sub = image_data - bkg.background

# Detecting sources:
# ------------------

threshold = detect_threshold(image_sub, snr=3.0)  # 3-sigma threshold
segm = detect_sources(image_sub, threshold, npixels=5)


# Measuring source properties:
# -----------------------------

catalog = SourceCatalog(image_sub, segm)
tbl = catalog.to_table()

"""
props = source_properties(image_sub, segm)
tbl = props.to_table()
"""

print("Detected sources:", len(tbl))

# printing tbl
for i, row in enumerate(tbl):
    xcen, ycen = row.xcentroid.value, row.ycentroid.value
    a = row.semimajor_axis_sigma.value
    b = row.semiminor_axis_sigma.value
    theta = row.orientation.value  # in radians

    print(f"[{i}] x={xcen:.1f}, y={ycen:.1f}, a={a:.1f}, b={b:.1f}, theta={theta:.2f} rad")


# Performing aperture photometry:
# -------------------------------

positions = [(row.xcentroid.value, row.ycentroid.value) for row in tbl]
apertures = [EllipticalAperture(pos, 3*row.semimajor_axis_sigma, 3*row.semiminor_axis_sigma,
                                row.orientation) for pos, row in zip(positions, tbl)]
phot_table = aperture_photometry(image_sub, apertures)
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output


# Creating an annulus around an aperture to calculate the background

# Scaling factors
scale = 3.0          # inner annulus boundary is the same as the aperture
annulus_ratio = 1.5  # annulus outer edge is 1.5× further than inner edge

annulus_apertures = []
for row in tbl:
    x, y = row.xcentroid, row.ycentroid
    a = row.semimajor_axis_sigma
    b = row.semiminor_axis_sigma
    theta = row.orientation

    annulus = EllipticalAnnulus(
        (x, y),
        a_in=scale * a,
        a_out=scale * a * annulus_ratio,
        b_in=scale * b,
        b_out=scale * b * annulus_ratio,
        theta=theta
    )
    annulus_apertures.append(annulus)

# subtracting the background:

annulus_table = aperture_photometry(image_sub, annulus_apertures)

phot_table['annulus_sum'] = annulus_table['aperture_sum']

bkg_means = [annulus_table['aperture_sum'][i] / annulus_apertures[i].area
             for i in range(len(annulus_apertures))]

phot_table['bkg_mean'] = bkg_means

bkg_sub_fluxes = [phot_table['aperture_sum'][i] - bkg_means[i] * apertures[i].area
                  for i in range(len(apertures))]

phot_table['bkg_subtracted_flux'] = bkg_sub_fluxes


print(phot_table)


# Adding photometric Zeropoints:
# -------------------------------


fluxes = np.array(phot_table['bkg_subtracted_flux'])
valid = fluxes > 0
ab_aperture_mags = MaskedColumn(np.zeros_like(fluxes), mask=~valid)
ab_aperture_mags[valid] = -2.5 * np.log10(fluxes[valid]*image_header['PIXAR_SR']) -6.1  # converts to AB mags, not normal Vega mags
phot_table['ab_aperture_mag'] = ab_aperture_mags


# Plotting the image with the apertures:
# ---------------------------------------

norm = simple_norm(image_sub, 'sqrt', percent=99) #stretch factor (scale)
plt.figure(figsize=(10, 8))
plt.imshow(image_sub, cmap='inferno', norm=norm, origin='lower')
for aperture in apertures:
    aperture.plot(color='cyan', lw=1.0)
for annulus in annulus_apertures:
    annulus.plot(color='cyan', lw=1.0, alpha=0.7)
plt.title('Detected Sources with Apertures')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.colorbar(label='Intensity')
plt.show()



# Elliptical Isophotal Photometry:
# ================================

isophotal_fluxes = []

# Using the same detected sources as for aperture photometry

for i, row in enumerate(tbl):

    xcen = row.xcentroid.value
    ycen = row.ycentroid.value
    a = row.semimajor_axis_sigma.value
    b = row.semiminor_axis_sigma.value
    theta = row.orientation.value

    # Define a cutout around the source
    cutout_size = int(6 * a)  
    try:
        cutout = Cutout2D(image_sub, (xcen, ycen), (cutout_size, cutout_size))
    except Exception as e:
        print(f"Skipping source {i} (cutout failed): {e}")
        continue

    # Create initial EllipseGeometry
    try:
        geom = EllipseGeometry(x0=cutout.shape[1] // 2,
                               y0=cutout.shape[0] // 2,
                               sma=a,
                               eps=1 - (b / a),
                               pa=theta)
        ellipse = Ellipse(cutout.data, geometry=geom)
        isolist = ellipse.fit_image()
        table = isolist.to_table()

        print(f"  Fitted {len(table)} isophotes for source {i}")
        
        """
        # Ploting the intensity profile
        plt.figure()
        plt.plot(table['sma'], table['intens'], label=f"Source {i}")
        plt.xlabel("Semi-major axis (pixels)")
        plt.ylabel("Mean Intensity")
        plt.title(f"Isophote Profile - Source {i}")
        plt.legend()
        plt.show()
        """

    except Exception as e:
        print(f"  Failed isophote fitting for source {i}: {e}")
        continue
    
    # Compute total isophotal flux
    try:
        model_image = build_ellipse_model(cutout.data.shape, isolist)
        iso_flux = model_image.sum()
        isophotal_fluxes.append(iso_flux)
        print(f"  Isophotal flux for source {i}: {iso_flux:.3e} Jy")
    except Exception as e:
        print(f"  Failed to compute model flux for source {i}: {e}")



# Adding the isophotal fluxes to phot_table

phot_table['iso_fluxes'] = isophotal_fluxes

# Converting the isophotal fluxes into AB magnitudes 

iso_fluxes = np.array(isophotal_fluxes)
valid_iso = iso_fluxes > 0
ab_iso_mags = MaskedColumn(np.zeros_like(iso_fluxes), mask=~valid_iso)
ab_iso_mags[valid_iso] = -2.5 * np.log10(iso_fluxes[valid_iso] * image_header['PIXAR_SR']) - 6.1

# Adding AB magnitudes to table

phot_table['ab_iso_mag'] = ab_iso_mags


# PSF Photometry:
# ===============

# Select a few bright, isolated from your source table  --> Must they be stars?
# For simplicity, here we manually filter small, round sources

bright_stars = tbl[(tbl['kron_flux'] > 1e-6) & (tbl['ellipticity'] < 0.2)]  # should a size limit be imposed?
stars_tbl = bright_stars[['xcentroid', 'ycentroid']]

# Creating cutouts of the sourses selected for EPSF

nddata = NDData(data=image_sub)
star_cutouts = extract_stars(nddata, stars_tbl, size=25)

# Building the EPSFs and storing them in an array

epsf_builder = EPSFBuilder(oversampling=4)  # oversampling=4 is the resolution of the EPSF
epsf, fitted_stars = epsf_builder.build_epsf(star_cutouts)  # builds a model of the PSF by stacking and averaging the star images
psf_array = epsf.data       # Final PSF

# Creating the Sersic model

sersic_params = []

for i, row in enumerate(tbl):
    x0, y0 = row['xcentroid'].value, row['ycentroid'].value
    a = row['semimajor_axis_sigma'].value
    b = row['semiminor_axis_sigma'].value
    theta = row['orientation'].value

    # Making cutouts for all the galaxies 

    cutout_size = int(6 * a)
    try:
        cutout = Cutout2D(image_sub, (x0, y0), (cutout_size, cutout_size))
    except Exception as e:
        print(f"[{i}] Cutout failed: {e}")
        sersic_params.append((np.nan,) * 7)  # ** Notice this may also be usefull for before
        continue

    # Builds coordinate grids for evaluating the model and centers on the middle of the cutouts

    y, x = np.mgrid[:cutout.shape[0], :cutout.shape[1]]
    x_cen, y_cen = cutout.shape[1] // 2, cutout.shape[0] // 2

    # Initial guess for the parameters of the Sérsic model
    p0 = [cutout.data.max(), a, 2.0, x_cen, y_cen, 1 - b/a, theta]  # amp, r_eff, n, x0, y0, ellip, theta

    # This function tells the fitting algorithm how to measure error between the observed cutout and the PSF-convolved Sérsic model

    def residual(params, x, y, data, psf):
        try:
            model = Sersic2D(*params)
            model_img = model(x, y)
            conv = convolve_fft(model_img, psf)
            return (conv - data).ravel()
        except Exception:
            return np.ones_like(data).ravel() * 1e9
        
    # optimizing the model parameters: changing the Sérsic parameters until the PSF-convolved model best matches the cutout

    try:
        result = least_squares(residual, p0, args=(x, y, cutout.data, psf_array), max_nfev=100)
        sersic_params.append(result.x)

    except Exception as e:
        print(f"[{i}] Sérsic fitting failed: {e}")
        sersic_params.append((np.nan,) * 7)


params_arr = np.array(sersic_params)

# Galaxy structural parameters corrected for the PSF : amplitude, r_eff, n, x0, y0, ellip, theta

phot_table['sersic_amp'] = Column(params_arr[:, 0])
phot_table['sersic_r_eff'] = Column(params_arr[:, 1])
phot_table['sersic_n'] = Column(params_arr[:, 2])
phot_table['sersic_x0'] = Column(params_arr[:, 3] + tbl['xcentroid'] - cutout_size // 2)
phot_table['sersic_y0'] = Column(params_arr[:, 4] + tbl['ycentroid'] - cutout_size // 2)
phot_table['sersic_ellip'] = Column(params_arr[:, 5])
phot_table['sersic_theta'] = Column(params_arr[:, 6])


sersic_flux = []

for i, row in phot_table.iterrows():
    amp = row['sersic_amp']
    r_eff = row['sersic_r_eff']
    n = row['sersic_n']

    if np.any(np.isnan([amp, r_eff, n])):
        sersic_flux.append(np.nan)
    else:
        # This is the analytical approximation of the total flux of a Sérsic profile
        bn = 2 * n - 1/3  
        flux = amp * (2 * np.pi * r_eff**2 * n * np.exp(bn)) / (bn**(2 * n))
        sersic_flux.append(flux)

phot_table['sersic_flux'] = sersic_flux

# Convert to AB mag
fluxes = np.array(sersic_flux)
valid = fluxes > 0
ab_sersic_mags = MaskedColumn(np.zeros_like(fluxes), mask=~valid)
ab_sersic_mags[valid] = -2.5 * np.log10(fluxes[valid] * image_header['PIXAR_SR']) - 6.1
phot_table['ab_sersic_mag'] = ab_sersic_mags

    



