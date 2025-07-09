# First Draft Aperture Photometry Pipeline:
# ==========================================

# Importing libraries:
# -------------------

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.table import MaskedColumn
from astropy.nddata import Cutout2D
import numpy as np
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.segmentation import source_properties
from photutils.aperture import EllipticalAperture, aperture_photometry
from photutils.aperture import EllipticalAnnulus
from photutils.isophote import EllipseGeometry, Ellipse
from photutils.isophote import build_ellipse_model


# Loading FITS Image:
# -------------------

fits_file = 'image.fits'  # Replace with the actual FITS file path
hdu = fits.open(fits_file)[0]
image_header = hdu.header
data = hdu.data.astype(float)
hdu.close()

image_data = data / image_header['PHOTMJSR'] #Converts the image from surface brightness to flux

# Estimating and subrtacting background:
# --------------------------------------

bkg_estimator = MedianBackground() # Use mefian or mean?
bkg = Background2D(image_data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
image_sub = image_data - bkg.background

# Detecting sources:
# ------------------

threshold = detect_threshold(image_sub, snr=3.0)  # 3-sigma threshold
segm = detect_sources(image_sub, threshold, npixels=5)


# Measureing source porperties:
# -----------------------------

props = source_properties(image_sub, segm)
tbl = props.to_table()


print("Detected sources:", len(tbl))

# printing tbl
for i, row in enumerate(tbl):
    xcen, ycen = row.xcentroid.value, row.ycentroid.value
    a = row.semimajor_axis_sigma.value
    b = row.semiminor_axis_sigma.value
    theta = row.orientation.value  # in radians

    print(f"[{i}] x={xcen:.1f}, y={ycen:.1f}, a={a:.1f}, b={b:.1f}, theta={theta:.2f} rad")


# Perfroming aperture photometry:
# -------------------------------

positions = [(row.xcentroid, row.ycentroid) for row in tbl]
apertures = [EllipticalAperture(pos, 3*row.semimajor_axis_sigma, 3*row.semiminor_axis_sigma,
                                row.orientation) for pos, row in zip(positions, tbl)]
phot_table = aperture_photometry(image_sub, apertures)
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output


# creating an annulus arround an aperture to calculate the background

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
ab_mags = MaskedColumn(np.zeros_like(fluxes), mask=~valid)
ab_mags[valid] = -2.5 * np.log10(fluxes[valid]) -6.1  # converts to AB mags, not normal Vega mags
phot_table['ab_mag'] = ab_mags


# Plotting the image with the appertures:
# ---------------------------------------

norm = simple_norm(image_sub, 'sqrt', percent=99) #strech factor (scale)
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



# adding the isophotal fluxes to phot_table

phot_table['iso_fluxes'] = isophotal_fluxes










""" The following is usefull to create am isophote map later if wanted

# Setting brightness threshold levels (in pixel values):
# ----------------------------------------------------------

vmin = np.min(image_data)  # least bright pixel value
vmax = np.max(image_data)  # most bright pixel value

# Establishing the boundaries for the Isophote map. .

# In this case we use logspace but this could also be done in linspace

log_brigtness_levels = np.logspace( vmin + 0.1*(vmax - vmin) , vmax , num=6) 

print(log_brigtness_levels)

"""


