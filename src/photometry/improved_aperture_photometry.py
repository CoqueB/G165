# Aperture Photometry Pipeline:
# =============================

# Importing libraries:
# -------------------

import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.table import MaskedColumn
from astropy.table import Table
import numpy as np
from astropy.units import Quantity
from astropy.table import join
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.background import Background2D
from photutils.segmentation import SourceFinder
from photutils.utils import calc_total_error
from photutils.segmentation import SourceCatalog



# Loading FITS Image:
# -------------------

fits_file = "/mnt/c/Users/Coque/Desktop/astronomy_research/G165/cutouts/cutout_f444.fits"
hdul = fits.open(fits_file)
hdu = hdul[0]
image_header = hdu.header
data = hdu.data.astype(float)
hdul.close()

image_data = data * image_header['PHOTMJSR']  # Convert to MJy


# Estimating and subtracting background:
# --------------------------------------

bkg = Background2D(image_data, (50, 50), filter_size=(3, 3)) 
image_sub = image_data - bkg.background


# Calculating Uncertainty:
# ------------------------

wht_file = '/mnt/c/users/Coque/Desktop/astronomy_research/G165/cutouts/cutout_f444_wht.fits'
wht_hdul = fits.open(wht_file)
wht_hdu = wht_hdul[0]
weight_data = wht_hdu.data.astype(float)
wht_hdul.close()

exposure_time = image_header["XPOSURE"]
exposure_time_map = (exposure_time * bkg.background_rms_median**2 * weight_data )
background_rms = 1 / np.sqrt(weight_data)
data_rms = calc_total_error( image_data, background_rms, exposure_time_map + 1e-8 )




# Source Detection (with Gaussian smoothing):
# ------------------------------------------

threshold = 3.8 * bkg.background_rms  
finder = SourceFinder(npixels=10, deblend=False, nlevels=16, contrast=0.1) #decrease n_levels and increase contrast

# defining a mask uing the weightfile to ony detect sources in the image

mask = weight_data <= 0.001    #True in pixels that are to be ignored
segm = finder(image_sub, threshold, mask=mask)


# Creating Output Directory:
# --------------------------

output_dir = "/mnt/c/Users/Coque/Desktop/astronomy_research/G165/output/"
os.makedirs(output_dir, exist_ok=True)


# Saving segmentation map:
# ------------------------

segm_data = segm.data.astype(np.int32)
segm_hdu = fits.PrimaryHDU(data=segm_data, header=image_header)
segm_hdu.writeto(os.path.join(output_dir, 'segmentation_map.fits'), overwrite=True)
print("Segmentation map saved as 'segmentation_map.fits'")


# Extracting source properties:
# -----------------------------

catalog = SourceCatalog(image_sub, segm, error=data_rms)
tbl = catalog.to_table(columns=[
    'label', 'xcentroid', 'ycentroid',
    'semimajor_sigma', 'semiminor_sigma', 'orientation',
    'kron_flux'])

tbl.rename_column('label', 'id')


# Defining reusable functions:
# ----------------------------

# strip unit if present
def strip_quantity(x):
    return x.value if isinstance(x, Quantity) else x

#Convert flux to AB magnitude

def flux_to_ab_mag(flux, pixar_sr, zeropoint=6.1):
    flux = np.array(flux)
    valid = flux > 0
    mags = MaskedColumn(np.zeros_like(flux), mask=~valid)
    mags[valid] = -2.5 * np.log10(flux[valid] * pixar_sr) - zeropoint
    return mags


# Extracting vectorized positions and forming fixed circular apertures and annul:
# -------------------------------------------------------------------------------

positions = np.transpose([tbl['xcentroid'].data, tbl['ycentroid'].data])

apertures = CircularAperture(positions, r=14.0)
annulus_apertures = CircularAnnulus(positions, r_in=14.0, r_out=21.0)


# Performing aperture photometry:
# -------------------------------

phot_table = aperture_photometry(image_sub, apertures)

annulus_table = aperture_photometry(image_sub, annulus_apertures)

# assigning IDs to each source

phot_table['id'] = np.arange(len(phot_table))
annulus_table['id'] = np.arange(len(annulus_table))

# Perfoming background and subtraction and calculating magnitudes:
# ----------------------------------------------------------------

bkg_mean = annulus_table['aperture_sum'] / annulus_apertures.area
bkg_sub_flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)

phot_table['bkg_mean'] = bkg_mean
phot_table['bkg_subtracted_flux'] = bkg_sub_flux

phot_table['ab_aperture_mag'] = flux_to_ab_mag(phot_table['bkg_subtracted_flux'], image_header['PIXAR_SR'])

# Calculating the AB Magnitudes of the Kron photometry aswell
tbl['ab_kron_mag'] = flux_to_ab_mag(tbl['kron_flux'], image_header['PIXAR_SR'])


# Adding Kron photometry to phot_table:
# -------------------------------------

tbl['id'] = np.arange(len(tbl))
kron_info = tbl['id', 'kron_flux', 'ab_kron_mag']
phot_table = join(phot_table, kron_info, keys='id', join_type='left')


# Creating ra-dec file and .reg file:
# -----------------------------------

phot_table_ra_dec = phot_table.copy()
phot_table_ra_dec.keep_columns(['xcenter', 'ycenter'])

region_filename = os.path.join(output_dir, "sources.reg")
with open(region_filename, "w") as f:
    for x, y in zip(phot_table['xcenter'], phot_table['ycenter']): 
        f.write(f"circle({x},{y},{14})\n") 

print(f"DS9 region file saved to: {region_filename}")

# Plotting the image with apertures:
# ----------------------------------

norm = simple_norm(image_sub, 'sqrt', vmin= -0.01, vmax= 0.5)
plt.figure(figsize=(10, 8))
plt.imshow(image_sub, cmap='inferno', norm=norm, origin='lower')

for aperture in apertures:
    aperture.plot(color='cyan', lw=0.5)

#for annulus in annulus_apertures:
    #annulus.plot(color='cyan', lw=0.5, alpha=0.7)

for src in catalog:
    kron_ap = src.kron_aperture
    if kron_ap is not None:
        kron_ap.plot(color='violet', lw=0.5, alpha=0.8)


plt.title('Detected Sources with Apertures')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.colorbar(label='Intensity')
plt.savefig(os.path.join(output_dir, 'aperture_overlay.png'), dpi=300, bbox_inches='tight')



# Printing and storing results:
# -----------------------------

print("Photometry complete. Results:")
print(phot_table)
phot_table.write(os.path.join(output_dir, 'photometry_results.csv'),format='csv', overwrite=True)
phot_table_ra_dec.write(os.path.join(output_dir, 'photometry_results_ra_dec.csv'),format='csv', overwrite=True)
print("Photometry results saved to 'photometry_results.csv")