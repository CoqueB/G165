
import os
from astropy.io import fits
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

def load_image(fits_file):
    hdul = fits.open(fits_file)
    hdu = hdul[0]
    image_header = hdu.header
    data = hdu.data.astype(float)
    hdul.close()
    image_data = data * image_header['PHOTMJSR']  # Convert to MJy
    return image_header, image_data

def subtract_background(image_data):
    bkg = Background2D(image_data, (50, 50), filter_size=(3, 3)) 
    image_sub = image_data - bkg.background
    return image_sub, bkg

def load_weightfile(wht_file):
    wht_hdul = fits.open(wht_file)
    wht_hdu = wht_hdul[0]
    weight_data = wht_hdu.data.astype(float)
    wht_hdul.close()
    return weight_data

def calculate_uncertainty(image_header, image_data, weight_data, bkg):
    exposure_time = image_header["XPOSURE"]
    exposure_time_map = (exposure_time * bkg.background_rms_median**2 * weight_data )
    background_rms = 1 / np.sqrt(weight_data)
    data_rms = calc_total_error( image_data, background_rms, exposure_time_map + 1e-8 )
    return data_rms, background_rms

def make_outputdir():
    output_dir = "/mnt/c/Users/Coque/Desktop/astronomy_research/G165/output/"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def source_detection(bkg, weight_data, image_sub, image_header, output_dir):
    threshold = 3.8 * bkg.background_rms  
    finder = SourceFinder(npixels=10, deblend=False, nlevels=16, contrast=0.1) #decrease n_levels and increase contrast

    # defining a mask uing the weightfile to ony detect sources in the image
    mask = weight_data <= 0.001    #True in pixels that are to be ignored
    segm = finder(image_sub, threshold, mask=mask)
    segm_data = segm.data.astype(np.int32)
    segm_hdu = fits.PrimaryHDU(data=segm_data, header=image_header)
    segm_hdu.writeto(os.path.join(output_dir, 'segmentation_map.fits'), overwrite=True)
    print("Segmentation map saved as 'segmentation_map.fits'")
    return segm

def extract_source_properties(image_sub, segm, data_rms ):
    catalog = SourceCatalog(image_sub, segm, error=data_rms)
    tbl = catalog.to_table(columns=['label', 'xcentroid', 'ycentroid','semimajor_sigma',
     'semiminor_sigma', 'orientation','kron_flux'])
    tbl.rename_column('label', 'id')
    return tbl, catalog

def strip_quantity(x):
    return x.value if isinstance(x, Quantity) else x

def flux_to_ab_mag(flux, pixar_sr, zeropoint=6.1):
    flux = np.array(flux)
    valid = flux > 0
    mags = MaskedColumn(np.zeros_like(flux), mask=~valid)
    mags[valid] = -2.5 * np.log10(flux[valid] * pixar_sr) - zeropoint
    return mags

def my_aperture_photometry(tbl, image_sub, image_header):

    # defining annuli and appertures
    positions = np.transpose([tbl['xcentroid'].data, tbl['ycentroid'].data])
    apertures = CircularAperture(positions, r=14.0)
    annulus_apertures = CircularAnnulus(positions, r_in=14.0, r_out=21.0)

    # Performing aperture photometry:
    phot_table = aperture_photometry(image_sub, apertures)
    annulus_table = aperture_photometry(image_sub, annulus_apertures)

    # assigning IDs to each source
    phot_table['id'] = np.arange(len(phot_table))
    annulus_table['id'] = np.arange(len(annulus_table))

    # Perfoming background and subtraction and calculating magnitudes:
    bkg_mean = annulus_table['aperture_sum'] / annulus_apertures.area
    bkg_sub_flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)
    phot_table['bkg_mean'] = bkg_mean
    phot_table['bkg_subtracted_flux'] = bkg_sub_flux
    phot_table['ab_aperture_mag'] = flux_to_ab_mag(phot_table['bkg_subtracted_flux'], image_header['PIXAR_SR'])

    return phot_table, apertures, annulus_apertures

def kron_photometry(tbl, image_header, phot_table):

    # Calculating the AB Magnitudes of the Kron photometry
    tbl['ab_kron_mag'] = flux_to_ab_mag(tbl['kron_flux'], image_header['PIXAR_SR'])

    # Adding Kron photometry to phot_table:
    tbl['id'] = np.arange(len(tbl))
    kron_info = tbl['id', 'kron_flux', 'ab_kron_mag']
    phot_table = join(phot_table, kron_info, keys='id', join_type='left')
    
    return phot_table
