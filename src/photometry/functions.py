
import os
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn
from astropy.table import Table
import numpy as np
import astropy.units as u
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
    wcs = WCS(image_header)
    #data = hdu.data.astype(float)
    data = hdu.data.astype(np.float32)
    hdul.close()
    image_data = data
    return image_header, image_data, wcs

def subtract_background(image_data):
    bkg = Background2D(image_data, (50, 50), filter_size=(3, 3)) 
    image_sub = image_data - bkg.background
    return image_sub, bkg

def load_weightfile(wht_file):
    wht_hdul = fits.open(wht_file)
    wht_hdu = wht_hdul[0]
    # weight_data = wht_hdu.data.astype(float)
    weight_data = wht_hdu.data.astype(np.float32)
    wht_hdul.close()
    return weight_data

def calculate_uncertainty(image_header, image_data, weight_data, bkg):
    exposure_time = image_header["XPOSURE"]
    print("XPOSURE =", exposure_time)
    exposure_time_map = (exposure_time * bkg.background_rms_median**2 * weight_data )

    #***
    print("bkg.background_rms_median =", bkg.background_rms_median)
    print("bkg.background_rms_median**2 =", bkg.background_rms_median**2)
    #***

    background_rms = np.zeros_like(weight_data, dtype=np.float32)
    mask = weight_data > 1e-3
    background_rms[mask] = 1 / np.sqrt(weight_data[mask])
    # print(background_rms[background_rms > 0][:20])   

    # ***
    valid = weight_data > 0
    print("'valid' means: weight_data > 0")
    print("weight statistics")
    print("  min valid =", np.nanmin(weight_data[valid]))
    print("  median valid =", np.nanmedian(weight_data[valid]))
    print("  max valid =", np.nanmax(weight_data[valid]))

    print("background rms statistics")
    print("  min =", np.nanmin(background_rms[valid]))
    print("  median =", np.nanmedian(background_rms[valid]))
    print("  max =", np.nanmax(background_rms[valid]))

    print("gain map statistics")
    print(f"  min    = {np.nanmin(exposure_time_map):.15e}")
    print(f"  median = {np.nanmedian(exposure_time_map):.15e}")
    print(f"  max    = {np.nanmax(exposure_time_map):.15e}")
    # ***

    print("bkg sub done")
    #data_rms = calc_total_error( image_data, background_rms, exposure_time + 1e-8)
    data_rms = calc_total_error( image_data, background_rms, exposure_time_map + 1e-8 )
    print(" calc error done")
    return data_rms, background_rms

def make_outputdir():
    output_dir = os.path.expanduser("~/G165/output/")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def source_detection(bkg, weight_data, image_sub, image_header, output_dir):
    threshold = 3.6 * bkg.background_rms
    print("starting")  
    finder = SourceFinder(npixels=10, deblend=True, nlevels=32, contrast=0.1)

    # defining a mask uing the weightfile to ony detect sources in the image
    mask = weight_data <= 0    #True in pixels that are to be ignored
    segm = finder(image_sub, threshold, mask=mask)
    segm_data = segm.data.astype(np.int32)
    segm_hdu = fits.PrimaryHDU(data=segm_data, header=image_header)
    segm_hdu.writeto(os.path.join(output_dir, 'segmentation_map.fits'), overwrite=True)
    print("Segmentation map saved as 'segmentation_map.fits'")
    return segm

def calculate_gini(flux):
    flux = np.asarray(flux).flatten()    
    flux = flux[np.isfinite(flux)]   # Remove NaNs

    # Gini assumes positive values.
    # For background-subtracted images, small negative values can occur.
    flux = flux[flux > 0]
    n = len(flux)
    if n == 0:
        return np.nan
    flux = np.sort(flux)
    mean_flux = np.nanmean(flux)
    if mean_flux == 0:
        return 0.0
    index = np.arange(1, n + 1)
    gini = np.sum((2 * index - n - 1) * flux) / (mean_flux * n * (n - 1))
    return gini

def extract_source_properties(image_sub, segm, data_rms, header):
    catalog = SourceCatalog(image_sub, segm, error=data_rms)
    tbl = catalog.to_table(columns=['label','xcentroid','ycentroid','semimajor_sigma',
    'semiminor_sigma','orientation','segment_flux','segment_fluxerr','kron_flux','kron_fluxerr'])
    

    # Filtering for sources with SNR >= 3
    # SNR = segment_flux / segment_fluxerr
    # segment_flux sums all pixel values in the source 
    # segment_fluxerr is the quadrature sum of errors over the same pixels

    snr = tbl['segment_flux'] / tbl['segment_fluxerr']
    tbl['snr'] = snr

    # ***
    print(f"segment_flux range:    {np.nanmin(tbl['segment_flux']):.4f} to {np.nanmax(tbl['segment_flux']):.4f}")
    print(f"segment_fluxerr range: {np.nanmin(tbl['segment_fluxerr']):.4f} to {np.nanmax(tbl['segment_fluxerr']):.4f}")
    print(f"SNR range:             {np.nanmin(snr):.4f} to {np.nanmax(snr):.4f}")
    print(f"SNR median:            {np.nanmedian(snr):.4f}")
    print(f"Sources with SNR >= 3: {(snr >= 3).sum()}")
    print(f"Sources with SNR >= 0: {(snr >= 0).sum()}")
    print(f"Sources with negative SNR: {(snr < 0).sum()}")
    #***

    n_before = len(tbl)
    tbl = tbl[snr >= 3]
    tbl = tbl.copy() 
    n_after = len(tbl)
    print(f"SNR filter (>= {3}): kept {n_after}/{n_before} sources")

    #"""
    # Calculate Gini coefficient for each source
    gini_values = []
    for label in tbl['label']:
        source_pixels = image_sub[segm.data == label]
        gini_values.append(calculate_gini(source_pixels))
    tbl['gini'] = gini_values

    # Filtering for sources with Gini Coefficient >= 0.5
    gini_mask = tbl['gini'] >= 0.5
    n_before = len(tbl)
    tbl = tbl[gini_mask]
    tbl = tbl.copy() 
    n_after = len(tbl)
    print(f"Gini filter (>= 0.5): kept {n_after}/{n_before} sources")
    #"""


    # Image is in MJy/sr, kron_flux is the SUM over Kron aperture
    # kron_flux = Σ(MJy/sr) over N pixels in aperture
    # To get total flux we need to account for solid angle
    # Each pixel has solid angle = PIXAR_SR
    # Total flux = kron_flux × PIXAR_SR
    
    pixel_area_sr = header['PIXAR_SR']  # steradians per pixel
    
    # Convert to total flux in Jy
    # kron_flux [MJy/sr] × pixel_area [sr/pixel] = MJy/pixel
    # But kron_flux is summed over N pixels, so it's (MJy/sr)×pixels
    # Multiply by pixel_area to get MJy, then ×10^6 for Jy

    kron_flux_jy = tbl['kron_flux'] * pixel_area_sr * 1e6
    kron_fluxerr_jy = tbl['kron_fluxerr'] * pixel_area_sr * 1e6
    
    # Adding units
    kron_flux_jy = kron_flux_jy * u.Jy
    kron_fluxerr_jy = kron_fluxerr_jy * u.Jy
    
    # Converting to AB magnitudes
    mag = kron_flux_jy.to(u.ABmag)
    magerr = 2.5 / np.log(10) * (kron_fluxerr_jy / kron_flux_jy)
    magerr = magerr.value * u.ABmag

    tbl['ab_kron_mag'] = mag
    tbl['ab_kron_mag_err'] = magerr

    tbl.rename_column('label', 'id')
    return tbl, catalog

def strip_quantity(x):
    return x.value if isinstance(x, Quantity) else x
    
def my_aperture_photometry(tbl, image_sub, wcs, data_rms, header):

    # defining annuli and appertures
    positions = np.transpose([tbl['xcentroid'].data, tbl['ycentroid'].data])
    apertures = CircularAperture(positions, r=14.0)
    annulus_apertures = CircularAnnulus(positions, r_in=14.0, r_out=21.0)

    # Performing aperture photometry:
    phot_table = aperture_photometry(image_sub, apertures, error = data_rms)
    annulus_table = aperture_photometry(image_sub, annulus_apertures,error = data_rms)

    # assigning IDs to each source
    phot_table['id'] = np.arange(len(phot_table))
    annulus_table['id'] = np.arange(len(annulus_table))

    # converting to sky coordinates
    ra, dec = wcs.pixel_to_world_values( tbl['xcentroid'].data, tbl['ycentroid'].data )

    phot_table['ra'] = ra
    phot_table['dec'] = dec

    # Perfoming background and subtraction:
    bkg_mean = annulus_table['aperture_sum'] / annulus_apertures.area
    bkg_sub_flux = phot_table['aperture_sum'] - (bkg_mean * apertures.area)

    # background subtraction error propagation 

    bkg_var_perpix = ( annulus_table['aperture_sum_err']**2 / annulus_apertures.area**2)  # variance of background mean per pixel
    bkg_var_ap = bkg_var_perpix * apertures.area**2                                       # background variance inside source aperture
    flux_err = np.sqrt( phot_table['aperture_sum_err']**2 + bkg_var_ap )                  # total flux error
    phot_table['bkg_subtracted_flux_err'] = flux_err

    # calculating magnitudes
    phot_table['bkg_mean'] = bkg_mean
    phot_table['bkg_subtracted_flux'] = bkg_sub_flux
    
    pixel_area_sr = header['PIXAR_SR'] * u.sr

    mag, magerr = fluxes2mags( phot_table['bkg_subtracted_flux'], phot_table['bkg_subtracted_flux_err'], pixel_area_sr)

    phot_table['ab_aperture_mag'] = mag
    phot_table['ab_aperture_mag_err'] = magerr

    return phot_table, apertures, annulus_apertures

def fluxes2mags(flux_sb, fluxerr_sb, pixel_area_sr):
    #Convert surface brightness (MJy/sr) summed over pixels into AB magnitudes.

    # Attach units (surface brightness)
    flux_sb = flux_sb * u.MJy / u.sr
    fluxerr_sb = fluxerr_sb * u.MJy / u.sr

    # Convert to flux density (Jy)
    flux = (flux_sb * pixel_area_sr).to(u.Jy)
    fluxerr = (fluxerr_sb * pixel_area_sr).to(u.Jy)

    # AB magnitude
    mag = flux.to(u.ABmag)

    # Magnitude uncertainty
    magerr = 2.5 / np.log(10) * (fluxerr / flux)

    return mag, magerr


def kron_photometry(tbl, phot_table):

    # copying to prevent mutation of the original
    tbl = tbl.copy()                 
    phot_table = phot_table.copy()   

    # Adding Kron photometry to phot_table:
    tbl['id'] = np.arange(len(tbl))
    kron_info = tbl['id', 'kron_flux', 'ab_kron_mag', 'ab_kron_mag_err']
    phot_table = join(phot_table, kron_info, keys='id', join_type='left')
    
    return phot_table