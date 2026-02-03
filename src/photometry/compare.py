import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

"""

bands = ["f090", "f115", "f150", "f200", "f277", "f356", "f410", "f444"]

image_files = {
    "f090": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f090w_30mas_20230403_drz.fits",
    "f115": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f115w_30mas_20230403_drz.fits",
    "f150": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f150w_30mas_20230403_drz.fits",
    "f200": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f200w_30mas_20230403_drz.fits",
    "f277": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f277w_30mas_20230403_drz.fits",
    "f356": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f356w_30mas_20230403_drz.fits",
    "f410": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f410m_30mas_20230403_drz.fits",
    "f444": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f444w_30mas_20230403_drz.fits",
}

weight_files = { b: image_files[b].replace("_drz.fits", "_wht.fits") for b in bands }

base_output_dir = "/mnt/c/Users/Coque/Desktop/astronomy_research/G165/graphs_output/"


def mask_catalog_with_weight(cat, wcs, weight, wht_min=0.001):

    sky = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    x, y = wcs.world_to_pixel(sky)

    x = np.array(x)
    y = np.array(y)

    # Inside image bounds
    inside = (
        (x >= 0) & (x < weight.shape[1]) &
        (y >= 0) & (y < weight.shape[0])
    )

    # Initialize mask
    good = np.zeros(len(cat), dtype=bool)
    good[inside] = weight[y[inside].astype(int), x[inside].astype(int)] > wht_min

    return cat[good]


def open_catalogs(band):
    ref = Table.read("./phot_massimo_iso.cat", format="ascii")
    cat = Table.read(f"./output/photometry_results_{band}.csv", format="csv")

    print("Massimo columns:")
    print(ref.colnames)

    print("\nMy catalog columns:")
    print(cat.colnames)

    return ref, cat

def search(ref, cat):
    c_ref = SkyCoord(ra=ref['ra'] * u.deg, dec=ref['dec'] * u.deg)
    c_cat = SkyCoord(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg)

    max_sep = 1 * u.arcsec

    idx_cat, idx_ref, sep2d, _ = c_cat.search_around_sky(c_ref, max_sep)

    good = (idx_cat < len(cat)) & (idx_ref < len(ref))
    idx_cat = idx_cat[good]
    idx_ref = idx_ref[good]
    sep2d = sep2d[good]

    ref_matched = ref[idx_ref]
    cat_matched = cat[idx_cat]

    return max_sep, idx_ref, idx_cat, sep2d, ref_matched, cat_matched

def create_regions_file( idx_ref, idx_cat, ref, cat, max_sep, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    region_filename = os.path.join(output_dir, "matches.reg")

    with open(region_filename, "w") as f:

        f.write("fk5\n")

        # Draw ref sources as circles
        for i in range(len(ref)):
            ra = ref['ra'][i]
            dec = ref['dec'][i]
            #f.write(f"circle({ra},{dec},{max_sep.to(u.arcsec).value}\") # color=red\n")
            f.write(f"circle({ra},{dec},0.1\") # color=red\n")

        # Draw matched cat sources as smaller points inside annuli
        for ref_idx, cat_idx in zip(idx_ref, idx_cat):
            ra = cat['ra'][cat_idx]
            dec = cat['dec'][cat_idx]
            f.write(f"circle({ra},{dec},0.1\") # color=green\n")

    print(f"DS9 region file saved to: {region_filename}")



def plot_aperture_source_flux(ref, cat,ref_flux_col,cat_flux_col,output_dir):

    if ref_flux_col not in ref.colnames:
        raise ValueError(
            f"'{ref_flux_col}' not found in ref catalog. "
            f"Available columns: {ref.colnames}" )

    if cat_flux_col not in cat.colnames:
        raise ValueError(
            f"'{cat_flux_col}' not found in cat catalog. "
            f"Available columns: {cat.colnames}" )

    ref_flux = ref[ref_flux_col]
    cat_flux = cat[cat_flux_col]

    # Remove non-physical values
    ref_flux = ref_flux[ref_flux > 0]
    cat_flux = cat_flux[cat_flux > 0]

    # Shared logarithmic bins
    bins = np.logspace(
        np.log10(min(ref_flux.min(), cat_flux.min())),
        np.log10(max(ref_flux.max(), cat_flux.max())),
        50  )

    plt.figure(figsize=(7,5))
    plt.hist(ref_flux, bins=bins, histtype='step', label='Reference catalog')
    plt.hist(cat_flux, bins=bins, histtype='step', label='Comparison catalog')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('mag')
    plt.ylabel('Number of sources')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'aperture_hist.png'), dpi=300, bbox_inches='tight')

def plot_kron_source_flux(ref, cat,ref_flux_col,cat_flux_col,output_dir):

    if ref_flux_col not in ref.colnames:
        raise ValueError(
            f"'{ref_flux_col}' not found in ref catalog. "
            f"Available columns: {ref.colnames}" )

    if cat_flux_col not in cat.colnames:
        raise ValueError(
            f"'{cat_flux_col}' not found in cat catalog. "
            f"Available columns: {cat.colnames}" )

    ref_flux = ref[ref_flux_col]
    cat_flux = cat[cat_flux_col]

    # Remove non-physical values
    ref_flux = ref_flux[ref_flux > 0]
    cat_flux = cat_flux[cat_flux > 0]

    # Shared logarithmic bins
    bins = np.logspace(
        np.log10(min(ref_flux.min(), cat_flux.min())),
        np.log10(max(ref_flux.max(), cat_flux.max())),
        50 )

    plt.figure(figsize=(7,5))
    plt.hist(ref_flux, bins=bins, histtype='step', label='Reference catalog')
    plt.hist(cat_flux, bins=bins, histtype='step', label='Comparison catalog')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('mag')
    plt.ylabel('Number of sources')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'kron_hist.png'), dpi=300, bbox_inches='tight')


def plot_pixel_flux(image_data, output_dir):
    pixels = image_data[np.isfinite(image_data)]
    plt.figure(figsize=(7,5))
    plt.hist(pixels, bins=500, histtype='step')
    plt.yscale('log')
    plt.xlabel('Pixel flux')
    plt.ylabel('Number of pixels')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_flux_histogram.png'), dpi=300, bbox_inches='tight')

def plot_delta_mag_vs_mag(ref, cat, ref_mag_col, cat_mag_col, output_dir, band):

    if ref_mag_col not in ref.colnames:
        raise ValueError(f"{ref_mag_col} not in reference catalog")

    if cat_mag_col not in cat.colnames:
        raise ValueError(f"{cat_mag_col} not in comparison catalog")

    ref_mag = ref[ref_mag_col]
    cat_mag = cat[cat_mag_col]

    # Delta magnitude = mine - Massimo's
    delta_mag = cat_mag - ref_mag

    # Remove non-finite values
    good = np.isfinite(ref_mag) & np.isfinite(cat_mag)
    ref_mag = ref_mag[good]
    delta_mag = delta_mag[good]

    plt.figure(figsize=(7,5))
    plt.scatter(ref_mag, delta_mag, s=5, alpha=0.5)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)

    plt.xlabel('mag')
    plt.ylabel('Δmag')
    plt.xlim(10, 40)   
    plt.ylim(-10, 20)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'{band}_delta_mag_vs_mag.png'), dpi=300, bbox_inches='tight')

def plot_mag_err_vs_mag(cat, mag_col, mag_err_col, output_dir, band):

    if mag_col not in cat.colnames:
        raise ValueError(f"{mag_col} not in catalog")

    if mag_err_col not in cat.colnames:
        raise ValueError(f"{mag_err_col} not in catalog")

    mag = cat[mag_col]
    mag_err = cat[mag_err_col]

    good = np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    mag = mag[good]
    mag_err = mag_err[good]

    plt.figure(figsize=(7,5))
    plt.scatter(mag, mag_err, s=5, alpha=0.5)

    plt.xlabel('mag')
    plt.ylabel('err(mag)')
    plt.yscale('log')
    plt.tight_layout()

    plt.savefig( os.path.join(output_dir, f'{band}_mag_err_vs_mag.png'), dpi=300, bbox_inches='tight')


for band in bands:
    print(f"\n=== Processing {band.upper()} ===")

    # Load image + weight
    img_hdu = fits.open(image_files[band])[0]
    wht_hdu = fits.open(weight_files[band])[0]

    wcs = WCS(img_hdu.header)
    image_data = img_hdu.data
    weight = wht_hdu.data

    # Output directory per band
    output_dir = os.path.join(base_output_dir, band)
    os.makedirs(output_dir, exist_ok=True)

    # Open catalogs
    massimo_cat, my_cat = open_catalogs(band)

    # Mask using weight map
    my_cat = mask_catalog_with_weight(my_cat, wcs, weight).copy()
    massimo_cat = mask_catalog_with_weight(massimo_cat, wcs, weight).copy()

    # Cross-match
    max_sep, idx_ref, idx_cat, sep2d, ref_matched, cat_matched = search(massimo_cat, my_cat)

    # Regions file
    create_regions_file(
        idx_ref, idx_cat,
        my_cat, massimo_cat,
        max_sep, output_dir )

    # Column names
    if band == "f410":
        ref_mag_col = band + "m"
    else:
        ref_mag_col = band + "w"        
    cat_ap_col = "ab_aperture_mag"
    cat_kron_col = "ab_kron_mag"

    # Plotting
    plot_aperture_source_flux(
        ref=massimo_cat,
        cat=my_cat,
        ref_flux_col=ref_mag_col,
        cat_flux_col=cat_ap_col,
        output_dir=output_dir )

    plot_kron_source_flux(
        ref=massimo_cat,
        cat=my_cat,
        ref_flux_col=ref_mag_col,
        cat_flux_col=cat_kron_col,
        output_dir=output_dir )

    plot_delta_mag_vs_mag(
        ref=ref_matched,
        cat=cat_matched,
        ref_mag_col=ref_mag_col,
        cat_mag_col=cat_ap_col,
        output_dir=output_dir,
        band = band )


    plot_mag_err_vs_mag(
        cat=my_cat,
        mag_col=cat_ap_col,
        mag_err_col="ab_aperture_mag_err",
        output_dir=output_dir,
        band = band )
"""


#########################################################


img_hdu = fits.open("/mnt/c/Users/Coque/Desktop/astronomy_research/G165/original_small_cutouts/cutout1.fits")[0]
wht_hdu = fits.open("/mnt/c/Users/Coque/Desktop/astronomy_research/G165/original_small_cutouts/cutout1_wht.fits")[0]

wcs = WCS(img_hdu.header)
image_data = img_hdu.data
weight = wht_hdu.data

output_dir = "/mnt/c/Users/Coque/Desktop/astronomy_research/G165/graphs_output/"


def mask_catalog_with_weight(cat, wcs, weight, wht_min=0.001):

    sky = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    x, y = wcs.world_to_pixel(sky)

    x = np.array(x)
    y = np.array(y)

    # Inside image bounds
    inside = (
        (x >= 0) & (x < weight.shape[1]) &
        (y >= 0) & (y < weight.shape[0])
    )

    # Initialize mask
    good = np.zeros(len(cat), dtype=bool)
    good[inside] = weight[y[inside].astype(int), x[inside].astype(int)] > wht_min

    return cat[good]


def open_catalogs():
    ref = Table.read("./phot_massimo_iso.cat",format="ascii")
    cat = Table.read("./output/photometry_results_small_cutout.csv", format="csv")

    print("Massimo columns:")
    print(ref.colnames)

    print("\nMy catalog columns:")
    print(cat.colnames)

    return ref, cat

def search(ref, cat):
    c_ref = SkyCoord(ra=ref['ra'] * u.deg, dec=ref['dec'] * u.deg)
    c_cat = SkyCoord(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg)

    max_sep = 1 * u.arcsec

    idx_cat, idx_ref, sep2d, _ = c_cat.search_around_sky(c_ref, max_sep)

    good = (idx_cat < len(cat)) & (idx_ref < len(ref))
    idx_cat = idx_cat[good]
    idx_ref = idx_ref[good]
    sep2d = sep2d[good]

    ref_matched = ref[idx_ref]
    cat_matched = cat[idx_cat]

    return max_sep, idx_ref, idx_cat, sep2d, ref_matched, cat_matched

def create_regions_file( idx_ref, idx_cat, ref, cat, max_sep, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    region_filename = os.path.join(output_dir, "matches.reg")

    with open(region_filename, "w") as f:

        f.write("fk5\n")

        # Draw ref sources as circles
        for i in range(len(ref)):
            ra = ref['ra'][i]
            dec = ref['dec'][i]
            #f.write(f"circle({ra},{dec},{max_sep.to(u.arcsec).value}\") # color=red\n")
            f.write(f"circle({ra},{dec},0.1\") # color=red\n")

        # Draw matched cat sources as smaller points inside annuli
        for ref_idx, cat_idx in zip(idx_ref, idx_cat):
            ra = cat['ra'][cat_idx]
            dec = cat['dec'][cat_idx]
            f.write(f"circle({ra},{dec},0.2\") # color=green\n")

    print(f"DS9 region file saved to: {region_filename}")



def plot_aperture_source_flux(ref, cat,ref_flux_col,cat_flux_col,output_dir):

    if ref_flux_col not in ref.colnames:
        raise ValueError(
            f"'{ref_flux_col}' not found in ref catalog. "
            f"Available columns: {ref.colnames}"
        )

    if cat_flux_col not in cat.colnames:
        raise ValueError(
            f"'{cat_flux_col}' not found in cat catalog. "
            f"Available columns: {cat.colnames}"
        )

    ref_flux = ref[ref_flux_col]
    cat_flux = cat[cat_flux_col]

    # Remove non-physical values
    ref_flux = ref_flux[ref_flux > 0]
    cat_flux = cat_flux[cat_flux > 0]

    # Shared logarithmic bins
    bins = np.logspace(
        np.log10(min(ref_flux.min(), cat_flux.min())),
        np.log10(max(ref_flux.max(), cat_flux.max())),
        50
    )

    plt.figure(figsize=(7,5))
    plt.hist(ref_flux, bins=bins, histtype='step', label='Reference catalog')
    plt.hist(cat_flux, bins=bins, histtype='step', label='Comparison catalog')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('mag')
    plt.ylabel('Number of sources')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'aperture_hist.png'), dpi=300, bbox_inches='tight')

def plot_kron_source_flux(ref, cat,ref_flux_col,cat_flux_col,output_dir):

    if ref_flux_col not in ref.colnames:
        raise ValueError(
            f"'{ref_flux_col}' not found in ref catalog. "
            f"Available columns: {ref.colnames}"
        )

    if cat_flux_col not in cat.colnames:
        raise ValueError(
            f"'{cat_flux_col}' not found in cat catalog. "
            f"Available columns: {cat.colnames}"
        )

    ref_flux = ref[ref_flux_col]
    cat_flux = cat[cat_flux_col]

    # Remove non-physical values
    ref_flux = ref_flux[ref_flux > 0]
    cat_flux = cat_flux[cat_flux > 0]

    # Shared logarithmic bins
    bins = np.logspace(
        np.log10(min(ref_flux.min(), cat_flux.min())),
        np.log10(max(ref_flux.max(), cat_flux.max())),
        50
    )

    plt.figure(figsize=(7,5))
    plt.hist(ref_flux, bins=bins, histtype='step', label='Reference catalog')
    plt.hist(cat_flux, bins=bins, histtype='step', label='Comparison catalog')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('mag')
    plt.ylabel('Number of sources')
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'kron_hist.png'), dpi=300, bbox_inches='tight')


def plot_pixel_flux(image_data, output_dir):
    pixels = image_data[np.isfinite(image_data)]
    plt.figure(figsize=(7,5))
    plt.hist(pixels, bins=500, histtype='step')
    plt.yscale('log')
    plt.xlabel('Pixel flux')
    plt.ylabel('Number of pixels')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pixel_flux_histogram.png'), dpi=300, bbox_inches='tight')

def plot_delta_mag_vs_mag(ref, cat, ref_mag_col, cat_mag_col, output_dir):

    if ref_mag_col not in ref.colnames:
        raise ValueError(f"{ref_mag_col} not in reference catalog")

    if cat_mag_col not in cat.colnames:
        raise ValueError(f"{cat_mag_col} not in comparison catalog")

    ref_mag = ref[ref_mag_col]
    cat_mag = cat[cat_mag_col]

    # Delta magnitude = mine - Massimo's
    delta_mag = cat_mag - ref_mag

    # Remove non-finite values
    good = np.isfinite(ref_mag) & np.isfinite(cat_mag)
    ref_mag = ref_mag[good]
    delta_mag = delta_mag[good]

    plt.figure(figsize=(7,5))
    plt.scatter(ref_mag, delta_mag, s=5, alpha=0.5)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)

    plt.xlabel('mag')
    plt.ylabel('Δmag')
    plt.xlim(10, 40)   
    plt.ylim(-10, 20)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'delta_mag_vs_mag.png'), dpi=300, bbox_inches='tight')

def plot_mag_err_vs_mag(cat, mag_col, mag_err_col, output_dir):

    if mag_col not in cat.colnames:
        raise ValueError(f"{mag_col} not in catalog")

    if mag_err_col not in cat.colnames:
        raise ValueError(f"{mag_err_col} not in catalog")

    mag = cat[mag_col]
    mag_err = cat[mag_err_col]

    good = np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    mag = mag[good]
    mag_err = mag_err[good]

    plt.figure(figsize=(7,5))
    plt.scatter(mag, mag_err, s=5, alpha=0.5)

    plt.xlabel('mag')
    plt.ylabel('err(mag)')
    plt.yscale('log')
    plt.xlim(10, 40)   
    plt.tight_layout()

    plt.savefig( os.path.join(output_dir, 'mag_err_vs_mag.png'), dpi=300, bbox_inches='tight')


massimo_cat, my_cat = open_catalogs() # ref, cat

my_cat = mask_catalog_with_weight(my_cat, wcs, weight).copy()
massimo_cat = mask_catalog_with_weight(massimo_cat, wcs, weight).copy()

max_sep, idx_ref, idx_cat, sep2d, ref_matched, cat_matched = search(massimo_cat, my_cat)
create_regions_file( idx_ref, idx_cat, my_cat, massimo_cat, max_sep, output_dir)

plot_aperture_source_flux(ref=massimo_cat,cat=my_cat, ref_flux_col='f090w', cat_flux_col='ab_aperture_mag', output_dir=output_dir)
plot_kron_source_flux(ref=massimo_cat,cat=my_cat, ref_flux_col='f090w', cat_flux_col='ab_kron_mag', output_dir=output_dir)

# plot_pixel_flux(image_data, output_dir)

plot_delta_mag_vs_mag(ref=ref_matched, cat=cat_matched, ref_mag_col='f090w', cat_mag_col='ab_aperture_mag',output_dir=output_dir)


plot_mag_err_vs_mag(cat=cat_matched, mag_col='ab_aperture_mag', mag_err_col='ab_aperture_mag_err',  output_dir=output_dir)



