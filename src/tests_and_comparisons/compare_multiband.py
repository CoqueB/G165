
# Multiband:


import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky


bands = ["f090", "f115", "f150", "f200", "f277", "f356", "f410", "f444"]

image_files = {
    "f090": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f090w_30mas_20230403_drz.fits",
    "f115": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f115w_30mas_20230403_drz.fits",
    "f150": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f150w_30mas_20230403_drz.fits",
    "f200": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f200w_30mas_20230403_drz.fits",
    "f277": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f277w_30mas_20230403_drz.fits",
    "f356": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f356w_30mas_20230403_drz.fits",
    "f410": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f410m_30mas_20230403_drz.fits",
    "f444": "/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f444w_30mas_20230403_drz.fits"}

weight_files = { b: image_files[b].replace("_drz.fits", "_wht.fits") for b in bands }

base_output_dir = os.path.expanduser("~/G165/multiband_output/")


def mask_catalog_with_weight(cat, wcs, weight, wht_min=0.001):

    sky = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    x, y = wcs.world_to_pixel(sky)
    x = np.array(x)
    y = np.array(y)

    # Inside image bounds
    inside = (
        (x >= 0) & (x < weight.shape[1]) &
        (y >= 0) & (y < weight.shape[0]) )

    good = np.zeros(len(cat), dtype=bool)
    good[inside] = weight[y[inside].astype(int), x[inside].astype(int)] > wht_min
    return cat[good]


def open_catalogs(band, ref_mag_col):
    ref = Table.read("./phot_massimo_iso.cat", format="ascii")
    cat = Table.read(f"./output/results_for_multiband/photometry_results_{band}.csv", format="csv")

    print("ref columns:")
    print(ref.colnames)

    print("\ncat columns:")
    print(cat.colnames)

    print("Filtering Bad Magnitudes:")
    print(f"    ref before filtering: {len(ref)} sources")
    print(f"    cat before filtering: {len(cat)} sources")

    # Filter ref for reasonable magnitudes 
    good_ref = (ref[ref_mag_col] > 0) & (ref[ref_mag_col] <  28.54) # 28.54 comes from Frye +24
    ref = ref[good_ref]
    print(f"ref after filtering: {len(ref)} sources")
    print(f"  (Removed {(~good_ref).sum()} sources with {ref_mag_col} outside 0-28.54 range)")

    # Filter cat for reasonable Kron magnitudes
    good_cat = (cat['ab_kron_mag'] > 0) & (cat['ab_kron_mag'] <  28.54) # 28.54 comes from Frye +24
    cat = cat[good_cat]
    print(f"cat after filtering: {len(cat)} sources")
    print(f"  (Removed {(~good_cat).sum()} sources with ab_kron_mag outside 0-28.54 range)")

    return ref, cat


def compute_search_radius_tiered(mag):
    mag = np.array(mag)
    radius = np.zeros_like(mag, dtype=float)

    # Brightest galaxies
    radius[mag <= 21] = 0.25

    # Intermediate magnitudes
    radius[(mag >= 21) & (mag < 25)] = 0.12

    # Faint galaxies
    radius[mag >= 25] = 0.09

    return radius * u.arcsec


def search(ref, cat, ref_mag_col, cat_mag_col='ab_kron_mag'):
    c_ref = SkyCoord(ra=ref['ra'] * u.deg, dec=ref['dec'] * u.deg)
    c_cat = SkyCoord(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg)

    # Nearest neighbor matching
    idx, sep2d, _ = match_coordinates_sky(c_cat, c_ref)

    # Use the average magnitude of both catalogs
    ref_mag = ref[ref_mag_col][idx]
    cat_mag = cat[cat_mag_col]
    avg_mag = 0.5 * (ref_mag + cat_mag)

    # Computing tiered search radius
    dynamic_radius = compute_search_radius_tiered(avg_mag)

    print("\nMatching with tiered search radius")
    for r in np.unique(dynamic_radius):
        print(f"  Radius used: {r}")

    # Radius criteria
    good_matches = sep2d < dynamic_radius

    idx_cat = np.where(good_matches)[0]
    idx_ref = idx[good_matches]
    sep2d_matched = sep2d[good_matches]

    ref_matched = ref[idx_ref]
    cat_matched = cat[idx_cat]

    print(f"  Matches found: {len(idx_cat)}")
    if len(idx_cat) > 0:
        print(f"  Median separation: {np.median(sep2d_matched).to(u.arcsec):.3f}")

    return dynamic_radius, idx_ref, idx_cat, sep2d_matched, ref_matched, cat_matched


def create_regions_file(idx_ref, idx_cat, ref, cat, max_sep, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    region_filename = os.path.join(output_dir, "matches.reg")

    with open(region_filename, "w") as f:
        f.write("fk5\n")
        f.write("# red = ref, green = cat, orange = match lines\n\n")

        # Orange lines connect matched pairs
        for ref_idx, cat_idx in zip(idx_ref, idx_cat):
            ra_ref = ref['ra'][ref_idx]
            dec_ref = ref['dec'][ref_idx]
            ra_cat = cat['ra'][cat_idx]
            dec_cat = cat['dec'][cat_idx]
            f.write(f"line({ra_ref},{dec_ref},{ra_cat},{dec_cat}) # color=orange width=3\n")

        # Reference catalogue's sources appear as red circles
        for i in range(len(ref)):
            ra = ref['ra'][i]
            dec = ref['dec'][i]
            f.write(f"circle({ra},{dec},0.1\") # color=red width=1\n")

        # Comparison catalogue's matched sources appear as green circles
        for ref_idx, cat_idx in zip(idx_ref, idx_cat):
            ra = cat['ra'][cat_idx]
            dec = cat['dec'][cat_idx]
            f.write(f"circle({ra},{dec},0.15\") # color=#00CC00 width=2\n")

    print(f"DS9 region file saved to: {region_filename}")
    print(f"  Red circles: {len(ref)} sources from ref catalog")
    print(f"  Green circles: {len(idx_cat)} matched sources from cat catalog")
    print(f"  Orange lines: {len(idx_cat)} connections between matched pairs")


def plot_kron_source_flux(ref, cat, ref_flux_col, cat_flux_col, output_dir, band):

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

    plt.savefig(os.path.join(output_dir, f'{band}_kron_hist.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_delta_mag_vs_mag(ref, cat, ref_mag_col, cat_mag_col, output_dir, band):

    if ref_mag_col not in ref.colnames:
        raise ValueError(f"{ref_mag_col} not in reference catalog")

    if cat_mag_col not in cat.colnames:
        raise ValueError(f"{cat_mag_col} not in comparison catalog")

    ref_mag = ref[ref_mag_col]
    cat_mag = cat[cat_mag_col]

    # Delta magnitude = mine - Massimo's   ,so (cat -ref)
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
    plt.xlim(17, 28)
    plt.ylim(-2.5, 2.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f'{band}_delta_mag_vs_mag.png'), dpi=300, bbox_inches='tight')
    plt.close()


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
    plt.xlim(10, 40)
    plt.tight_layout()

    plt.savefig( os.path.join(output_dir, f'{band}_mag_err_vs_mag.png'), dpi=300, bbox_inches='tight')
    plt.close()


def merge_multiband_catalog(bands, max_sep=0.05*u.arcsec, mag_col='ab_kron_mag'):
    """
    Merges per-band photometry CSVs into one catalog, matched by
    coordinates. Uses the first available band as the positional anchor,
    then cross-matches every other band onto it and adds its magnitude
    as a new column named ab_kron_mag_{band}.
    """

    catalog_dir = os.path.expanduser("~/G165/output/results_for_multiband")

    # Load whichever bands actually have a CSV
    per_band_cat = {}
    for band in bands:
        cat_file = os.path.join(catalog_dir, f"photometry_results_{band}.csv")
        if not os.path.exists(cat_file):
            print(f"  [merge] Skipping {band}: {cat_file} not found")
            continue
        per_band_cat[band] = Table.read(cat_file, format="csv")

    if len(per_band_cat) == 0:
        raise FileNotFoundError(f"No per-band catalogs found in {catalog_dir}")

    available_bands = list(per_band_cat.keys())
    anchor_band = "f200"
    if anchor_band not in per_band_cat:
        raise ValueError(f"f200 catalog not found in {catalog_dir} — cannot use it as anchor")
    print(f"\n[merge] Using {anchor_band} as positional anchor ({len(per_band_cat[anchor_band])} sources)")

    # Build the multiband table from the anchor band's positions
    multiband = Table()
    multiband['ra'] = per_band_cat[anchor_band]['ra']
    multiband['dec'] = per_band_cat[anchor_band]['dec']
    multiband[f'ab_kron_mag_{anchor_band}'] = per_band_cat[anchor_band][mag_col]

    c_anchor = SkyCoord(ra=multiband['ra'] * u.deg, dec=multiband['dec'] * u.deg)

    # Cross-match every other band onto the anchor's positions
    for band in [b for b in available_bands if b != anchor_band]:
        cat = per_band_cat[band]
        c_band = SkyCoord(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg)

        idx, sep2d, _ = match_coordinates_sky(c_anchor, c_band)
        good = sep2d < max_sep

        col = np.full(len(multiband), np.nan)
        col[good] = cat[mag_col][idx[good]]
        multiband[f'ab_kron_mag_{band}'] = col

        print(f"  [merge] {band}: matched {good.sum()}/{len(multiband)} anchor sources "
              f"within {max_sep}")
        
    file_name = 'multiband_catalouge.csv'
    multiband.write(os.path.join(base_output_dir, file_name),format='csv', overwrite=True)
    print("Produced multiband catalouge with", len(multiband),"sources")
    return multiband


def plot_color_color_overlaid(ref, merged_cat, output_dir):
    """
    Color-color diagrams with both catalogs overplotted:
        black = reference (Massimo)
        red   = my catalog
    """

    panels = [
        (('f090', 'f115'), ('f115', 'f150')),
        (('f115', 'f150'), ('f150', 'f200')),
        (('f150', 'f200'), ('f200', 'f277')),
        (('f150', 'f200'), ('f277', 'f356')),
        (('f150', 'f200'), ('f356', 'f444')),
        (('f200', 'f277'), ('f356', 'f444')),
        (('f277', 'f356'), ('f356', 'f444')),
        (('f115', 'f150'), ('f200', 'f277')),
        (('f115', 'f444'), ('f200', 'f444')),
        (('f150', 'f200'), ('f277', 'f444')),]

    fig, axes = plt.subplots(5, 2, figsize=(11, 21))
    axes = axes.flatten()

    for i, (ycolor, xcolor) in enumerate(panels):
        ax = axes[i]

        # reference catalog
        ref_cols = [
            xcolor[0] + "w",
            xcolor[1] + "w",
            ycolor[0] + "w",
            ycolor[1] + "w"]

        # f410 is a medium band
        ref_cols = [c.replace("f410w", "f410m") for c in ref_cols]

        if all(c in ref.colnames for c in ref_cols):
            y_ref = ref[ref_cols[2]] - ref[ref_cols[3]]
            x_ref = ref[ref_cols[0]] - ref[ref_cols[1]]

            good = np.isfinite(x_ref) & np.isfinite(y_ref)
            ax.plot(
                x_ref[good],
                y_ref[good],
                '.',
                color='black',
                ms=1.5,
                alpha=0.5,
                label='Reference' if i == 0 else None )

        # my catalog
        cat_cols = [
            f'ab_kron_mag_{xcolor[0]}',
            f'ab_kron_mag_{xcolor[1]}',
            f'ab_kron_mag_{ycolor[0]}',
            f'ab_kron_mag_{ycolor[1]}']

        if all(c in merged_cat.colnames for c in cat_cols):
            y_cat = merged_cat[cat_cols[2]] - merged_cat[cat_cols[3]]
            x_cat = merged_cat[cat_cols[0]] - merged_cat[cat_cols[1]]

            good = np.isfinite(x_cat) & np.isfinite(y_cat)
            ax.plot(
                x_cat[good],
                y_cat[good],
                '.',
                color='red',
                ms=1.5,
                alpha=0.5,
                label='My catalog' if i == 0 else None)

        ax.set_xlabel(f"{xcolor[0].upper()}W-{xcolor[1].upper()}W")
        ax.set_ylabel(f"{ycolor[0].upper()}W-{ycolor[1].upper()}W")

        """
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        """

    axes[0].legend(markerscale=5)

    plt.tight_layout()
    outfile = os.path.join(output_dir,'color_color_diagrams_overlaid.png')
    fig.savefig(outfile, dpi=200)
    print(f"Overlaid color-color grid saved to {outfile}")
    plt.close(fig)

def plot_cmd_f200_overlaid(ref, merged_cat, output_dir):

    ref_mag = ref['f200w']
    ref_color = ref['f200w'] - ref['f277w']

    cat_mag = merged_cat['ab_kron_mag_f200']
    cat_color = (
        merged_cat['ab_kron_mag_f200']
        - merged_cat['ab_kron_mag_f277'])

    good_ref = np.isfinite(ref_mag) & np.isfinite(ref_color)
    good_cat = np.isfinite(cat_mag) & np.isfinite(cat_color)

    plt.figure(figsize=(7,7))

    plt.plot( ref_color[good_ref], ref_mag[good_ref],'.', color='black', ms=2, alpha=0.5, label='Reference')

    plt.plot( cat_color[good_cat], cat_mag[good_cat], '.', color='red', ms=2,  alpha=0.5, label='My catalog')

    plt.xlabel('F200W - F277W')
    plt.ylabel('F200W')

    plt.gca().invert_yaxis()

    plt.xlim(-1, 1.5)
    plt.ylim(30, 15)

    plt.legend()
    plt.tight_layout()

    outfile = os.path.join(output_dir, 'color_mag_f200_overlaid.png')

    plt.savefig(outfile, dpi=200)
    plt.close()

    print(f"color-mag f200 saved to {outfile}")



def plot_cmd_f277_overlaid(ref, merged_cat, output_dir):

    ref_mag = ref['f277w']
    ref_color = ref['f200w'] - ref['f277w']

    cat_mag = merged_cat['ab_kron_mag_f277']
    cat_color = (
        merged_cat['ab_kron_mag_f200']
        - merged_cat['ab_kron_mag_f277']  )

    good_ref = np.isfinite(ref_mag) & np.isfinite(ref_color)
    good_cat = np.isfinite(cat_mag) & np.isfinite(cat_color)

    plt.figure(figsize=(7,7))

    plt.plot(ref_color[good_ref], ref_mag[good_ref], '.', color='black', ms=2, alpha=0.5, label='Reference')

    plt.plot(cat_color[good_cat], cat_mag[good_cat], '.', color='red', ms=2, alpha=0.5, label='My catalog')

    plt.xlabel('F200W - F277W')
    plt.ylabel('F277W')

    plt.gca().invert_yaxis()

    plt.xlim(-1, 1.5)
    plt.ylim(30, 15)

    plt.legend()
    plt.tight_layout()

    outfile = os.path.join(
        output_dir,
        'color_mag_f277_overlaid.png')

    plt.savefig(outfile, dpi=200)
    plt.close()

    print(f"color-mag f277 saved to {outfile}")


# Main loop: process each band exactly like compare.py does

# Store matched catalogs from each band here, so we can build
# combined ref/cat tables for the color-color diagrams at the end
ref_matched_per_band = {}
cat_matched_per_band = {}

for band in bands:
    print(f"\n Processing {band} ")

    # Column names (f410 is a medium band so "m" suffix in Massimo's catalog)
    ref_mag_col = band + "m" if band == "f410" else band + "w"
    cat_kron_col = "ab_kron_mag"

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
    massimo_cat, my_cat = open_catalogs(band, ref_mag_col)

    # Mask using weight map
    my_cat = mask_catalog_with_weight(my_cat, wcs, weight).copy()
    massimo_cat = mask_catalog_with_weight(massimo_cat, wcs, weight).copy()

    # Cross-match with tiered search radius
    max_sep, idx_ref, idx_cat, sep2d, ref_matched, cat_matched = search(massimo_cat, my_cat, ref_mag_col=ref_mag_col, cat_mag_col=cat_kron_col )

    # Regions file with orange match lines
    create_regions_file(idx_ref, idx_cat, massimo_cat, my_cat, max_sep, output_dir)

    # Histograms
    plot_kron_source_flux(ref=massimo_cat, cat=my_cat, ref_flux_col=ref_mag_col, cat_flux_col=cat_kron_col, output_dir=output_dir, band=band)

    # Delta mag vs mag
    plot_delta_mag_vs_mag(ref=ref_matched, cat=cat_matched, ref_mag_col=ref_mag_col, cat_mag_col=cat_kron_col, output_dir=output_dir, band=band)

    # plot_mag_err_vs_mag(cat=cat_matched, mag_col=cat_ap_col, mag_err_col='ab_aperture_mag_err', output_dir=output_dir, band=band)


    # Store matched catalogs for the color-color diagrams below
    # This produces 8 pairs of matched catalogs for the multi-band analysis

    ref_matched_per_band[band] = ref_matched
    cat_matched_per_band[band] = cat_matched


# Color-color diagrams

# Use f200 matched catalog as the reference source for all overlay plots
base_band = "f200"

try:
    merged_cat = merge_multiband_catalog(bands)

    # overplot both catalogs
    plot_color_color_overlaid( ref_matched_per_band[base_band], merged_cat, base_output_dir)
    
    # Color magnitude diagrams
    plot_cmd_f200_overlaid(ref_matched_per_band[base_band], merged_cat, base_output_dir)
    plot_cmd_f277_overlaid(ref_matched_per_band[base_band], merged_cat, base_output_dir)

except FileNotFoundError as e:
    print(f"\n[color-color] Skipping multiband catalog diagrams: {e}")
