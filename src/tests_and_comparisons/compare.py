import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky


# Single Band:
# ------------


img_hdu = fits.open("/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f200w_30mas_20230403_drz.fits")[0]
wht_hdu = fits.open("/mnt/c/Users/Coque/Downloads/mosaic_plckg165_nircam_f200w_30mas_20230403_wht.fits")[0]

wcs = WCS(img_hdu.header)
image_data = img_hdu.data
weight = wht_hdu.data

output_dir = os.path.expanduser("~/G165/comparison_output/")


def mask_catalog_with_weight(cat, wcs, weight, wht_min=0.001):

    sky = SkyCoord(cat['ra'], cat['dec'], unit='deg')
    x, y = wcs.world_to_pixel(sky)
    x = np.array(x)
    y = np.array(y)

    # Inside image bounds
    inside = ( (x >= 0) & (x < weight.shape[1]) & (y >= 0) & (y < weight.shape[0]) )

    # Initialize mask
    good = np.zeros(len(cat), dtype=bool)
    good[inside] = weight[y[inside].astype(int), x[inside].astype(int)] > wht_min

    return cat[good]


def open_catalogs():
    ref = Table.read("./phot_massimo_iso.cat",format="ascii")
    cat = Table.read("./output/gini_test/photometry_results_f200.csv", format="csv")
    # cat = Table.read("./output/f_200_deblending_on/photometry_results_f200.csv", format="csv")

    print("Massimo columns:")
    print(ref.colnames)

    print("\nMy catalog columns:")
    print(cat.colnames)

    print("Filtering Bad Magnitudes:")
    print(f"    Massimo catalog before filtering: {len(ref)} sources")
    print(f"    My catalog before filtering: {len(cat)} sources")
    
    # Filter ref for reasonable Kron magnitudes
    good_ref = (ref['f200w'] > 0) & (ref['f200w'] < 27)
    ref = ref[good_ref]
    print(f"Massimo catalog after filtering: {len(ref)} sources")
    print(f"  (Removed {(~good_ref).sum()} sources with f200w outside 0-27 range))")
    
    # Filter cat for reasonable Kron magnitudes
    good_cat = (cat['ab_kron_mag'] > 0) & (cat['ab_kron_mag'] < 27)
    cat = cat[good_cat]
    print(f"My catalog after filtering: {len(cat)} sources")
    print(f"  (Removed {(~good_cat).sum()} sources with ab_kron_mag outside 0-27 range)")

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


def search(ref, cat, ref_mag_col='f200w', cat_mag_col='ab_kron_mag'):
    c_ref = SkyCoord(ra=ref['ra'] * u.deg, dec=ref['dec'] * u.deg)
    c_cat = SkyCoord(ra=cat['ra'] * u.deg, dec=cat['dec'] * u.deg)

    # Nearest neighbor matching
    idx, sep2d, _ = match_coordinates_sky(c_cat, c_ref)

    # Use the average magnitude of both catalogs
    ref_mag = ref[ref_mag_col][idx]
    cat_mag = cat[cat_mag_col]
    avg_mag = 0.5 * (ref_mag + cat_mag)

    # Compute tiered search radius
    dynamic_radius = compute_search_radius_tiered(avg_mag)

    print("\nMatching with tiered search radius")
    for r in np.unique(dynamic_radius):
        print(f"  Radius used: {r}")

    # Apply radius criteria
    good_matches = sep2d < dynamic_radius

    idx_cat = np.where(good_matches)[0]
    idx_ref = idx[good_matches]
    sep2d_matched = sep2d[good_matches]

    ref_matched = ref[idx_ref]
    cat_matched = cat[idx_cat]

    print(f"  Matches found: {len(idx_cat)}")
    print(f"  Median separation: {np.median(sep2d_matched).to(u.arcsec):.3f}")

    return dynamic_radius, idx_ref, idx_cat, sep2d_matched, ref_matched, cat_matched

def create_regions_file(idx_ref, idx_cat, ref, cat, max_sep, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    region_filename = os.path.join(output_dir, "matches.reg")
 
    with open(region_filename, "w") as f:
        f.write("fk5\n")
        f.write("# Catalog Matching Visualization\n")
        f.write("# Red = ref (Massimo's) catalog\n")
        f.write("# Green = cat (my) catalog\n")
        f.write("# Orange lines = Matched pairs\n\n")
 
        # Orange lines connect matched pairs
        # (Draw lines first so circles appear on top)
        for ref_idx, cat_idx in zip(idx_ref, idx_cat):
            ra_ref = ref['ra'][ref_idx]
            dec_ref = ref['dec'][ref_idx]
            ra_cat = cat['ra'][cat_idx]
            dec_cat = cat['dec'][cat_idx]
            f.write(f"line({ra_ref},{dec_ref},{ra_cat},{dec_cat}) # color=orange width=3\n")
        
        # Reference catalouge's sources appear as red circles
        for i in range(len(ref)):
            ra = ref['ra'][i]
            dec = ref['dec'][i]
            f.write(f"circle({ra},{dec},0.1\") # color=red width=1\n")
 
        # Compparison catalouge's matched sources appear as green circles
        for ref_idx, cat_idx in zip(idx_ref, idx_cat):
            ra = cat['ra'][cat_idx]
            dec = cat['dec'][cat_idx]
            f.write(f"circle({ra},{dec},0.15\") # color=#00CC00 width=2\n")
 
    print(f"DS9 region file saved to: {region_filename}")
    print(f"  Red circles: {len(ref)} sources from ref catalog")
    print(f"  Green circles: {len(idx_cat)} matched sources from cat catalog")
    print(f"  Orange lines: {len(idx_cat)} connections between matched pairs")



def plot_kron_source_flux(ref, cat,ref_flux_col,cat_flux_col,output_dir):

    if ref_flux_col not in ref.colnames:
        raise ValueError( f"'{ref_flux_col}' not found in ref catalog. Available columns: {ref.colnames}" )

    if cat_flux_col not in cat.colnames:
        raise ValueError( f"'{cat_flux_col}' not found in cat catalog. Available columns: {cat.colnames}" )

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
    plt.xlim(17, 28)
    plt.ylim(-2.5, 2.5)
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

    
def analyze_worst_outliers(ref_matched, cat_matched, ref_mag_col, cat_mag_col, output_dir, delta_mag_threshold=1.75):
  
    # Calculate delta mag
    ref_mag = ref_matched[ref_mag_col]
    cat_mag = cat_matched[cat_mag_col]
    delta_mag = cat_mag - ref_mag
    
    # Filter valid values
    valid = np.isfinite(ref_mag) & np.isfinite(delta_mag)
    outlier_mask = valid & (np.abs(delta_mag) > delta_mag_threshold)

    outlier_indices = np.where(outlier_mask)[0]
    n_outliers = len(outlier_indices)
    
    if n_outliers == 0:
        print(f"No outliers found with |Δmag| > {delta_mag_threshold:.1f}")
        return None
    
    # Sort outliers by absolute delta mag (worst first)
    abs_delta_outliers = np.abs(delta_mag[outlier_mask])
    sorted_order = np.argsort(abs_delta_outliers)[::-1]  # Descending
    worst_indices = outlier_indices[sorted_order]
    
    # Create output table
    outlier_table = Table()
    outlier_table['rank'] = np.arange(1, len(worst_indices) + 1)
    outlier_table['ra'] = cat_matched['ra'][worst_indices]
    outlier_table['dec'] = cat_matched['dec'][worst_indices]
    outlier_table['ref_mag'] = ref_mag[worst_indices]
    outlier_table['my_mag'] = cat_mag[worst_indices]
    outlier_table['delta_mag'] = delta_mag[worst_indices]
    outlier_table['abs_delta_mag'] = np.abs(delta_mag[worst_indices])
    
    # Add extra info if available
    if 'kron_flux' in cat_matched.colnames:
        outlier_table['kron_flux'] = cat_matched['kron_flux'][worst_indices]
    if 'ab_aperture_mag' in cat_matched.colnames:
        outlier_table['aperture_mag'] = cat_matched['ab_aperture_mag'][worst_indices]
    
    # Print summary
    print(f"Outliers with |Δmag| > {delta_mag_threshold:.1f}")
    print(f"Total outliers found: {n_outliers}")
    print(f"  Too faint (Δmag > +{delta_mag_threshold:.1f}): {(delta_mag[outlier_mask] > delta_mag_threshold).sum()}")
    print(f"  Too bright (Δmag < -{delta_mag_threshold:.1f}): {(delta_mag[outlier_mask] < -delta_mag_threshold).sum()}")
    print(f"\nTop {min(20, n_outliers)} worst outliers:")
    print(f"{'Rank':>5s} {'RA':>12s} {'Dec':>12s} {'Ref Mag':>9s} {'My Mag':>9s} {'Δmag':>9s}")
    
    
    # Print up to 20 worst outliers
    for i, row in enumerate(outlier_table[:20]):
        print(f"{row['rank']:5d} {row['ra']:12.6f} {row['dec']:12.6f} "
              f"{row['ref_mag']:9.2f} {row['my_mag']:9.2f} {row['delta_mag']:+9.2f}")
    
    if n_outliers > 20:
        print(f"... and {n_outliers - 20} more outliers")
    
    # Save to CSV
    outlier_file = os.path.join(output_dir, 'outliers.csv')
    outlier_table.write(outlier_file, format='csv', overwrite=True)
    print(f"\nOutlier table saved to: {outlier_file}")

    return outlier_table


def create_outlier_regions_file(outlier_table, output_dir, region_name='worst_outliers.reg'):
    
    # Creates a DS9 region file highlighting the worst outliers.
       
    region_file = os.path.join(output_dir, region_name)
    
    with open(region_file, 'w') as f:
        f.write("fk5\n")
        f.write("# Worst Photometric Outliers\n")
        f.write("# Red = Too Faint, Blue = Too Bright\n")
        f.write("# Circle size = severity (larger = worse)\n\n")
        
        for row in outlier_table:
            ra = row['ra']
            dec = row['dec']
            delta = row['delta_mag']
            rank = row['rank']
            
            # Color based on direction of offset
            if delta > 0:
                color = "red"    # I measured fainter
                label = "FAINT"
            else:
                color = "cyan"   # I measured brighter
                label = "BRIGHT"
            
            # Size based on severity (larger = worse)
            # Scale from 0.5" to 2.0" based on |delta|
            abs_delta = abs(delta)
            size = min(0.5 + abs_delta * 0.15, 2.0)  # Cap at 2"
            
            # Draw circle
            f.write(f"circle({ra:.6f},{dec:.6f},{size:.2f}\") # color={color} width=1\n")
            
    
    print(f"Outlier region file saved to: {region_file}")
    return region_file


def plot_outliers_on_image(outlier_table, ref_matched, cat_matched, output_dir):
    
    #Creates a plot showing where outliers are located on the image.
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get all matched sources
    all_ra = cat_matched['ra']
    all_dec = cat_matched['dec']
    
    ax = axes[0]
    
    # Plots all sources as small gray dots
    ax.scatter(all_ra, all_dec, s=1, c='lightgray', alpha=0.5, label='All sources')
    
    # Plots outliers with color coding
    for row in outlier_table:
        if row['delta_mag'] > 0:
            color = 'red'
            marker = 'o'
        else:
            color = 'cyan'
            marker = 's'
        
        # Size proportional to severity
        size = 50 + abs(row['delta_mag']) * 20
        
        ax.scatter(row['ra'], row['dec'], s=size, c=color, marker=marker,
                  edgecolors='black', linewidths=1, alpha=0.7,
                  label=f"#{row['rank']}: Δ={row['delta_mag']:.1f}" if row['rank'] <= 5 else "")
    
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title('Spatial Distribution of Worst Outliers')
    ax.invert_xaxis()  # RA increases to the left
    
    # Add legend for top 5 only
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:6], labels[:6], fontsize=8, loc='best')
    
    ax.grid(True, alpha=0.3)
    
    # Delta mag vs magnitude for outliers
    ax = axes[1]
    
    # Plot all sources
    ref_mag_all = ref_matched['f200w']
    cat_mag_all = cat_matched['ab_kron_mag'] if 'ab_kron_mag' in cat_matched.colnames else cat_matched['ab_aperture_mag']
    delta_mag_all = cat_mag_all - ref_mag_all
    
    valid_all = np.isfinite(ref_mag_all) & np.isfinite(delta_mag_all)
    
    ax.scatter(ref_mag_all[valid_all], delta_mag_all[valid_all], 
              s=3, c='lightgray', alpha=0.3, label='All sources')
    
    # Highlight outliers
    for row in outlier_table:
        if row['delta_mag'] > 0:
            color = 'red'
            marker = 'o'
        else:
            color = 'cyan'
            marker = 's'
        
        size = 100 + abs(row['delta_mag']) * 10
        
        ax.scatter(row['ref_mag'], row['delta_mag'], s=size, c=color, marker=marker,
                  edgecolors='black', linewidths=2, alpha=0.8, zorder=10)
        
        
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Massimo mag (f200w)')
    ax.set_ylabel('Δmag (mine - Massimo)')
    ax.set_title('Outliers in Magnitude Comparison')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(18, 32)
    ax.set_ylim(-10, 10)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'worst_outliers_visualization.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Outlier visualization saved to: {plot_file}")
    plt.close()
    

massimo_cat, my_cat = open_catalogs() # ref, cat

my_cat = mask_catalog_with_weight(my_cat, wcs, weight).copy()
massimo_cat = mask_catalog_with_weight(massimo_cat, wcs, weight).copy()

max_sep, idx_ref, idx_cat, sep2d, ref_matched, cat_matched = search(massimo_cat, my_cat)

create_regions_file(idx_ref, idx_cat, massimo_cat, my_cat, max_sep, output_dir)


#plot_aperture_source_flux(ref=massimo_cat,cat=my_cat, ref_flux_col='f200w', cat_flux_col='ab_aperture_mag', output_dir=output_dir)
plot_kron_source_flux(ref=massimo_cat,cat=my_cat, ref_flux_col='f200w', cat_flux_col='ab_kron_mag', output_dir=output_dir)

# plot_pixel_flux(image_data, output_dir)

plot_delta_mag_vs_mag(ref=ref_matched, cat=cat_matched, ref_mag_col='f200w', cat_mag_col='ab_kron_mag',output_dir=output_dir)

# plot_mag_err_vs_mag(cat=cat_matched, mag_col='ab_aperture_mag', mag_err_col='ab_aperture_mag_err',  output_dir=output_dir)


# Analyze worst outliers
if len(ref_matched) > 0 and len(cat_matched) > 0:
   
    
    outlier_table = analyze_worst_outliers(
        ref_matched, cat_matched,
        ref_mag_col='f200w',
        cat_mag_col='ab_kron_mag',
        output_dir=output_dir,
        delta_mag_threshold=1.6 )
    
    # Create region file for outliers
    create_outlier_regions_file(outlier_table, output_dir)
    
    # Create visualization plot
    plot_outliers_on_image(outlier_table, ref_matched, cat_matched, output_dir)
    

    print("Outlier analysis complete")




