

# Takes in catalogues of cluster members
# keeps only sources with 3.25 <= zspec <= 3.75, cross-matches their positions against
# Massimo's reference catalog and my multiband catalog, and overlays them on the
# F200W vs (F200W-F277W) color-magnitude diagram produced by compare_multiband.py

import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, match_coordinates_sky


z_min, z_max = 0.325, 0.375
max_sep = 0.3 * u.arcsec
redshift_catalogs_dir = "/mnt/c/Users/Coque/Downloads/"

redshift_catalogs = {
    # "arclets": os.path.join(redshift_catalogs_dir, "arclets.csv"),
    # "field_members": os.path.join(redshift_catalogs_dir, "field_members.csv"),
    "spectra_prelim": os.path.join(redshift_catalogs_dir, "catalog_G165_spectra_prelim_24Aug22.txt"),
    # "sdss_0.35": os.path.join(redshift_catalogs_dir, "Redshift035Galaxies.txt"),
}

# Massimo's reference catalog (same file compare_multiband.py reads)
ref_path= "./phot_massimo_iso.cat"

# My merged multiband catalog. Comes from merge_multiband_catalog() in compare_multiband.py
cat_path = os.path.expanduser("~/G165/multiband_output/multiband_catalouge.csv")


output_dir = os.path.expanduser("~/G165/multiband_output/")
output_file = os.path.join(output_dir, "color_mag_f200_highz_overlay.png")

# Temporary masking: plot_ids known to fall outside the mosaic footprint
masked_plot_ids = {8, 18, 49, 50, 59, 86, 89}

# Duplicate sources

duplicate_plot_ids = {23, 62, 43, 84, 54, 67, 88, 83,92, 94, 96, 98, 102, 104, 106, 108}

def load_csv_zcat(path, catalog_name):
    """arclets.csv / field_members.csv: comma-separated, has 'ra','dec','zspec'."""
    t = Table.read(path, format="csv")

    id_col = "#id" if "#id" in t.colnames else t.colnames[0]

    zspec = np.asarray(t["zspec"], dtype=float)
    good = np.isfinite(zspec) & (zspec >= z_min) & (zspec <= z_max)
    sel = t[good]

    out = Table()
    out["id"] = [str(v) for v in sel[id_col]]
    out["ra"] = np.asarray(sel["ra"], dtype=float)
    out["dec"] = np.asarray(sel["dec"], dtype=float)
    out["zspec"] = np.asarray(sel["zspec"], dtype=float)
    out["source_catalog"] = [catalog_name] * len(sel)
    return out


def load_txt_zcat(path, catalog_name):
    """catalog_G165_spectra_prelim_24Aug22.txt: whitespace-separated, no header,
    columns are id, ra, dec, zspec."""
    t = Table.read(path, format="ascii", names=["id", "ra", "dec", "zspec"])

    zspec = np.asarray(t["zspec"], dtype=float)
    good = np.isfinite(zspec) & (zspec >= z_min) & (zspec <= z_max)
    sel = t[good]

    out = Table()
    out["id"] = [str(v) for v in sel["id"]]
    out["ra"] = np.asarray(sel["ra"], dtype=float)
    out["dec"] = np.asarray(sel["dec"], dtype=float)
    out["zspec"] = np.asarray(sel["zspec"], dtype=float)
    out["source_catalog"] = [catalog_name] * len(sel)
    return out


def load_sexagesimal_csv_zcat(path, catalog_name):
    """Redshift__0_35_Galaxies.txt: comma-separated, no header,
    columns are id, ra (hms), dec (dms), zspec, mag."""
    ids, ras, decs, zs = [], [], [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            sid, ra_str, dec_str, z_str = parts[0], parts[1], parts[2], parts[3]

            try:
                z = float(z_str)
            except ValueError:
                continue

            if not (z_min <= z <= z_max):
                continue

            coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.hourangle, u.deg))
            ids.append(sid)
            ras.append(coord.ra.deg)
            decs.append(coord.dec.deg)
            zs.append(z)

    out = Table()
    out["id"] = ids
    out["ra"] = ras
    out["dec"] = decs
    out["zspec"] = zs
    out["source_catalog"] = [catalog_name] * len(ids)
    return out


def build_highz_catalog():
    """Loads all four redshift catalogues and stacks the sources with
    z_min <= zspec <= z_max into a single table."""

    loaders = {
        "arclets": load_csv_zcat,
        "field_members": load_csv_zcat,
        "spectra_prelim": load_txt_zcat,
        "sdss_0.35": load_sexagesimal_csv_zcat }
     

    pieces = []
    for name, path in redshift_catalogs.items():
        if not os.path.exists(path):
            print(f"  [highz] Skipping '{name}': {path} not found")
            continue

        loader = loaders[name]
        t = loader(path, name)
        print(f"  [highz] {name}: {len(t)} sources with {z_min} <= z <= {z_max}")
        if len(t) > 0:
            pieces.append(t)

    if len(pieces) == 0:
        raise RuntimeError(
            f"No sources with {z_min} <= zspec <= {z_max} found in any of the four catalogues.")

    highz = vstack(pieces, metadata_conflicts="silent")
    highz['plot_id'] = np.arange(1, len(highz) + 1)

    drop_ids = masked_plot_ids | duplicate_plot_ids
    if drop_ids:
        keep = ~np.isin(highz['plot_id'], list(drop_ids))
        print(f"[highz] Masking {(~keep).sum()} sources: {sorted(drop_ids)}")
        highz = highz[keep]

    print(f"\n[highz] Total high-z sources across all catalogues: {len(highz)}")
    return highz


def cross_match_mags(highz, cat, mag_col_x, mag_col_y, max_sep=max_sep):
    """For each high-z source, finds the nearest neighbour in `cat` and returns
    its mag_col_x / mag_col_y values (NaN if no match within max_sep)."""

    c_highz = SkyCoord(ra=highz["ra"] * u.deg, dec=highz["dec"] * u.deg)
    c_cat = SkyCoord(ra=np.asarray(cat["ra"], dtype=float) * u.deg,
                      dec=np.asarray(cat["dec"], dtype=float) * u.deg)

    idx, sep2d, _ = match_coordinates_sky(c_highz, c_cat)
    matched = sep2d < max_sep

    mag_x = np.full(len(highz), np.nan)
    mag_y = np.full(len(highz), np.nan)
    sep_arcsec = np.full(len(highz), np.nan)
    matched_ra = np.full(len(highz), np.nan)
    matched_dec = np.full(len(highz), np.nan)

    mag_x[matched] = np.asarray(cat[mag_col_x], dtype=float)[idx[matched]]
    mag_y[matched] = np.asarray(cat[mag_col_y], dtype=float)[idx[matched]]
    sep_arcsec[matched] = sep2d[matched].to(u.arcsec).value
    matched_ra[matched] = np.asarray(cat["ra"], dtype=float)[idx[matched]]
    matched_dec[matched] = np.asarray(cat["dec"], dtype=float)[idx[matched]]

    return mag_x, mag_y, matched, sep_arcsec, matched_ra, matched_dec


def plot_highz_overlay(ref, merged_cat, highz):

    # background: same as plot_cmd_f200_overlaid() in compare_multiband.py 
    ref_mag = np.asarray(ref["f200w"], dtype=float)
    ref_color = ref_mag - np.asarray(ref["f277w"], dtype=float)

    cat_mag = np.asarray(merged_cat["ab_kron_mag_f200"], dtype=float)
    cat_color = cat_mag - np.asarray(merged_cat["ab_kron_mag_f277"], dtype=float)

    good_ref = np.isfinite(ref_mag) & np.isfinite(ref_color)
    good_cat = np.isfinite(cat_mag) & np.isfinite(cat_color)

    # foreground: high-z sources matched onto each catalog
    ref_hz_mag, ref_hz_color_mag2, ref_matched, ref_sep, ref_ra, ref_dec = cross_match_mags(highz, ref, "f200w", "f277w")
    ref_hz_color = ref_hz_mag - ref_hz_color_mag2

    cat_hz_mag, cat_hz_color_mag2, cat_matched, cat_sep, cat_ra, cat_dec = cross_match_mags(highz, merged_cat, "ab_kron_mag_f200", "ab_kron_mag_f277")
    cat_hz_color = cat_hz_mag - cat_hz_color_mag2

    print("\n[match] High-z sources found in Massimo's catalog: "
          f"{ref_matched.sum()}/{len(highz)}")
    print("[match] High-z sources found in Jorge's catalog: "
          f"{cat_matched.sum()}/{len(highz)}")

    n_both = (ref_matched & cat_matched).sum()
    n_ref_only = (ref_matched & ~cat_matched).sum()
    n_cat_only = (~ref_matched & cat_matched).sum()
    n_neither = (~ref_matched & ~cat_matched).sum()

    print(f"  Both catalogs: {n_both}")
    print(f"  Massimo only:  {n_ref_only}")
    print(f"  Mine only:     {n_cat_only}")
    print(f"  Neither:       {n_neither}")

    plt.figure(figsize=(14, 7))

    plt.plot(ref_mag[good_ref], ref_color[good_ref], '.', color='black',
              ms=2, alpha=0.4, label='Massimo cat')
    plt.plot(cat_mag[good_cat], cat_color[good_cat], '.', color='red',
              ms=2, alpha=0.4, label='Jorge cat')

    if ref_matched.any():
        plt.scatter(ref_hz_mag[ref_matched], ref_hz_color[ref_matched],
                    s=130, facecolors='none', edgecolors='blue',
                    linewidths=1.6, marker='o', zorder=5,
                    label=f'z={z_min}-{z_max} in Massimo cat')

    if cat_matched.any():
        plt.scatter(cat_hz_mag[cat_matched], cat_hz_color[cat_matched],
                    s=160, facecolors='none', edgecolors='gold',
                    linewidths=1.6, marker='s', zorder=5,
                    label=f'z={z_min}-{z_max} in Jorge cat')
        
    for i in np.where(ref_matched)[0]:
        plt.text(ref_hz_mag[i] + 0.1, ref_hz_color[i], str(highz['plot_id'][i]),
             fontsize=7, color='blue', alpha=0.8, clip_on=True)

    for i in np.where(cat_matched)[0]:
        plt.text(cat_hz_mag[i] + 0.1, cat_hz_color[i], str(highz['plot_id'][i]),
             fontsize=7, color='goldenrod', alpha=0.8, clip_on=True)

    plt.xlabel('F200W')
    plt.ylabel('F200W - F277W')


    plt.xlim(15, 30)
    plt.ylim(-1, 1.5)

    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()

    for i in np.where(ref_matched)[0]:
        print(i, ref_hz_mag[i], ref_hz_color[i])

    print(f"\nOverlay plot saved to: {output_file}")
    return (ref_hz_mag, ref_hz_color, ref_matched, ref_ra, ref_dec,
        cat_hz_mag, cat_hz_color, cat_matched, cat_ra, cat_dec)


def write_color_selected_regions(cat,ra_col,dec_col,mag200_col,mag277_col,outfile,color="green"):

    mag200 = np.asarray(cat[mag200_col], dtype=float)
    mag277 = np.asarray(cat[mag277_col], dtype=float)
    color_mag = mag200 - mag277

    good = (
        np.isfinite(mag200)
        & np.isfinite(mag277)
        & (mag200 < 23.0)
        & (color_mag >= 0.15)
        & (color_mag <= 0.30))

    with open(outfile, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"global color={color} width=2\n")
        f.write("fk5\n")

        for row in cat[good]:
            f.write(f'circle({row[ra_col]},{row[dec_col]},0.5")\n')

    print(f"Wrote {good.sum()} regions to {outfile}")

def plot_highz_zoomed(ref, merged_cat, ref_hz_mag, ref_hz_color, ref_matched,cat_hz_mag, cat_hz_color, cat_matched, highz):

    fig, ax = plt.subplots(figsize=(10, 7))

    ref_mag_all = np.asarray(ref["f200w"], dtype=float)
    ref_color_all = ref_mag_all - np.asarray(ref["f277w"], dtype=float)
    cat_mag_all = np.asarray(merged_cat["ab_kron_mag_f200"], dtype=float)
    cat_color_all = cat_mag_all - np.asarray(merged_cat["ab_kron_mag_f277"], dtype=float)

    good_ref = np.isfinite(ref_mag_all) & np.isfinite(ref_color_all)
    good_cat = np.isfinite(cat_mag_all) & np.isfinite(cat_color_all)

    ax.plot(ref_mag_all[good_ref], ref_color_all[good_ref], '.', color='black', ms=2, alpha=0.3)
    ax.plot(cat_mag_all[good_cat], cat_color_all[good_cat], '.', color='red', ms=2, alpha=0.3)

    if ref_matched.any():
        ax.scatter(ref_hz_mag[ref_matched], ref_hz_color[ref_matched],
                   s=130, facecolors='none', edgecolors='blue',
                   linewidths=1.6, marker='o', zorder=5,
                   label=f'z={z_min}-{z_max} in Massimo\'s cat')

    if cat_matched.any():
        ax.scatter(cat_hz_mag[cat_matched], cat_hz_color[cat_matched],
                   s=160, facecolors='none', edgecolors='gold',
                   linewidths=1.6, marker='s', zorder=5,
                   label=f'z={z_min}-{z_max} in my cat')
    
    for i in np.where(ref_matched)[0]:
        ax.text(ref_hz_mag[i] + 0.15, ref_hz_color[i], str(highz['plot_id'][i]),
            fontsize=5, color='blue', alpha=0.8)

    for i in np.where(cat_matched)[0]:
        ax.text(cat_hz_mag[i] + 0.15, cat_hz_color[i], str(highz['plot_id'][i]),
            fontsize=5, color='goldenrod', alpha=0.8)

    both = ref_matched & cat_matched
    for i in np.where(both)[0]:
        ax.plot([ref_hz_mag[i], cat_hz_mag[i]],
                [ref_hz_color[i], cat_hz_color[i]],
                color='orange', linewidth=0.8, alpha=0.7, zorder=4)

    ax.set_xlabel('F200W')
    ax.set_ylabel('F200W - F277W')
    ax.set_xlim(16, 24)
    ax.set_ylim(-0.25, 0.10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    outfile = os.path.join(output_dir, "color_mag_f200_highz_zoomed.png")
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Zoomed CMD saved to: {outfile}")


def create_highz_regions_file(ref_matched, ref_ra, ref_dec, cat_matched, cat_ra, cat_dec, highz):

    region_file = os.path.join(output_dir, "highz_sources.reg")
    both = ref_matched & cat_matched

    with open(region_file, "w") as f:
        f.write("fk5\n")
        f.write(f"# z={z_min}-{z_max} cluster members\n")
        f.write("# red = Massimo, green = mine, orange = match lines\n\n")

        for i in np.where(both)[0]:
            f.write(f"line({ref_ra[i]:.6f},{ref_dec[i]:.6f},"
                    f"{cat_ra[i]:.6f},{cat_dec[i]:.6f}) # color=orange width=2\n")

        for i in np.where(ref_matched)[0]:
            pid = highz['plot_id'][i]
            f.write(f"circle({ref_ra[i]:.6f},{ref_dec[i]:.6f},0.25\") # color=red width=2 text={{{pid}}}\n")

        for i in np.where(cat_matched)[0]:
            pid = highz['plot_id'][i]
            f.write(f"circle({cat_ra[i]:.6f},{cat_dec[i]:.6f},0.25\") # color=#00CC00 width=2 text={{{pid}}}\n")

    print(f"Regions file saved to: {region_file}")
    print(f"  Red circles (Massimo): {ref_matched.sum()}")
    print(f"  Green circles (mine): {cat_matched.sum()}")
    print(f"  Orange lines (both): {both.sum()}")


def plot_delta_mag_highz(ref_hz_mag, cat_hz_mag, ref_matched, cat_matched, highz):

    both = ref_matched & cat_matched
    if both.sum() == 0:
        print("[delta mag] No sources matched in both catalogs — skipping.")
        return

    my_mag = cat_hz_mag[both]
    delta = ref_hz_mag[both] - cat_hz_mag[both]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(my_mag, delta, s=40, color='blue', alpha=0.8, zorder=3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    median_offset = np.nanmedian(delta)
    ax.axhline(median_offset, color='red', linestyle='--', linewidth=1,
               label=f'median offset = {median_offset:+.3f} mag')

    """for i in np.where(both)[0]:
        plt.text(
            ref_hz_mag[i] + 0.03,
            delta[i],
            str(highz["plot_id"][i]),
            fontsize=6
    )"""

    indices = np.where(both)[0]

    for i in indices:
        x = cat_hz_mag[i]
        y = ref_hz_mag[i] - cat_hz_mag[i]

        ax.text(
            x + 0.03,
            y,
            str(highz["plot_id"][i]),
            fontsize=6,
            clip_on=True
        )

    ax.set_xlabel('My F200W magnitude')
    ax.set_ylabel('Δmag (Massimo - mine)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    outfile = os.path.join(output_dir, "delta_mag_highz.png")
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Delta mag plot saved to: {outfile}")
    print(f"  Sources plotted: {both.sum()}")
    print(f"  Median offset: {median_offset:+.3f} mag")

def write_highz_summary_csv(highz, ref_matched, cat_matched, output_dir):
    """
    Saves a CSV with one row per high-z source showing its ID, coordinates,
    redshift, source catalog name, and whether it was matched in Massimo's,
    Jorge's, or Frye's catalog.
    Note: 'spectra_prelim' entries come from Frye's catalog.
    """
    out = Table()
    out['plot_id']       = highz['plot_id']
    out['source_id']     = highz['id']
    out['ra']            = highz['ra']
    out['dec']           = highz['dec']
    out['zspec']         = highz['zspec']
    out['in_frye']  = [sc == 'spectra_prelim' for sc in highz['source_catalog']]
    out['in_massimo']    = ref_matched
    out['in_mine']       = cat_matched

    outfile = os.path.join(output_dir, "highz_sources_summary.csv")
    out.write(outfile, format='csv', overwrite=True)
    print(f"Summary CSV saved to: {outfile} ({len(out)} sources)")


def inspect_source(plot_id, highz, ref, merged_cat, max_sep=max_sep):
    # Diagnostic: dumps everything known about about a source with plot_id
    row = highz[highz['plot_id'] == plot_id]
    if len(row) == 0:
        print(f"No source with plot_id={plot_id}")
        return
    row = row[0]

    print(f"source plot_id = {plot_id}")
    print(f"  catalog id     : {row['id']}")
    print(f"  origin catalog : {row['source_catalog']}")
    print(f"  RA, Dec (deg)  : {row['ra']:.7f}, {row['dec']:.7f}")
    c = SkyCoord(row['ra'] * u.deg, row['dec'] * u.deg)
    print(f"  RA, Dec (sexa) : {c.to_string('hmsdms', precision=2)}")
    print(f"  zspec          : {row['zspec']:.5f}")

    for label, cat, racol, deccol, cols in [
        ("Massimo", ref, "ra", "dec", ["f200w", "f277w"]),
        ("Jorge",   merged_cat, "ra", "dec",
         ["ab_kron_mag_f200", "ab_kron_mag_f277"]),
    ]:
        print(f"\n--- {label} catalog ---")
        cc = SkyCoord(np.asarray(cat[racol], float) * u.deg,
                      np.asarray(cat[deccol], float) * u.deg)
        sep = c.separation(cc).to(u.arcsec).value
        order = np.argsort(sep)[:3]

        for rank, j in enumerate(order):
            flag = "Matched" if sep[j] < max_sep.to(u.arcsec).value else "too far"
            print(f"  [{rank}] sep = {sep[j]:.3f}\" ({flag})")
            print(f"      row index : {j}")
            print(f"      RA, Dec   : {cat[racol][j]:.7f}, {cat[deccol][j]:.7f}")
            for cn in cols:
                if cn in cat.colnames:
                    print(f"      {cn:20s}: {float(cat[cn][j]):.4f}")
            if all(cn in cat.colnames for cn in cols):
                m1, m2 = float(cat[cols[0]][j]), float(cat[cols[1]][j])
                print(f"      color (200-277)     : {m1 - m2:.4f}")
            # dump every other column for the best match only
            if rank == 0:
                print("full row:")
                for cn in cat.colnames:
                    print(f"         {cn:24s} = {cat[cn][j]}")
"""
def find_duplicate_highz(highz, tol=0.5 * u.arcsec):
    # report high-z entries that sit within `tol` of each other
    c = SkyCoord(highz['ra'] * u.deg, highz['dec'] * u.deg)
    idx1, idx2, sep, _ = c.search_around_sky(c, tol)
    keep = idx1 < idx2          # each pair once, drop self-matches
    idx1, idx2, sep = idx1[keep], idx2[keep], sep[keep]

    print(f"\n[dup] {len(idx1)} pairs within {tol}")
    for a, b, s in zip(idx1, idx2, sep):
        print(f"  plot_id {highz['plot_id'][a]:3d} (id={highz['id'][a]}) <-> "
              f"plot_id {highz['plot_id'][b]:3d} (id={highz['id'][b]}) : "
              f"{s.to(u.arcsec).value:.4f}\"  "
              f"z={highz['zspec'][a]:.5f} / {highz['zspec'][b]:.5f}")
    return idx1, idx2, sep
"""


# Main

if __name__ == "__main__":

    print(f"Selecting sources with {z_min} <= zspec <= {z_max}")
    highz = build_highz_catalog()

    print("\nLoading Massimo and Jorge photometry catalogs")
    ref = Table.read(ref_path, format="ascii")
    merged_cat = Table.read(cat_path, format="csv")

    # inspecting anomalous sources:
    inspect_source( 7, highz, ref, merged_cat)
    inspect_source(58, highz, ref, merged_cat)
    inspect_source(79, highz, ref, merged_cat)
    inspect_source(80, highz, ref, merged_cat)

    # find_duplicate_highz(highz)

    write_color_selected_regions(
        merged_cat,
        ra_col="ra",
        dec_col="dec",
        mag200_col="ab_kron_mag_f200",
        mag277_col="ab_kron_mag_f277",
        outfile=os.path.join(output_dir, "anomalies.reg"),
        color="green")
    
    (ref_hz_mag, ref_hz_color, ref_matched, ref_ra, ref_dec,
    cat_hz_mag, cat_hz_color, cat_matched, cat_ra, cat_dec) = plot_highz_overlay(ref, merged_cat, highz)

    plot_highz_zoomed(ref, merged_cat, ref_hz_mag, ref_hz_color, ref_matched, cat_hz_mag, cat_hz_color, cat_matched, highz)
    
    create_highz_regions_file(ref_matched, ref_ra, ref_dec,cat_matched, cat_ra, cat_dec, highz)

    plot_delta_mag_highz(ref_hz_mag, cat_hz_mag, ref_matched, cat_matched, highz)

    write_highz_summary_csv(highz, ref_matched, cat_matched, output_dir)

