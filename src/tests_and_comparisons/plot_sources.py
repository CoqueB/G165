

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

    mag_x[matched] = np.asarray(cat[mag_col_x], dtype=float)[idx[matched]]
    mag_y[matched] = np.asarray(cat[mag_col_y], dtype=float)[idx[matched]]
    sep_arcsec[matched] = sep2d[matched].to(u.arcsec).value

    return mag_x, mag_y, matched, sep_arcsec


def plot_highz_overlay(ref, merged_cat, highz):

    # background: same as plot_cmd_f200_overlaid() in compare_multiband.py 
    ref_mag = np.asarray(ref["f200w"], dtype=float)
    ref_color = ref_mag - np.asarray(ref["f277w"], dtype=float)

    cat_mag = np.asarray(merged_cat["ab_kron_mag_f200"], dtype=float)
    cat_color = cat_mag - np.asarray(merged_cat["ab_kron_mag_f277"], dtype=float)

    good_ref = np.isfinite(ref_mag) & np.isfinite(ref_color)
    good_cat = np.isfinite(cat_mag) & np.isfinite(cat_color)

    # foreground: high-z sources matched onto each catalog
    ref_hz_mag, ref_hz_color_mag2, ref_matched, ref_sep = cross_match_mags(
        highz, ref, "f200w", "f277w")
    ref_hz_color = ref_hz_mag - ref_hz_color_mag2

    cat_hz_mag, cat_hz_color_mag2, cat_matched, cat_sep = cross_match_mags(
        highz, merged_cat, "ab_kron_mag_f200", "ab_kron_mag_f277")
    cat_hz_color = cat_hz_mag - cat_hz_color_mag2

    print("\n[match] High-z sources found in Massimo's catalog: "
          f"{ref_matched.sum()}/{len(highz)}")
    print("[match] High-z sources found in my catalog: "
          f"{cat_matched.sum()}/{len(highz)}")

    for i in range(len(highz)):
        bits = []
        if ref_matched[i]:
            bits.append(f"Massimo at {ref_sep[i]:.3f}\"")
        if cat_matched[i]:
            bits.append(f"mine at {cat_sep[i]:.3f}\"")
        status = ", ".join(bits) if bits else "no match in either catalog"
        print(f"    {highz['id'][i]} (z={highz['zspec'][i]:.3f}, "
              f"from {highz['source_catalog'][i]}): {status}")

    plt.figure(figsize=(7, 7))

    plt.plot(ref_mag[good_ref], ref_color[good_ref], '.', color='black',
              ms=2, alpha=0.4, label='Reference')
    plt.plot(cat_mag[good_cat], cat_color[good_cat], '.', color='red',
              ms=2, alpha=0.4, label='My catalog')

    if ref_matched.any():
        plt.scatter(ref_hz_mag[ref_matched], ref_hz_color[ref_matched],
                    s=130, facecolors='none', edgecolors='blue',
                    linewidths=1.6, marker='o', zorder=5,
                    label=f'z={z_min}-{z_max} in Massimo\'s cat')

    if cat_matched.any():
        plt.scatter(cat_hz_mag[cat_matched], cat_hz_color[cat_matched],
                    s=160, facecolors='none', edgecolors='gold',
                    linewidths=1.6, marker='s', zorder=5,
                    label=f'z={z_min}-{z_max} in my cat')

    plt.xlabel('F200W')
    plt.ylabel('F200W - F277W')


    plt.xlim(15, 30)
    plt.ylim(-1, 1.5)

    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()

    print(f"\nOverlay plot saved to: {output_file}")


# Main

if __name__ == "__main__":

    print(f"Selecting sources with {z_min} <= zspec <= {z_max}")
    highz = build_highz_catalog()

    print("\nLoading reference and my photometry catalogs")
    ref = Table.read(ref_path, format="ascii")
    merged_cat = Table.read(cat_path, format="csv")

    plot_highz_overlay(ref, merged_cat, highz)