from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import os

# Loading FITS Image:
# -------------------

fits_file = '/mnt/c/users/Coque/Downloads/mosaic_plckg165_nircam_f444w_30mas_20230403_drz.fits'  # Replace with the actual FITS file path for LINUX
hdul = fits.open(fits_file)       # This is the HDUList object
hdu = hdul[0]                     # This is the PrimaryHDU
image_header = hdu.header
wcs = WCS(image_header)
data = hdu.data.astype(float)
hdul.close()                      

# Save dimensions:
# ----------------

print("Image shape:", data.shape)  # returns (ny, nx)
coordinates = data.shape

size = (coordinates[1]//2, coordinates[0]//2)
print("Image shape:", size)  # returns (nx, ny)

# Make sure output directory exists, or creating it:
# --------------------------------------------------

output_dir = "cutouts"
os.makedirs(output_dir, exist_ok=True)

# Createing cutouts:
# ------------------

# Define positions (x, y)
positions = {
    "cutout_f444": (3 * coordinates[1] // 4, coordinates[0] // 4),  # lower right
}

for name, pos in positions.items():
    cutout = Cutout2D(data, position=pos, size=size, wcs=wcs)

    # Build header from cutout WCS
    cutout_header = cutout.wcs.to_header()

    # Preserve key header values if present
    for key in ["PHOTMJSR", "PIXAR_SR"]:
        if key in image_header:
            cutout_header[key] = image_header[key]

    # Create and save FITS file
    hdu_cutout = fits.PrimaryHDU(data=cutout.data, header=cutout_header)
    output_path = os.path.join(output_dir, f"{name}.fits")
    hdu_cutout.writeto(output_path, overwrite=True)
    
    print(f"Saved {output_path}")

 






