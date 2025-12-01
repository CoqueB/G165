# opening a simulated HST image:
# ------------------------------

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.datasets import (load_simulated_hst_star_image,
                                make_noise_image)

hdu = load_simulated_hst_star_image()
data = hdu.data
data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                         stddev=5.0, seed=123)
norm = simple_norm(data, 'sqrt', percent=99.0)
# plt.imshow(data, norm=norm, origin='lower', cmap='viridis')


# performing aperture photmetry:
# ------------------------------


from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus, ApertureStats
import numpy as np


# creating an aperture arround a point

# ^ might need to use a star finder here

positions = (30.0, 30.0)
aperture = CircularAperture(positions, r=3.0)


# creating a table with the aperture photometry data


phot_table = aperture_photometry(data, aperture)
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
print(phot_table)


# creating an annulus arround an aperture to calculate the background

anulus_aperture = CircularAnnulus(positions, r_in = 40, r_out = 50)
aperture_stats = ApertureStats(data, anulus_aperture)
bkg_mean = aperture_stats.mean
total_bkg_flux = bkg_mean * aperture.area

star_flux = phot_table["aperture_sum"] - total_bkg_flux

# ^ how do I update star:flux to the table?

print(total_bkg_flux)
print(star_flux)

phot_table["aperture_sum"] == star_flux

print(phot_table)






