from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus, ApertureStats
import numpy as np

"""

# creating an aperture arround a point

positions = (30.0, 30.0)
aperture = CircularAperture(positions, r=3.0)


# creating a table with the aperture photometry data

data = np.ones((100, 100))
phot_table = aperture_photometry(data, aperture)
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
print(phot_table)


# creating an annulus arround an aperture to calculate the background

anulus_aperture = CircularAnnulus(positions, r_in = 40, r_out = 50)
aperture_stats = ApertureStats(data, anulus_aperture)
bkg_mean = aperture_stats.mean
total_bkg_flux = bkg_mean * aperture.area

star_flux = phot_table["aperture_sum"] - total_bkg_flux

# to caclulate m use : m= -2.5*np.log10(star_flux)
# plot star_flux vs XX of spitzer catalouge

"""




#trying to open a fits image

from astropy.io import fits 

fitsfile = fits.open("./data/mosaic_plckg165_nircam_f090w_30mas_20230403_drz.fits")


image_data = fitsfile[0].data


import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

plt.figure()
plt.imshow(image_data, origin = "lower")
plt.colorbar()
plt.show()
