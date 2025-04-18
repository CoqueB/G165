from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus, ApertureStats
import numpy as np

positions = (30.0, 30.0)
aperture = CircularAperture(positions, r=3.0)

#check this later
data = np.ones((100, 100))
phot_table = aperture_photometry(data, aperture)
phot_table['aperture_sum'].info.format = '%.8g'  # for consistent table output
print(phot_table)

anulus_aperture = CircularAnnulus(positions, r_in = 40, r_out = 50)
aperture_stats = ApertureStats(data, anulus_aperture)
bkg_mean = aperture_stats.mean
total_bkg_flux = bkg_mean * aperture.area

star_flux = phot_table["aperture_sum"] - total_bkg_flux

# maybe caclulate m? use : m= -2.5*np.log10(star_flux)
# plot star_flux vs XX of spitzer catalouge

