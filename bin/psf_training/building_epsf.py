"""
    To build a proper PSF model, we need to have:
    1. Good sample of high S/N isolated stars
    2. Have a large sample

"""

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from photutils.datasets import (load_simulated_hst_star_image,
                                make_noise_image)

from photutils.detection import find_peaks # find_peaks to find stars & their positions
from astropy.table import Table
from astropy.stats import sigma_clipped_stats # for sigma clipping (subtracring background)

#------------------------------------#
#---Loading & creating sample data---#
#------------------------------------#

hdu = load_simulated_hst_star_image()
data = hdu.data # 2D numpy array
data += make_noise_image(data.shape, distribution='gaussian', mean=10.0,
                         stddev=5.0, seed=123)

# simple_norm(data, <stretch function>, percent= <clips the data>) for better visualization
norm = simple_norm(data, 'sqrt', percent=99.0)

# Plotting & saving the image
plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.tight_layout()
plt.title('Simulated HST Star Image')
plt.show()
plt.savefig('psf_training/figures/epsf_sample.png')


#-------------------------------------#
#---Finding stars & their positions---#
#-------------------------------------#

peaks_tbl = find_peaks(data, threshold=500.0) # Choose stars with peak piexl values > 500
peaks_tbl['peak_value'].info.format = '%.8g' # 8 sig figs
# <QTable length=431>
#    name     dtype  format
# ---------- ------- ------
#         id   int64       
#     x_peak   int64       
#     y_peak   int64       
# peak_value float64   %.8g


#-----------------------#
#---Producing Cutouts---#
#-----------------------#
size = 25
hsize = (size - 1) / 2
x = peaks_tbl['x_peak']  
y = peaks_tbl['y_peak']  

# Ensures you have enough room (at least <hsize> pixels
# in all directions) to extract a full cutout.
mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
        (y > hsize) & (y < (data.shape[0] -1 - hsize)))  

stars_tbl = Table()
stars_tbl['x'] = x[mask]  
stars_tbl['y'] = y[mask]  

# Gets the mean, median, and stddev of the data (basicallg avg, 
# mean etc. pixel value in the image) *EXCLUDING* the stars
#  by using the statistical method "sigma clipping"
mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)  
data -= median_val # gets rid of the background (median insted of mean because stars are outliers & skew the mean)
