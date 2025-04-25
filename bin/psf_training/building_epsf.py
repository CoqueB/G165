"""
    To build a proper PSF model, we need to have:
    1. Good sample of high S/N isolated stars
    2. Have a large sample

"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from astropy.visualization import simple_norm
from photutils.datasets import (load_simulated_hst_star_image,
                                make_noise_image)

from photutils.detection import find_peaks # find_peaks to find stars & their positions
from photutils.psf import extract_stars 
from photutils.psf import EPSFBuilder
from astropy.table import Table
from astropy.stats import sigma_clipped_stats # for sigma clipping (subtracring background)
from astropy.nddata import NDData

matplotlib.use('TkAgg') # Use TkAgg backend for interactive plotting

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
plt.title('Simulated HST Star Image')
#plt.show()


#-------------------------------------#
#---Finding stars & their positions---#
#-------------------------------------#

peaks_tbl = find_peaks(data, threshold=500.0) # Choose stars with peak pixel values > 500
peaks_tbl['peak_value'].info.format = '%.8g' # 8 sig figs
print(peaks_tbl.columns) # <TableColumns names=('id','x_peak','y_peak','peak_value')>
# <QTable length=431>
#    name     dtype  format
# ---------- ------- ------
#         id   int64       
#     x_peak   int64       
#     y_peak   int64       
# peak_value float64   %.8g


#----------------------------------#
#---Producing & Plotting Cutouts---#
#----------------------------------#
size = 25
hsize = (size - 1) / 2
x = peaks_tbl['x_peak']  
y = peaks_tbl['y_peak']  

# Ensures you have enough room (at least <hsize> pixels
# in all directions) to extract a full cutout.
mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
        (y > hsize) & (y < (data.shape[0] -1 - hsize)))  # Understand this

stars_tbl = Table()
stars_tbl['x'] = x[mask]  
stars_tbl['y'] = y[mask]  

# Gets the mean, median, and stddev of the data (basicallg avg, 
# mean etc. pixel value in the image) *EXCLUDING* the stars
# by using the statistical method "sigma clipping"
mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.0)  
data -= median_val # gets rid of the background (median insted of mean because stars are outliers & skew the mean)

plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.title('Sigma Clipped Data')
#plt.show()

# Tools like extract_stars() expect NDData objects, 
# the following line converts the 2D numpy array to 
# an NDData object This lets Astropy/Photutils know: 
# that theses are not just raw pixels, but an image 
# with possible metadata 
nddata = NDData(data=data) 


# In <nddata>, creates a <size> sized cutouts, centered 
# at <stars_tbl['x']> and <stars_tbl['y']>
stars = extract_stars(nddata, stars_tbl, size=25)
print(len(stars.data)) # Output: 404, meaning we have 404 stars
print(stars.data[0].shape) # Output: (25, 25) shape of the first cutout, can also use <stars.cutout_shape>

nrows = 5
ncols = 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 25),
                       squeeze=True)

images = []
# Flattens the 2D array of axes into a 1D array of 25 plots, 
# allowing indexing of axes with ax[i] in a loop instead of ax[row][col]
ax = ax.ravel() 

for i in range(nrows * ncols):
    norm = simple_norm(stars[i], 'log', percent=99.0) # applies log stretch to the cutout
    
    # Plots the i-th star cutout (2D array) into subplot ax[i] stores each cutout in <im>
    im = ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis') 
    images.append(im)

fig.colorbar(images[0], ax=ax.ravel().tolist(), 
             orientation='vertical', fraction=0.02, pad=0.04) # adds a colorbar to the figure
plt.savefig('figures/psf_cutouts.png', dpi=300, bbox_inches='tight') # saves the figure
plt.show()


#----------------------------------#
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 25), squeeze=False)

# Number of stars to show
nstars = 5
cutout_size = stars[0].data.shape[0]  # assume square cutouts (25)

# Set figure size so each pixel is visible clearly (e.g., scale=0.4 inches per pixel)
scale = 0.4
fig_width = 3 * cutout_size * scale  # 3 plots per star (cutout + 2 profiles)
fig_height = nstars * cutout_size * scale

fig, ax = plt.subplots(nrows=nstars, ncols=3, figsize=(fig_width, fig_height), squeeze=False)

for i in range(nstars):
    star_data = stars[i].data

    # Plot 1: Star cutout
    norm = simple_norm(star_data, 'log', percent=99.0)
    ax[i, 0].imshow(star_data, norm=norm, origin='lower', cmap='viridis')
    ax[i, 0].set_title(f'Star {i+1}')
    
    # Plot 2: Vertical profile (sum across rows)
    column_sum = star_data.sum(axis=0)
    ax[i, 1].plot(range(star_data.shape[1]), column_sum, color='tab:blue')
    ax[i, 1].set_title('Brightness vs X')
    ax[i, 1].set_xlabel('X Position')
    ax[i, 1].set_ylabel('Sum of Brightness')
    
    # Plot 3: Horizontal profile (sum across columns)
    row_sum = star_data.sum(axis=1)
    ax[i, 2].plot(row_sum, range(star_data.shape[0]), color='tab:orange')
    ax[i, 2].set_title('Brightness vs Y')
    ax[i, 2].set_ylabel('Y Position')
    ax[i, 2].set_xlabel('Sum of Brightness')
    ax[i, 2].invert_yaxis()  # Match image orientation

# Optional cleanup for tighter layout
for axs in ax.ravel():
    axs.label_outer()  # hide x/y ticks where not needed

plt.tight_layout()
plt.savefig('figures/star_profiles_grid.png', dpi=300, bbox_inches='tight')
plt.show()



#--------------------------------#
#---Building the ePSF Function---#
#--------------------------------#

# <oversampling=4>: controls the resolution of the ePSF
# <maxiters=3>: maximum number of iterations to fit the ePSF
epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
                           progress_bar=False)  

# <epsf_builder(stars)>: fits the ePSF to the star cutouts
# <epsf>: the ePSF model, a 2D array that represents the avg PSF (how stars appear in image)
# <fitted_stars>: the fitted star cutouts, a list of 2D arrays 
epsf, fitted_stars = epsf_builder(stars) 

norm = simple_norm(epsf.data, 'log', percent=99.0)
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.savefig('figures/epsf.png', dpi=300, bbox_inches='tight') # saves the figure
plt.show()






# Use one of the fitted stars from the list (for example, the first one)
fitted_star = fitted_stars[0]  # Change the index to plot a different fitted star

# Normalize and plot the selected fitted star
norm = simple_norm(fitted_star, 'log', percent=99.0)
plt.imshow(fitted_star, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Fitted Star 1")  # Optional: add a title to the plot

# Save the figure as a PNG file
plt.savefig('figures/fitted_star_1.png', dpi=300, bbox_inches='tight')
plt.show()
