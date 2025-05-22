"""
    To build a proper PSF model, we need to have:
    1. Good sample of high S/N isolated stars
    2. Have a large sample

    To Do:
        Pixelize and un-interpolate the brightness graphs
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from astropy.visualization import simple_norm
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch

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
norm = simple_norm(data, 'sqrt', percent=95.0)

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
fixed_vmin = 20


for i in range(nrows * ncols):
    
    # 1) Stretch the cutout for better visualization
    norm = simple_norm(stars[i], 'log', percent=95.0) # applies log stretch to the cutout
    # Plots the i-th star cutout (2D array) into subplot ax[i] stores each cutout in <im>
    im = ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis') 
    

    # 2) Set a fixed vmin and vmax for all cutouts, uncomment for this version
    #star_data = stars[i].data  # EPSFStar object, so get the .data
    #vmax = star_data.max() # Use brightest pixel for this cutout 
    #im = ax[i].imshow(star_data, origin='lower', cmap='viridis',
    #                  vmin=fixed_vmin, vmax=vmax)
    images.append(im)

fig.colorbar(images[0], ax=ax.ravel().tolist(), 
             orientation='vertical', fraction=0.02, pad=0.04) # adds a colorbar to the figure
plt.savefig('figures/psf_cutouts.png', dpi=300, bbox_inches='tight') # saves the figure
plt.show()





#-----------------------------------#
#---Interactive Plotting of Stars---#
#-----------------------------------#
nstars = len(stars) # Total number of stars (404)
current_index = [0]  # so we can modify it inside callbacks

# Set up figure and axes
cutout_size = stars[0].data.shape[0] # The width & height of the cutout
fig = plt.figure(figsize=(15, 5)) # Creates a figure with size 10x3 inches
gs = fig.add_gridspec(1, 5, width_ratios=[1.5, 1, 1, 0.5, 0.5]) # Defines a 1 row, 5 column grid layout 



# <width_ratios> sets the relative width of each column

# Positions plots and buttons in the grid
ax_img = fig.add_subplot(gs[0])
ax_x = fig.add_subplot(gs[1])
ax_y = fig.add_subplot(gs[2])
ax_prev = fig.add_subplot(gs[3])
ax_next = fig.add_subplot(gs[4])


ax_img.set_aspect('equal')  # Keep cutout square
ax_x.set_aspect('auto')     # Allow auto for brightness plots
ax_y.set_aspect('auto')

# Turns the last 2 plots into pressable buttons
btn_prev = Button(ax_prev, 'Previous')
btn_next = Button(ax_next, 'Next')

# Now change the button height and width via the `ax_prev` and `ax_next` axes
btn_prev.ax.set_aspect(2)  # Increase aspect ratio for height
btn_next.ax.set_aspect(2)  # Increase aspect ratio for height


# Plot updater function
def update_plot(index):
    # Clears the first 3 plots on those axes
    ax_img.clear()
    ax_x.clear()
    ax_y.clear()

    # Gets the star data and normalizes it for better visualization
    star_data = stars[index].data
    
    # Gets the star position from the stars table
    center_x = stars_tbl['x'][index] 
    center_y = stars_tbl['y'][index]

    # 2 ways to normalize the data
    
    # 1. Rescaling each cut independently:
    #   - One cutout might have a bright star nearby, skewing the scaling
    # Resultig in each cutout having a different background 
    #norm = simple_norm(star_data, 'linear', percent=99.0)

    # 2. Having a set minimum and maximum pixel value for all cutouts:
    #   - This is better for comparing cutouts
    # Lower vmax = more contrast in faint regions
    # Higher vmax = preserves detail in bright regions, but can flatten dimmer areas
    # star_data.max() = just uses the brightest pixel in the cutout
    norm = ImageNormalize(vmin=0, vmax=3000, stretch=LogStretch())
    


    # Define extent: [x_min, x_max, y_min, y_max] in original coordinates
    hsize = (star_data.shape[0] - 1) / 2  # Half the size of the cutout
    
    # The extent parameter takes a list or tuple of four values: [left, right, bottom, top].
    # These values define the x and y coordinates of the image's corners. Comment or 
    # delete the <extent=extent> parameter to have the image plotted in pixel coordinates
    extent = [
        center_x - hsize, center_x + hsize,  # x limits in original coordinates
        center_y - hsize, center_y + hsize   # y limits in original coordinates
    ]

    # Adds the star cutout plot on the first axis/subplot
    ax_img.imshow(star_data, norm=norm, origin='lower', cmap='viridis', extent=extent)
    ax_img.set_title(f'Star {index + 1} of {nstars}')

    # .sum() method is a NumPy array method that computes the sum of the array's elements along a specified axis.
    col_sum = star_data.sum(axis=0) # Brightness summed vertically over each x
    row_sum = star_data.sum(axis=1) # Brightness summed horizontally over each y

    # Plots the sum column brightnesses against x
    ax_x.step(range(star_data.shape[1]), col_sum, where='mid', color='tab:blue')
    ax_x.set_title('Brightness vs X')
    ax_x.set_xlabel('X Position')
    ax_x.set_ylabel('Sum')

    # Plots the sum row brightnesses against y
    ax_y.step(row_sum, range(star_data.shape[0]), where='mid', color='tab:orange')
    ax_y.set_title('Brightness vs Y')
    ax_y.set_ylabel('Y Position')
    ax_y.set_xlabel('Sum')
    ax_y.invert_yaxis() # inverts to match the image orientation

    plt.draw() # Redraws the figure to show the updated plots


# Increments the index by 1, wraps around to the first star when you're at the last star
def next_star(event):
    # assume you're at index nstars-1 (<nstars>th star), when you 
    # press next it will go to index 0 (first star)
    current_index[0] = (current_index[0] + 1) % nstars 
    update_plot(current_index[0])

# Same as next_star() but decrements the index by 1
def prev_star(event):
    current_index[0] = (current_index[0] - 1) % nstars
    update_plot(current_index[0])

# When btn_<action> is pressed, call the function <action> which increments/decrements the index
btn_next.on_clicked(next_star)
btn_prev.on_clicked(prev_star)

# this line makes sure the first star
# is plotted at first by calling the update_plot() 
# function and clearing the canvas/axes
# then plt.show() displays the figure
update_plot(current_index[0]) 
plt.tight_layout()
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

norm = simple_norm(epsf.data, 'log', percent=95.0)
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.savefig('figures/epsf.png', dpi=300, bbox_inches='tight') # saves the figure
plt.show()



# Use one of the fitted stars from the list (for example, the first one)
fitted_star = fitted_stars[0]  # Change the index to plot a different fitted star

# Normalize and plot the selected fitted star
norm = simple_norm(fitted_star, 'log', percent=95.0)
plt.imshow(fitted_star, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Fitted Star 1")  # Optional: add a title to the plot

# Save the figure as a PNG file
plt.savefig('figures/fitted_star_1.png', dpi=300, bbox_inches='tight')
plt.show()
