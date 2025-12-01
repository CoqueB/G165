import numpy as np
from photutils.datasets import make_noise_image
from photutils.psf import CircularGaussianPRF, make_psf_model_image
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt

# For fitting & finding sources
from photutils.detection import DAOStarFinder
from photutils.psf import (CircularGaussianPRF, PSFPhotometry,
                           make_psf_model_image)

from astropy.table import QTable


#################################
#---Creating a synthetic image---
#################################

psf_model = CircularGaussianPRF(flux=1, fwhm=2.7)
psf_shape = (9, 9) 
n_sources = 10 # Number of sources to simulate
shape = (101, 101) # Shape of the image

data, true_params = make_psf_model_image(shape, psf_model, n_sources,
                                         model_shape=psf_shape,
                                         flux=(500, 700),
                                         min_separation=10, seed=0)

# Add noise to the image
# The noise is added to the data, and the absolute value of the noise is calculated
noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
data += noise
error = np.abs(noise)

# Plot the simulated data, and save it as a PNG file
plt.figure(figsize=(6, 6))
plt.imshow(data, origin='lower', cmap='viridis', interpolation='none')
plt.colorbar(label='Pixel Value')
plt.title('Simulated Data')
plt.tight_layout(),
#plt.savefig('psf_training/figures/simulated_image.png')  # Save it as a PNG file


#----------------------------#
#---FITTING MULTIPLE STARS---#
#----------------------------#

psf_model = psf_model # Using the same PSF model
fit_shape = (5, 5)
finder = DAOStarFinder(6.0, 2.0) # Set the threshold (6 sigma) and fwhm for star finding
psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder, 
                        aperture_radius=4) 

# Detects sources in data and returns the 
# photometry result with positions and fluxes
photometry_result = psfphot(data, error=error) 

# Save the result to a CSV file if needed
#photometry_result.write('psf_training/photometry_result.csv', format='csv',
#                          overwrite=True)  

# Create, plot, and save residıal image
resid = psfphot.make_residual_image(data) # original img - model img

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
norm = simple_norm(data, 'sqrt', percent=99)
ax[0].imshow(data, origin='lower', norm=norm)
ax[1].imshow(data - resid, origin='lower', norm=norm)
im = ax[2].imshow(resid, origin='lower', norm=norm)
ax[0].set_title('Data')
ax[1].set_title('Model')
ax[2].set_title('Residual Image')
plt.tight_layout()
plt.savefig('psf_training/figures/data_with_residual.png')


#------------------------------------#
#---Accessing individual star data---#
#------------------------------------#

# psfphot.finder_results['<column name>'].info.format = '.4f' -> 4 decimal places
print(psfphot.finder_results) # Prints the attributes of sources like position, flux, etc.

# psfphot.fit_info is a dictionary containing info about fitted and 
# unfitted sources. Keys: dict_keys(['fit_infos', 'fit_error_indices'])
print(psfphot.fit_info['fit_infos'][0])

# Structure of psfphot.fit_info Dictionaty:
"""
psfpht.fit_info = { 
    'fit_infos': [
    <list of dictionaries for each source>
        {'fun': difference between the observed pixel values and the values predicted by the PSF model, 
        'param_cov': covariance matrix (?), 
        'message': a human-readable message explaining why the optimization stopped, 
        'status': integer indicating the exit status of the optimization
        }
    
    ], 
    'fit_error_indices': <list of indices of sources that failed to fit>,
}

"""

#---Fitting a single star in the image---
init_params = QTable() 
init_params['x'] = [63]
init_params['y'] = [49]
# Produce photometry for just that single star at (63, 49)
photometry_result_single_fit = psfphot(data, error=error, init_params=init_params)
