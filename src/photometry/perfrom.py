import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy import units as u

from functions import load_image
from functions import subtract_background
from functions import load_weightfile
from functions import calculate_uncertainty
from functions import make_outputdir
from functions import source_detection
from functions import extract_source_properties
from functions import my_aperture_photometry
from functions import kron_photometry


class Photometry():
    def __init__(self, fits_file, wht_file):
        self.fits_file = fits_file
        self.wht_file = wht_file
        self.image_header, self.image_data = load_image(self.fits_file)
        self.image_sub, self.bkg = subtract_background(self.image_data)
        self.weight_data = load_weightfile(self.wht_file)
        self.data_rms, self.background_rms = calculate_uncertainty(self.image_header, self.image_data, self.weight_data, self.bkg)
        self.output_dir = make_outputdir()
        self.segm = source_detection(self.bkg, self.weight_data, self.image_sub, self.image_header, self.output_dir)
        self.tbl, self.catalog = extract_source_properties(self.image_sub, self.segm, self.data_rms)
        self.phot_table, self.apertures, self.annulus_apertures = my_aperture_photometry(self.tbl, self.image_sub, self.image_header)
        self.phot_table = kron_photometry(self.tbl, self.image_header, self.phot_table)

        self.create_regions_file(self.phot_table, self.output_dir, self.catalog)
        self.plotting_images(self.image_sub, self.apertures, self.annulus_apertures, self.output_dir, self.catalog)
        self.printing_storing(self.phot_table, self.output_dir)

    def create_regions_file(self, phot_table, output_dir, catalog):
        region_filename = os.path.join(output_dir, "sources.reg")
        with open(region_filename, "w") as f:
            for x, y in zip(phot_table['xcenter'], phot_table['ycenter']): 
                f.write(f"circle({x:.3f},{y:.3f},{14}) # color=cyan\n")
            
            for src in catalog:
                kron_ap = getattr(src, "kron_aperture", None)

                xk = yk = None
                pos = getattr(kron_ap, "positions", None)
                if pos is not None:
                    arr = np.asarray(pos)
                    # common cases:
                    # arr.shape == (2,)        -> [x, y]
                    # arr.shape == (1,2) or (N,2) -> [[x, y], ...]
                    # arr.size == 1            -> single scalar (treat as x)
                    if arr.ndim == 1 and arr.size >= 2:
                        xk, yk = float(arr[0]), float(arr[1])
                    elif arr.ndim >= 2 and arr.shape[-1] >= 2:
                        xk, yk = float(arr[0, 0]), float(arr[0, 1])
                    elif arr.size == 1:
                        xk = float(arr.ravel()[0])
                        yk = getattr(kron_ap, "ycenter", getattr(kron_ap, "y", None))
                else:
                    xk = getattr(kron_ap, "xcenter", getattr(kron_ap, "x", None))
                    yk = getattr(kron_ap, "ycenter", getattr(kron_ap, "y", None))

                a = getattr(kron_ap, "a", None)
                b = getattr(kron_ap, "b", None)
                theta = getattr(kron_ap, "theta", None)

                if theta is not None:
                    theta = float(theta.to(u.deg).value) % 360.0
                else:
                    theta = 0.0

                if None not in (xk, yk, a, b):
                    f.write(f"ellipse({xk:.3f},{yk:.3f},{a:.3f},{b:.3f},{theta:.3f}) # color=violet \n")
                elif None not in (xk, yk, a):
                    f.write(f"circle({xk:.3f},{yk:.3f},{a:.3f}) # color=violet \n")
                elif None not in (xk, yk):
                    f.write(f"circle({xk:.3f},{yk:.3f},14) # color=violet \n")

        print(f"DS9 region file saved to: {region_filename}")
    
    def plotting_images(self, image_sub, apertures, annulus_apertures, output_dir, catalog):
        norm = simple_norm(image_sub, 'sqrt', vmin= -0.01, vmax= 0.5)
        plt.figure(figsize=(10, 8))
        plt.imshow(image_sub, cmap='inferno', norm=norm, origin='lower')

        for aperture in apertures:
            aperture.plot(color='cyan', lw=0.5)

        #for annulus in annulus_apertures:
            #annulus.plot(color='cyan', lw=0.5, alpha=0.7)

        for src in catalog:
            kron_ap = src.kron_aperture
            if kron_ap is not None:
                kron_ap.plot(color='violet', lw=0.5, alpha=0.8)


        plt.title('Detected Sources with Apertures')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.colorbar(label='Intensity')
        plt.savefig(os.path.join(output_dir, 'aperture_overlay.png'), dpi=300, bbox_inches='tight')

    def save_for_plots(self, output_dir, catalog, ):


    def printing_storing(self, phot_table, output_dir):
        print("Photometry complete. Results:")
        print(phot_table)
        phot_table.write(os.path.join(output_dir, 'photometry_results.csv'),format='csv', overwrite=True)
        print("Photometry results saved to 'photometry_results.csv")

Photometry("/mnt/c/Users/Coque/Desktop/astronomy_research/G165/cutouts/cutout_f444.fits", '/mnt/c/users/Coque/Desktop/astronomy_research/G165/cutouts/cutout_f444_wht.fits')






