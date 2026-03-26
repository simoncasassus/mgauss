#!/usr/bin/env python3

import sys
import numpy as np
from astropy.io import fits
from mgauss import physical_to_cholesky, evaluate_gaussian_cholesky

def make_beam_image(input_fits='view.fits', output_fits='beam.fits'):
    """
    Creates a FITS file of a Gaussian beam based on the header of an input file.

    Reads BMAJ, BMIN, and BPA from the header of `input_fits`, generates a
    2D Gaussian beam centered in the image, and saves it to `output_fits`
    with the same header and dimensions as the input.
    """
    try:
        hdu = fits.open(input_fits)[0]
    except FileNotFoundError:
        print(f"Error: Input file '{input_fits}' not found.")
        return
    except IndexError:
        print(f"Error: No valid HDU found in '{input_fits}'.")
        return

    hdr = hdu.header
    shape = hdu.data.shape

    # Check for beam parameters in header
    beam_keys = ['BMAJ', 'BMIN', 'BPA']
    if not all(key in hdr for key in beam_keys):
        print("Error: BMAJ, BMIN, or BPA not found in the FITS header.")
        return

    # Get beam parameters (BMAJ/BMIN are in degrees)
    fwhm_maj_deg = hdr['BMAJ']
    fwhm_min_deg = hdr['BMIN']
    pa_deg = hdr['BPA']

    # Get pixel scale from header (degrees/pixel)
    if 'CDELT2' not in hdr:
        print("Error: CDELT2 keyword not found in header for pixel scale.")
        return
    pixscale_arcsec = np.abs(hdr['CDELT2']) * 3600.0

    # Convert FWHM from degrees to arcseconds
    fwhm_maj_arcsec = fwhm_maj_deg * 3600.0
    fwhm_min_arcsec = fwhm_min_deg * 3600.0

    # Convert physical parameters to Cholesky elements (in arcsec)
    L11_arcsec, L21_arcsec, L22_arcsec = physical_to_cholesky(
        fwhm_maj_arcsec, fwhm_min_arcsec, pa_deg
    )

    # Convert Cholesky elements from arcsec to pixels
    L11_pix = L11_arcsec / pixscale_arcsec
    L21_pix = L21_arcsec / pixscale_arcsec
    L22_pix = L22_arcsec / pixscale_arcsec

    # Define image center
    ny, nx = shape
    centroid_x = (nx - 1) / 2.0
    centroid_y = (ny - 1) / 2.0

    # Generate the Gaussian beam image
    beam_data = evaluate_gaussian_cholesky(
        shape, 1.0, centroid_x, centroid_y, L11_pix, L21_pix, L22_pix
    )

    # Create new HDU and write to file
    new_hdu = fits.PrimaryHDU(data=beam_data, header=hdr)
    new_hdu.writeto(output_fits, overwrite=True)
    print(f"Beam image successfully created and saved to '{output_fits}'.")


if __name__ == "__main__":
    make_beam_image()
