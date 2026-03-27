import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from fit import fit


def main():
    # ==========================================
    # 0. SELECT OPTIMIZER ('least_squares' or 'nautilus')
    # ==========================================
    # optimizer = "nautilus"
    optimizer = "least_squares"
    # --- 1. Load FITS template ---
    hdu = fits.open("view_whole.fits")
    image_shape = hdu[0].data.shape
    hdr0 = hdu[0].header

    # im_data = hdu[0].data
    im_data = np.squeeze(hdu[0].data)
    image_shape = im_data.shape
    wcs_data = WCS(hdr0)

    pixscale = np.abs(hdr0["CDELT1"]) * 3600.0
    rms0 = 6.563e-05 / (9.407e-05 / 6.563e-05)
    print("rms0", rms0)
    # --- 2. Initial Physical Parameters ---
    init_ra_abs = 212.04201745
    init_dec_abs = -41.39805308
    crval1, crval2 = wcs_data.wcs.crval
    ra_offset = (init_ra_abs - crval1) * 3600.0 * np.cos(np.radians(crval2))
    dec_offset = (init_dec_abs - crval2) * 3600.0

    init_phys = {
        "amplitude": 3.391e-04,
        "ra_offset_arcsec": ra_offset,
        "dec_offset_arcsec": dec_offset,
        "fwhm_maj_arcsec": 0.064,
        "fwhm_min_arcsec": 0.058,
        "pa_deg": -88.3,
    }
    uplimit_gauss_stddev = init_phys["fwhm_maj_arcsec"] * 2.0

    fit_flags = {
        "amplitude": False,
        "ra_offset_arcsec": False,
        "dec_offset_arcsec": False,
        "L11": False,
        "L21": False,
        "L22": False,
    }

    fit(
        im_data=im_data,
        wcs_data=wcs_data,
        hdr=hdr0,
        pixscale=pixscale,
        rms0=rms0,
        init_phys=init_phys,
        fit_flags=fit_flags,
        optimizer=optimizer,
        with_plot=True,
        run_sampler=True,  # Set to True to run Nautilus sampling, False to load previous results
        uplimit_gauss_stddev=uplimit_gauss_stddev,
    )


if __name__ == "__main__":
    main()
