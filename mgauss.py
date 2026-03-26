import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.wcs import WCS
from scipy.optimize import least_squares
import corner
from nautilus import Sampler


def physical_to_cholesky(fwhm_maj, fwhm_min, pa_deg):
    """Converts physical Gaussian parameters into Cholesky lower-triangular elements."""
    sigma_maj = fwhm_maj / (2 * np.sqrt(2 * np.log(2)))
    sigma_min = fwhm_min / (2 * np.sqrt(2 * np.log(2)))

    theta_rad = np.radians(90.0 + pa_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    Lambda = np.array([[sigma_maj**2, 0], [0, sigma_min**2]])

    C = R @ Lambda @ R.T
    L = np.linalg.cholesky(C)

    return L[0, 0], L[1, 0], L[1, 1]


def cholesky_to_physical(L11, L21, L22):
    """Converts Cholesky elements back into physical FWHMs and Position Angle."""
    c_xx = L11**2
    c_xy = L11 * L21
    c_yy = L21**2 + L22**2

    trace = c_xx + c_yy
    det = c_xx * c_yy - c_xy**2
    diff = np.sqrt(np.maximum(0.0, trace**2 - 4 * det))

    lambda1 = (trace + diff) / 2.0
    lambda2 = (trace - diff) / 2.0

    sigma_maj = np.sqrt(lambda1)
    sigma_min = np.sqrt(lambda2)

    factor = 2 * np.sqrt(2 * np.log(2))
    fwhm_maj = sigma_maj * factor
    fwhm_min = sigma_min * factor

    theta_rad = 0.5 * np.arctan2(2 * c_xy, c_xx - c_yy)
    pa_deg = np.degrees(theta_rad) - 90.0
    pa_deg = (pa_deg + 90) % 180 - 90

    return fwhm_maj, fwhm_min, pa_deg


def get_cholesky_jacobian(L11, L21, L22, eps=1e-8):
    """
    Numerically computes the Jacobian matrix of the Cholesky-to-Physical
    transformation using central finite differences.
    """
    J = np.zeros((3, 3))
    params = np.array([L11, L21, L22])

    for i in range(3):
        p_plus = params.copy()
        p_plus[i] += eps
        maj_p, min_p, pa_p = cholesky_to_physical(*p_plus)

        p_minus = params.copy()
        p_minus[i] -= eps
        maj_m, min_m, pa_m = cholesky_to_physical(*p_minus)

        J[0, i] = (maj_p - maj_m) / (2 * eps)
        J[1, i] = (min_p - min_m) / (2 * eps)

        dpa = pa_p - pa_m
        dpa = (dpa + 90) % 180 - 90
        J[2, i] = dpa / (2 * eps)

    return J


def evaluate_gaussian_cholesky(shape, amplitude, centroid_x, centroid_y, L11, L21, L22):
    """Evaluates a 2D Gaussian over a grid using Cholesky parameters in PIXEL units."""
    y_idx, x_idx = np.indices(shape)
    x_shifted = x_idx.flatten() - centroid_x
    y_shifted = y_idx.flatten() - centroid_y
    coords = np.vstack((x_shifted, y_shifted))

    L = np.array([[L11, 0.0], [L21, L22]])
    C = L @ L.T
    C_inv = np.linalg.inv(C)

    quad_form = np.sum(coords * (C_inv @ coords), axis=0)
    gaussian_flat = amplitude * np.exp(-0.5 * quad_form)

    return gaussian_flat.reshape(shape)


def gaussian_residuals(
    free_params_array, shape, data, free_keys, fixed_dict, wcs_data, pixscale, rms
):
    """Calculates the 1D array of noise-weighted residuals."""
    p = fixed_dict.copy()
    for key, val in zip(free_keys, free_params_array):
        p[key] = val

    x_pix, y_pix = wcs_data.wcs_world2pix(p["ra"], p["dec"], 0)

    L11_pix = p["L11"] / pixscale
    L21_pix = p["L21"] / pixscale
    L22_pix = p["L22"] / pixscale

    model = evaluate_gaussian_cholesky(
        shape, p["amplitude"], float(x_pix), float(y_pix), L11_pix, L21_pix, L22_pix
    )
    return ((model - data) / rms).flatten()


def fit(
    im_data=None,
    wcs_data=None,
    pixscale=0.01,  # arcsec/pixel
    rms0=None,  # Jy/beam
    init_phys=None,  #
    optimizer="least_squares",
    run_sampler=True,
    uplimit_gauss_stddev=1.0,
    centroid_domain=1 / 3600,  # 1arcsec
    with_plot=True,
    fit_flags=None,
):
    if rms0 is None:
        rms0 = np.std(im_data[np.isfinite(im_data)])

    print("\n--- Initial Guess Physical Parameters ---")
    physical_keys = [
        "amplitude",
        "ra",
        "dec",
        "fwhm_maj_arcsec",
        "fwhm_min_arcsec",
        "pa_deg",
    ]
    for key in physical_keys:
        if key == "amplitude":
            print(f"{key:>15}: {init_phys[key]:.3e}")
        elif key in ["ra", "dec"]:
            print(f"{key:>15}: {init_phys[key]:.6f} deg")
        else:
            print(f"{key:>15}: {init_phys[key]:.6f}")

    image_shape = im_data.shape
    L11_0, L21_0, L22_0 = physical_to_cholesky(
        init_phys["fwhm_maj_arcsec"], init_phys["fwhm_min_arcsec"], init_phys["pa_deg"]
    )

    # Order here is strictly maintained for matrix mapping later
    initial_params = {
        "amplitude": init_phys["amplitude"],
        "ra": init_phys["ra"],
        "dec": init_phys["dec"],
        "L11": L11_0,
        "L21": L21_0,
        "L22": L22_0,
    }

    # Set to False to FIT the parameter, True to FIX it.
    fixed_flags = {
        "amplitude": False,
        "ra": False,  # Now fitting the centroid RA!
        "dec": False,  # Now fitting the centroid Dec!
        "L11": False,
        "L21": False,
        "L22": False,
    }

    if fit_flags is not None:
        fixed_flags.update(fit_flags)

    free_keys = [k for k, is_fixed in fixed_flags.items() if not is_fixed]
    fixed_dict = {
        k: initial_params[k] for k, is_fixed in fixed_flags.items() if is_fixed
    }

    print(f"Fitting {len(free_keys)} free parameters using {optimizer}...")
    final_params = fixed_dict.copy()

    # ==========================================
    # OPTIMIZER BLOCK
    # ==========================================

    if optimizer == "least_squares":
        ls_bounds = {
            "amplitude": (0.0, np.inf),
            "ra": (0.0, 360.0),
            "dec": (-90.0, 90.0),
            "L11": (1e-6, np.inf),
            "L21": (-np.inf, np.inf),
            "L22": (1e-6, np.inf),
        }

        initial_guess_array = [initial_params[k] for k in free_keys]
        lower_bounds = [ls_bounds[k][0] for k in free_keys]
        upper_bounds = [ls_bounds[k][1] for k in free_keys]

        result = least_squares(
            gaussian_residuals,
            x0=initial_guess_array,
            bounds=(lower_bounds, upper_bounds),
            args=(
                image_shape,
                im_data,
                free_keys,
                fixed_dict,
                wcs_data,
                pixscale,
                rms0,
            ),
            method="trf",
        )

        final_params.update(dict(zip(free_keys, result.x)))
        maj_f, min_f, pa_f = cholesky_to_physical(
            final_params["L11"], final_params["L21"], final_params["L22"]
        )

        # --- Error Propagation (Delta Method) ---
        cov_free = np.linalg.pinv(result.jac.T @ result.jac)

        all_internal_keys = ["amplitude", "ra", "dec", "L11", "L21", "L22"]
        cov_full = np.zeros((6, 6))
        free_indices = [all_internal_keys.index(k) for k in free_keys]

        for i, idx_i in enumerate(free_indices):
            for j, idx_j in enumerate(free_indices):
                cov_full[idx_i, idx_j] = cov_free[i, j]

        # Jacobian Transformation matrix H
        H = np.eye(6)
        J_chol_phys = get_cholesky_jacobian(
            final_params["L11"], final_params["L21"], final_params["L22"]
        )
        H[3:6, 3:6] = J_chol_phys

        cov_phys = H @ cov_full @ H.T
        err_phys = np.sqrt(np.maximum(0, np.diag(cov_phys)))

        physical_keys = [
            "amplitude",
            "ra",
            "dec",
            "fwhm_maj_arcsec",
            "fwhm_min_arcsec",
            "pa_deg",
        ]
        final_phys_vals = [
            final_params["amplitude"],
            final_params["ra"],
            final_params["dec"],
            maj_f,
            min_f,
            pa_f,
        ]

        phys_val_dict = dict(zip(physical_keys, final_phys_vals))
        phys_err_dict = dict(zip(physical_keys, err_phys))

        print("\n--- Best Fit Physical Parameters & Standard Errors ---")
        for key in physical_keys:
            is_param_fixed = phys_err_dict[key] == 0.0

            if key == "amplitude":
                err_str = (
                    f"+/- {phys_err_dict[key]:.3e}" if not is_param_fixed else "(Fixed)"
                )
                print(f"{key:>15}: {phys_val_dict[key]:.3e} {err_str}")

            elif key in ["ra", "dec"]:
                # Value in degrees, but convert the error to arcseconds for readability
                err_in_arcsec = phys_err_dict[key] * 3600.0
                err_str = (
                    f"+/- {err_in_arcsec:.6f} arcsec"
                    if not is_param_fixed
                    else "(Fixed)"
                )
                print(f"{key:>15}: {phys_val_dict[key]:.8f} deg {err_str}")

            else:
                err_str = (
                    f"+/- {phys_err_dict[key]:.6f}" if not is_param_fixed else "(Fixed)"
                )
                print(f"{key:>15}: {phys_val_dict[key]:.6f} {err_str}")

    elif optimizer == "nautilus":
        # Bounding prior volume (+/- 0.01 deg is approx 36 arcsec search box for centroid)

        # Ensure the upper limit is at least 3x the user's initial guess
        dynamic_uplimit = max(uplimit_gauss_stddev, init_phys["fwhm_maj_arcsec"] * 3.0)

        # Bounding prior volume
        prior_bounds = {
            "amplitude": (1e-6, 1e-2),  # You might want to scale this dynamically too!
            "ra": (
                init_phys["ra"] - centroid_domain,
                init_phys["ra"] + centroid_domain,
            ),
            "dec": (
                init_phys["dec"] - centroid_domain,
                init_phys["dec"] + centroid_domain,
            ),
            "L11": (1e-4, dynamic_uplimit),
            "L21": (-dynamic_uplimit, dynamic_uplimit),
            "L22": (1e-4, dynamic_uplimit),
        }

        # prior_bounds = {
        #     "amplitude": (1e-6, 1e-2),
        #     "ra": (
        #         init_phys["ra"] - centroid_domain,
        #         init_phys["ra"] + centroid_domain,
        #     ),
        #     "dec": (
        #         init_phys["dec"] - centroid_domain,
        #         init_phys["dec"] + centroid_domain,
        #     ),
        #     "L11": (1e-4, uplimit_gauss_stddev),
        #     "L21": (-uplimit_gauss_stddev, uplimit_gauss_stddev),
        #     "L22": (1e-4, uplimit_gauss_stddev),
        # }

        def prior_transform(u):
            x = np.zeros_like(u)
            for i, key in enumerate(free_keys):
                lower, upper = prior_bounds[key]
                x[i] = u[i] * (upper - lower) + lower
            return x

        def log_likelihood(free_params_array):
            res = gaussian_residuals(
                free_params_array,
                image_shape,
                im_data,
                free_keys,
                fixed_dict,
                wcs_data,
                pixscale,
                rms0,
            )
            return -0.5 * np.sum(res**2)

        sampler = Sampler(prior_transform, log_likelihood, n_dim=len(free_keys))
        if run_sampler:
            sampler.run(verbose=True)
            points, log_w, log_l = sampler.posterior()
            np.save("dresult_points", points)
            np.save("dresult_log_w", log_w)
            np.save("dresult_log_l", log_l)
        else:
            points = np.load("dresult_points.npy", allow_pickle=True)
            log_w = np.load("dresult_log_w.npy", allow_pickle=True)
            log_l = np.load("dresult_log_l.npy", allow_pickle=True)

        weights = np.exp(log_w - np.max(log_w))

        # Map samples back to physical domain
        physical_samples_list = []
        for i in range(len(points)):
            p = fixed_dict.copy()
            for k, val in zip(free_keys, points[i]):
                p[k] = val

            maj, min_val, pa = cholesky_to_physical(p["L11"], p["L21"], p["L22"])
            physical_samples_list.append(
                [p["amplitude"], p["ra"], p["dec"], maj, min_val, pa]
            )

        physical_samples = np.array(physical_samples_list)
        all_physical_keys = [
            "amplitude",
            "ra",
            "dec",
            "fwhm_maj_arcsec",
            "fwhm_min_arcsec",
            "pa_deg",
        ]

        plot_keys = []
        if "amplitude" in free_keys:
            plot_keys.append("amplitude")
        if "ra" in free_keys:
            plot_keys.append("ra")
        if "dec" in free_keys:
            plot_keys.append("dec")
        if any(k in free_keys for k in ["L11", "L21", "L22"]):
            plot_keys.extend(["fwhm_maj_arcsec", "fwhm_min_arcsec", "pa_deg"])

        plot_indices = [all_physical_keys.index(k) for k in plot_keys]
        plot_samples = physical_samples[:, plot_indices]

        print("\n--- Physical Posterior Results (Nautilus) ---")
        for i, key in enumerate(plot_keys):
            quantiles = corner.quantile(
                plot_samples[:, i], [0.16, 0.5, 0.84], weights=weights
            )
            median = quantiles[1]
            err_minus = median - quantiles[0]
            err_plus = quantiles[2] - median

            if key == "fwhm_maj_arcsec":
                final_params["maj_eval"] = median
            elif key == "fwhm_min_arcsec":
                final_params["min_eval"] = median
            elif key == "pa_deg":
                final_params["pa_eval"] = median
            else:
                final_params[key] = median


            if key == "amplitude":
                print(f"{key:>15}: {median:.3e} (+{err_plus:.3e} / -{err_minus:.3e})")
            elif key in ["ra", "dec"]:
                err_plus_arcsec = err_plus * 3600.0
                err_minus_arcsec = err_minus * 3600.0
                print(
                    f"{key:>15}: {median:.8f} deg "
                    f"(+{err_plus_arcsec:.6f} / -{err_minus_arcsec:.6f} arcsec)"
                )
            else:
                print(f"{key:>15}: {median:.6f} (+{err_plus:.6f} / -{err_minus:.6f})")

        fig = corner.corner(
            plot_samples,
            weights=weights,
            # bins=20,
            labels=plot_keys,
            color="purple",
            plot_datapoints=False,
            range=np.repeat(0.999, len(plot_keys)),
        )
        fig.savefig("triangle.png")

        # fig_corner = corner.corner(
        #     plot_samples,
        #     weights=weights,
        #     labels=plot_keys,
        #     show_titles=True,
        #     title_kwargs={"fontsize": 12},
        # )
        # fig_corner.suptitle("Physical Parameter Posteriors", fontsize=16)

    # ==========================================
    # COMMON VISUALIZATION BLOCK
    # ==========================================
    if optimizer == "least_squares":
        maj_eval, min_eval, pa_eval = cholesky_to_physical(
            final_params["L11"], final_params["L21"], final_params["L22"]
        )
    else:
        maj_eval = final_params.get("maj_eval", init_phys["fwhm_maj_arcsec"])
        min_eval = final_params.get("min_eval", init_phys["fwhm_min_arcsec"])
        pa_eval = final_params.get("pa_eval", init_phys["pa_deg"])

    fit_x_pix, fit_y_pix = wcs_data.wcs_world2pix(
        final_params["ra"], final_params["dec"], 0
    )
    L11_f, L21_f, L22_f = physical_to_cholesky(
        maj_eval / pixscale, min_eval / pixscale, pa_eval
    )

    best_fit_image = evaluate_gaussian_cholesky(
        image_shape,
        final_params["amplitude"],
        float(fit_x_pix),
        float(fit_y_pix),
        L11_f,
        L21_f,
        L22_f,
    )

    final_phys_params = {
        "amplitude": final_params.get("amplitude", np.nan),
        "ra": final_params.get("ra", np.nan),
        "dec": final_params.get("dec", np.nan),
        "fwhm_maj_arcsec": maj_eval,
        "fwhm_min_arcsec": min_eval,
        "pa_deg": pa_eval,
    }

    if with_plot:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        vmin, vmax = np.percentile(im_data, [5, 99.5])

        im1 = ax1.imshow(im_data, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.set_title("Data")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(
            best_fit_image, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
        )
        ax2.set_title(f"Best Fit Model ({optimizer})")

        ellipse = Ellipse(
            xy=(float(fit_x_pix), float(fit_y_pix)),
            width=maj_eval / pixscale,
            height=min_eval / pixscale,
            angle=90 + pa_eval,
            edgecolor="white",
            fc="None",
            lw=1,
        )
        ax2.add_patch(ellipse)

        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        residuals = (im_data - best_fit_image) / rms0
        finite_residuals = residuals[np.isfinite(residuals)]
        im3 = ax3.imshow(
            residuals,
            origin="lower",
            cmap="RdBu",
            vmin=np.min(finite_residuals),
            vmax=np.max(finite_residuals),
        )
        ax3.set_title("Residuals (S/N)")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label=r"$\sigma$")

        plt.tight_layout()
        plt.show(block=False)

    return best_fit_image, final_phys_params


if __name__ == "__main__":
    # ==========================================
    # 0. SELECT OPTIMIZER ('least_squares' or 'nautilus')
    # ==========================================
    optimizer = "nautilus"

    # --- 1. Load FITS template ---
    hdu = fits.open("view_whole.fits")
    image_shape = hdu[0].data.shape
    hdr0 = hdu[0].header

    # im_data = hdu[0].data
    im_data = np.squeeze(hdu[0].data)
    image_shape = im_data.shape
    wcs_data = WCS(hdr0)

    pixscale = np.abs(hdr0["CDELT1"]) * 3600.0
    rms0 = 6.563e-05

    # --- 2. Initial Physical Parameters ---
    init_phys = {
        "amplitude": 3.7e-4,
        "ra": 212.04201760,
        "dec": -41.39805281,
        "fwhm_maj_arcsec": 0.064,
        "fwhm_min_arcsec": 0.058,
        "pa_deg": -88.3,
    }
    uplimit_gauss_stddev = init_phys["fwhm_maj_arcsec"] * 2.0

    fit_flags = {
        "amplitude": False,
        "ra": False,  
        "dec": False,  
        "L11": True,
        "L21": True,
        "L22": True,
    }

    fit(
        im_data=im_data,
        wcs_data=wcs_data,
        pixscale=pixscale,
        rms0=rms0,
        init_phys=init_phys,
        fit_flags=fit_flags,
        optimizer=optimizer,
        run_sampler=True,  # Set to True to run Nautilus sampling, False to load previous results
        uplimit_gauss_stddev=uplimit_gauss_stddev,
    )
