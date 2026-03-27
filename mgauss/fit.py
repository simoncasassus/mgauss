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

    crval1, crval2 = wcs_data.wcs.crval
    ra_deg = crval1 + p["ra_offset_arcsec"] / (3600.0 * np.cos(np.radians(crval2)))
    dec_deg = crval2 + p["dec_offset_arcsec"] / 3600.0
    x_pix, y_pix = wcs_data.wcs_world2pix(ra_deg, dec_deg, 0)

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
    hdr=None,
    pixscale=0.01,  # arcsec/pixel
    rms0=None,  # Jy/beam
    init_phys=None,  #
    optimizer="least_squares",
    run_sampler=True,
    uplimit_gauss_stddev=1.0,
    correlated_noise_correction=True,
    with_plot=True,
    fit_flags=None,
):
    bunit = ""
    if hdr and "BUNIT" in hdr:
        bunit = hdr["BUNIT"]

    crval1, crval2 = wcs_data.wcs.crval

    if rms0 is None:
        rms0 = np.std(im_data[np.isfinite(im_data)])

    if "BMAJ" in hdr.keys() and "BMIN" in hdr.keys():
        beam_maj_arcsec = hdr["BMAJ"]*3600.
        beam_min_arcsec = hdr["BMIN"]*3600.
        beam_area_arcsec2 = (np.pi * beam_maj_arcsec * beam_min_arcsec) / (
            4 * np.log(2)
        )
        pixel_area_arcsec2 = pixscale**2
        pixels_per_beam = beam_area_arcsec2 / pixel_area_arcsec2

        if correlated_noise_correction:
            ## sqrt(2) for a Gaussian beam. If the beam were a top-hat, we would use pixels_per_beam directly without the sqrt(2) factor.
            error_correction_factor = np.sqrt(pixels_per_beam)/np.sqrt(2)
            print(f"\n--- Correlated Noise Correction ---")
            print(f"Number of pixels per beam area: {pixels_per_beam:.2f}")
            print(f"Uncertainty correction factor: {error_correction_factor:.2f}")
            rms0 *= error_correction_factor
            print(f"Effective RMS for fitting: {rms0:.3e}")
    else:
        print("\n--- No Correlated Noise Correction Applied ---")

    print("\n--- Initial Guess Physical Parameters ---")
    physical_keys = [
        "amplitude",
        "ra_offset_arcsec",
        "dec_offset_arcsec",
        "fwhm_maj_arcsec",
        "fwhm_min_arcsec",
        "pa_deg",
    ]
    for key in physical_keys:
        if key == "amplitude":
            print(f"{key:>20}: {init_phys[key]:.3e}")
        elif key in ["ra_offset_arcsec", "dec_offset_arcsec"]:
            print(f"{key:>20}: {init_phys[key]:.6f} arcsec")
        else:
            print(f"{key:>20}: {init_phys[key]:.6f}")

    image_shape = im_data.shape
    L11_0, L21_0, L22_0 = physical_to_cholesky(
        init_phys["fwhm_maj_arcsec"], init_phys["fwhm_min_arcsec"], init_phys["pa_deg"]
    )

    # Order here is strictly maintained for matrix mapping later
    initial_params = {
        "amplitude": init_phys["amplitude"],
        "ra_offset_arcsec": init_phys["ra_offset_arcsec"],
        "dec_offset_arcsec": init_phys["dec_offset_arcsec"],
        "L11": L11_0,
        "L21": L21_0,
        "L22": L22_0,
    }

    # Set to False to FIT the parameter, True to FIX it.
    fixed_flags = {
        "amplitude": False,
        "ra_offset_arcsec": False,
        "dec_offset_arcsec": False,
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
            "ra_offset_arcsec": (-np.inf, np.inf),
            "dec_offset_arcsec": (-np.inf, np.inf),
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

        all_internal_keys = [
            "amplitude",
            "ra_offset_arcsec",
            "dec_offset_arcsec",
            "L11",
            "L21",
            "L22",
        ]
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
            "ra_offset_arcsec",
            "dec_offset_arcsec",
            "fwhm_maj_arcsec",
            "fwhm_min_arcsec",
            "pa_deg",
        ]
        final_phys_vals = [
            final_params["amplitude"],
            final_params["ra_offset_arcsec"],
            final_params["dec_offset_arcsec"],
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

            elif key in ["ra_offset_arcsec", "dec_offset_arcsec"]:
                err_str = (
                    f"+/- {phys_err_dict[key]:.6f} arcsec"
                    if not is_param_fixed
                    else "(Fixed)"
                )
                print(f"{key:>20}: {phys_val_dict[key]:.6f} arcsec {err_str}")

            else:
                err_str = (
                    f"+/- {phys_err_dict[key]:.6f}" if not is_param_fixed else "(Fixed)"
                )
                print(f"{key:>20}: {phys_val_dict[key]:.6f} {err_str}")

        chi2 = np.sum(result.fun**2)
        print(f"\n{'chi2':>15}: {chi2:.6f}")

        # --- Integrated Flux Calculation ---
        amp = phys_val_dict["amplitude"]
        amp_err = phys_err_dict["amplitude"]
        maj = phys_val_dict["fwhm_maj_arcsec"]
        maj_err = phys_err_dict["fwhm_maj_arcsec"]
        mini = phys_val_dict["fwhm_min_arcsec"]
        mini_err = phys_err_dict["fwhm_min_arcsec"]

        flux_factor = np.pi / (4 * np.log(2))
        flux = flux_factor * amp * maj * mini
        flux_label = "flux "

        if "beam" in bunit.lower():
            flux_label = "flux_jy"
            if beam_area_arcsec2 > 0:
                flux /= beam_area_arcsec2

        term_amp = (amp_err / amp) ** 2 if amp != 0 else 0
        term_maj = (maj_err / maj) ** 2 if maj != 0 else 0
        term_mini = (mini_err / mini) ** 2 if mini != 0 else 0

        if amp_err == 0 and maj_err == 0 and mini_err == 0:
            flux_err_str = "(Fixed)"
        else:
            flux_err = abs(flux) * np.sqrt(term_amp + term_maj + term_mini)
            flux_err_str = f"+/- {flux_err:.3e}"

        print(f"{flux_label:>15}: {flux:.3e} {flux_err_str}")

    elif optimizer == "nautilus":
        # Bounding prior volume (+/- 0.01 deg is approx 36 arcsec search box for centroid)

        # For Nautilus, especially with high effective noise, the likelihood can be
        # very flat. We use tighter priors around the initial guess to guide the
        # sampler to the region of interest.
        # We set the prior range to be 3x the initial Cholesky parameter values.
        l_max = max(abs(L11_0), abs(L21_0), abs(L22_0)) * 2.0
        l_min = min(abs(L11_0), abs(L22_0)) * 0.5

        print("l_max", l_max, "l_min", l_min)

        # Bounding prior volume
        prior_bounds = {
            "amplitude": (
                0.0,
                init_phys["amplitude"] * 2.0,
            ),  # You might want to scale this dynamically too!
            "ra_offset_arcsec": (
                init_phys["ra_offset_arcsec"] - l_max,
                init_phys["ra_offset_arcsec"] + l_max,
            ),
            "dec_offset_arcsec": (
                init_phys["dec_offset_arcsec"] - l_max,
                init_phys["dec_offset_arcsec"] + l_max,
            ),
            "L11": (l_min, l_max),
            "L21": (-l_max, l_max),
            "L22": (l_min, l_max),
        }

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

        sampler = Sampler(
            prior_transform,
            log_likelihood,
            n_dim=len(free_keys),
            n_live=len(free_keys) * 150,
            n_networks=16,
        )

        if run_sampler:
            sampler.run(verbose=True, n_eff=1000)
            points, log_w, log_l = sampler.posterior()
            np.save("dresult_points", points)
            np.save("dresult_log_w", log_w)
            np.save("dresult_log_l", log_l)
        else:
            points = np.load("dresult_points.npy", allow_pickle=True)
            log_w = np.load("dresult_log_w.npy", allow_pickle=True)
            log_l = np.load("dresult_log_l.npy", allow_pickle=True)

        chi2 = -2 * np.max(log_l)
        weights = np.exp(log_w - np.max(log_w))

        # Map samples back to physical domain
        physical_samples_list = []
        for i in range(len(points)):
            p = fixed_dict.copy()
            for k, val in zip(free_keys, points[i]):
                p[k] = val

            maj, min_val, pa = cholesky_to_physical(p["L11"], p["L21"], p["L22"])
            physical_samples_list.append(
                [
                    p["amplitude"],
                    p["ra_offset_arcsec"],
                    p["dec_offset_arcsec"],
                    maj,
                    min_val,
                    pa,
                ]
            )

        physical_samples = np.array(physical_samples_list)
        all_physical_keys = [
            "amplitude",
            "ra_offset_arcsec",
            "dec_offset_arcsec",
            "fwhm_maj_arcsec",
            "fwhm_min_arcsec",
            "pa_deg",
        ]

        plot_keys = []
        if "amplitude" in free_keys:
            plot_keys.append("amplitude")
        if "ra_offset_arcsec" in free_keys:
            plot_keys.append("ra_offset_arcsec")
        if "dec_offset_arcsec" in free_keys:
            plot_keys.append("dec_offset_arcsec")
        if any(k in free_keys for k in ["L11", "L21", "L22"]):
            plot_keys.extend(["fwhm_maj_arcsec", "fwhm_min_arcsec", "pa_deg"])

        plot_indices = [all_physical_keys.index(k) for k in plot_keys]
        plot_samples = physical_samples[:, plot_indices]

        phys_val_dict = {}
        phys_err_dict = {}

        print("\n--- Physical Posterior Results (Nautilus) ---")
        for i, key in enumerate(plot_keys):
            quantiles = corner.quantile(
                plot_samples[:, i], [0.16, 0.5, 0.84], weights=weights
            )
            median = quantiles[1]
            err_minus = median - quantiles[0]
            err_plus = quantiles[2] - median

            phys_val_dict[key] = median
            phys_err_dict[key] = (err_plus + err_minus) / 2.0

            if key == "fwhm_maj_arcsec":
                final_params["maj_eval"] = median
            elif key == "fwhm_min_arcsec":
                final_params["min_eval"] = median
            elif key == "pa_deg":
                final_params["pa_eval"] = median
            else:
                final_params[key] = median

            if key == "amplitude":
                print(f"{key:>20}: {median:.3e} (+{err_plus:.3e} / -{err_minus:.3e})")
            elif key in ["ra_offset_arcsec", "dec_offset_arcsec"]:
                print(
                    f"{key:>20}: {median:.6f} arcsec "
                    f"(+{err_plus:.6f} / -{err_minus:.6f} arcsec)"
                )
            else:
                print(f"{key:>20}: {median:.6f} (+{err_plus:.6f} / -{err_minus:.6f})")

        print(f"\n{'chi2':>15}: {chi2:.6f}")

        # Add fixed parameters to dicts for flux calculation
        if "amplitude" not in phys_val_dict:
            phys_val_dict["amplitude"] = initial_params["amplitude"]
            phys_err_dict["amplitude"] = 0.0

        if "fwhm_maj_arcsec" not in phys_val_dict:
            maj, mini, pa = cholesky_to_physical(
                initial_params["L11"], initial_params["L21"], initial_params["L22"]
            )
            phys_val_dict["fwhm_maj_arcsec"] = maj
            phys_err_dict["fwhm_maj_arcsec"] = 0.0
            phys_val_dict["fwhm_min_arcsec"] = mini
            phys_err_dict["fwhm_min_arcsec"] = 0.0

        # --- Integrated Flux Calculation ---
        amp = phys_val_dict["amplitude"]
        amp_err = phys_err_dict["amplitude"]
        maj = phys_val_dict["fwhm_maj_arcsec"]
        maj_err = phys_err_dict["fwhm_maj_arcsec"]
        mini = phys_val_dict["fwhm_min_arcsec"]
        mini_err = phys_err_dict["fwhm_min_arcsec"]

        flux_factor = np.pi / (4 * np.log(2))
        flux = flux_factor * amp * maj * mini
        flux_label = "flux_jy_arcsec2"

        if "beam" in bunit.lower():
            flux_label = "flux_jy"
            if beam_area_arcsec2 > 0:
                flux /= beam_area_arcsec2

        term_amp = (amp_err / amp) ** 2 if amp != 0 else 0
        term_maj = (maj_err / maj) ** 2 if maj != 0 else 0
        term_mini = (mini_err / mini) ** 2 if mini != 0 else 0

        if amp_err == 0 and maj_err == 0 and mini_err == 0:
            flux_err_str = "(Fixed)"
        else:
            flux_err = abs(flux) * np.sqrt(term_amp + term_maj + term_mini)
            flux_err_str = f"+/- {flux_err:.3e}"

        print(f"{flux_label:>15}: {flux:.3e} {flux_err_str}")

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

    ra_offset = final_params["ra_offset_arcsec"]
    dec_offset = final_params["dec_offset_arcsec"]
    ra_deg = crval1 + ra_offset / (3600.0 * np.cos(np.radians(crval2)))
    dec_deg = crval2 + dec_offset / 3600.0

    fit_x_pix, fit_y_pix = wcs_data.wcs_world2pix(ra_deg, dec_deg, 0)
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
        "ra_offset_arcsec": final_params.get("ra_offset_arcsec", np.nan),
        "dec_offset_arcsec": final_params.get("dec_offset_arcsec", np.nan),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
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
        plt.savefig("fit_results.png")
        plt.show()

    return best_fit_image, final_phys_params


