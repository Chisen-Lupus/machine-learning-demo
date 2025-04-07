#%% General function for image processing and machine learning preparation

from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy.signal import convolve2d
import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from astropy.coordinates import SkyCoord

def get_sci_cutout_with_matching_wcs(fits_path, target_wcs):
    """
    Reprojects the 'SCI' HDU of a FITS file onto a given WCS.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file containing the 'SCI' HDU.
    target_wcs : astropy.wcs.WCS
        The target WCS to reproject onto. Must have `array_shape` defined.

    Returns
    -------
    reprojected_data : numpy.ndarray
        Reprojected image data matching the target WCS.
    new_wcs : astropy.wcs.WCS
        The WCS object corresponding to the reprojected data (same as input).
    """
    with fits.open(fits_path) as hdul:
        sci_hdu = hdul['SCI']
        input_data = sci_hdu.data
        input_wcs = WCS(sci_hdu.header)

    shape_out = target_wcs.array_shape
    if shape_out is None:
        raise ValueError('target_wcs must have array_shape defined (e.g. from a Cutout2D or manually set).')

    reprojected_data, _ = reproject_interp(
        (input_data, input_wcs),
        output_projection=target_wcs,
        shape_out=shape_out
    )

    return reprojected_data


def iterative_nan_fill(array, kernel=None):
    if kernel is None:
        kernel = np.array([[0, 0.25, 0],
                           [0.25, 0, 0.25],
                           [0, 0.25, 0]])
    filled_array = array.copy()
    while np.isnan(filled_array).any():
        num_neighbors = convolve2d(~np.isnan(filled_array), kernel, mode='same', boundary='symm')
        neighbor_sum = convolve2d(np.nan_to_num(filled_array, nan=0), kernel, mode='same', boundary='symm')
        mask = np.isnan(filled_array) & (num_neighbors > 0)
        avg_neighbors = np.zeros_like(filled_array)
        valid = num_neighbors > 0
        avg_neighbors[valid] = neighbor_sum[valid] / num_neighbors[valid]
        filled_array[mask] = avg_neighbors[mask]
    return filled_array

# grism_data_filled = iterative_nan_fill(grism_data)

def normalize_to_95_percent_range(arr):
    """
    Normalize an array so that the 2.5th and 97.5th percentiles map to 0 and 1, respectively.
    Values below or above this range will be clipped to 0 or 1.
    
    Parameters
    ----------
    arr : numpy.ndarray
        Input array to normalize.

    Returns
    -------
    norm_arr : numpy.ndarray
        Normalized array with values in [0, 1], where the 2.5th percentile maps to 0 
        and the 97.5th percentile maps to 1.
    """
    p_low, p_high = np.percentile(arr, [2.5, 97.5])
    norm_arr = (arr - p_low) / (p_high - p_low)
    norm_arr = np.clip(norm_arr, 0, 1)
    return norm_arr

def calc_elliptical_source_polygon(wcs, ra, dec, a_pix, b_pix, theta_deg):
    scales = (u.Quantity([1, 0], u.pixel), u.Quantity([0, 1], u.pixel))
    pix_arcsec = proj_plane_pixel_scales(wcs) * u.deg.to(u.arcsec)

    a_arcsec = a_pix * pix_arcsec[0]
    b_arcsec = b_pix * pix_arcsec[1]
    theta_rad = np.deg2rad(theta_deg)

    # Parametric ellipse in sky coords
    phi = np.linspace(0, 2*np.pi, 16)
    dra = ((a_arcsec * np.cos(phi) * np.cos(theta_rad) -
            b_arcsec * np.sin(phi) * np.sin(theta_rad)) * u.arcsec).to(u.deg)
    ddec = ((a_arcsec * np.cos(phi) * np.sin(theta_rad) +
            b_arcsec * np.sin(phi) * np.cos(theta_rad)) * u.arcsec).to(u.deg)

    ra_poly = ra + dra
    dec_poly = dec + ddec

    # Convert to pixel
    coords = SkyCoord(ra_poly, dec_poly)
    x_poly, y_poly = wcs.world_to_pixel(coords)

    return x_poly, y_poly

#%% Used for ML labeling
