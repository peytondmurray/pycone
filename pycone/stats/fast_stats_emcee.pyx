cimport numpy
import numpy as np

cimport cython

@cython.boundscheck(False)
cpdef mavg(double[:] f, double width, double lag):
    """Calculate a lagged moving average of the dataset.

    Parameters
    ----------
    f : np.ndarray
        Data to calculate a lagged moving average of
    width : float
        The moving average is calculated by convolution with a flat kernel
        of size 2*width + 1 (so the moving average is always centered on the
        original data point). Floats are cast to int first
    lag : float
        Number of days to shift the moving average

    Returns
    -------
    np.ndarray
        Lagged moving average of `f`. The shape is the same as `f`, but values
        at the edge of the dataset are set to `np.nan`
    """
    width_int = int(width)
    lag_int = int(lag)

    window = 2 * width_int + 1
    average = np.convolve(f, np.ones(shape=(window,), dtype=float), mode="same") / window

    # Mask off the convolution at the edge
    average[:width_int] = np.nan
    average[-width_int:] = np.nan

    result = np.full(f.shape[0], np.nan, dtype=float)
    result[:-lag_int] = average[lag_int:]
    return result


cpdef lagged(long[:] c, float lag):
    """Shift the array `c` by `lag` number of days backward.

    That is,

        c[i] = lagged(c, lag)[i - lag]

    Parameters
    ----------
    c : np.ndarray
        Time series data to shift
    lag : int | float
        Magnitude of the shift, in days. Floats are cast to int first

    Returns
    -------
    np.ndarray
        An array which is the same shape as `c` but shifted by `lag` days.
        Values at the edge of the dataset are set to `np.nan`
    """
    lag_int = int(lag)

    result = np.full(c.shape[0], np.nan, dtype=float)
    result[:-lag_int] = c[lag_int:]
    return result


cpdef log_probability(
    theta, float[:] f, long[:] c, model
):
    """Calculate the log posterior.

    See https://python.arviz.org/en/stable/getting_started/ConversionGuideEmcee.html
    for why two values are returned here. In short it is to keep track of the log
    likelihood at each sample, allowing us to unlock sample stats.

    Parameters
    ----------
    theta : tuple
        Model parameter vector
    f : np.ndarray
        Temperature
    c : np.ndarray
        Cone number
    model : Model
        Model for which the log probability is to be calculated

    Returns
    -------
    (float, np.ndarray)
        Log posterior and pointwise log likelihood
    """
    lp = model.log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    log_likelihood_vect = model.log_likelihood_vector(theta, f, c)
    log_likelihood = np.nansum(log_likelihood_vect)

    return lp + log_likelihood
