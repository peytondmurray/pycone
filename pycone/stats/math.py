import numpy as np
import pandas as pd


def backward_integral(f: np.ndarray, t0: int | float) -> np.ndarray:
    """Sum the values f[i-t0] to f[i], and return the resulting array.

    Parameters
    ----------
    f : np.ndarray
        Temperature
    t0 : int | float
        Number of days to sum backward from each date

    Returns
    -------
    np.ndarray
        Sum of the temperature backward from each date
    """
    t0 = int(t0)
    width = t0 // 2
    result = np.full_like(f, np.nan, dtype=float)
    average = np.convolve(f, np.ones(shape=(t0,), dtype="float"), mode="same")

    average[:width] = np.nan
    average[-width:] = np.nan

    result[width:] = average[:-width]
    return result


def mavg(f: np.ndarray, width: float | int, lag: float | int) -> np.ndarray:
    """Calculate a lagged moving average of the dataset.

    Parameters
    ----------
    f : np.ndarray
        Data to calculate a lagged moving average of
    width : float | int
        The moving average is calculated by convolution with a flat kernel
        of size 2*width + 1 (so the moving average is always centered on the
        original data point). Floats are cast to int first
    lag : float | int
        Number of days to shift the moving average

    Returns
    -------
    np.ndarray
        Lagged moving average of `f`. The shape is the same as `f`, but values
        at the edge of the dataset are set to `np.nan`
    """
    width = int(width)
    lag = int(lag)

    window = 2 * width + 1
    average = np.convolve(f, np.ones(shape=(window,), dtype=float), mode="same") / window

    # Mask off the convolution at the edge
    average[:width] = np.nan
    average[-width:] = np.nan

    result = np.full_like(f, np.nan, dtype=float)
    result[lag:] = average[:-lag]
    return result


def lagged(c: np.ndarray, lag: int | float) -> np.ndarray:
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
    lag = int(lag)

    result = np.full_like(c, np.nan, dtype=float)
    result[lag:] = c[:-lag]
    return result


def fahrenheit_to_kelvin(data: pd.Series) -> pd.Series:
    """Conver fahrenheit to kelvin.

    Parameters
    ----------
    data : pd.Series
        Fahrenheit temperatures

    Returns
    -------
    pd.Series
        Kelvin temperatures
    """
    return (data - 32) * 5 / 9 + 273.15


def kelvin_to_fahrenheit(data: pd.Series) -> pd.Series:
    """Convert kelvin to fahrenheit.

    Parameters
    ----------
    data : pd.Series
        Kelvin temperatures

    Returns
    -------
    pd.Series
        Fahrenheit temperatures
    """
    return (data - 273.15) * 9 / 5 + 32
