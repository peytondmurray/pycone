import pathlib
from multiprocessing import Pool

import arviz as az
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as ss
import scipy.stats as st
import tarmac
import tqdm
from rich.console import Console

from .preprocess import load_data
from .util import add_days_since_start, read_data

az.style.use("default")
console = Console()


def get_data(
    weather_path: str = "weather.csv",
    cones_path: str = "cones.csv",
    impute_time: bool = False,
    site: int = 1,
) -> pd.DataFrame:
    """Get the cone and weather data.

    Parameters
    ----------
    weather_path : str
        Path to the raw weather data; this is NOT the averaged weather. Columns:

        tmean (degrees f)
        site
        year
        day_of_year

    cones_path : str
        Path to the raw cone data
    impute_time : bool
        If true, nan-valued data will be imputed for missing date ranges.
    site : int
        Site number to select. Currently, only data from a single site at a time is fit

    Returns
    -------
    pd.DataFrame
        Merged data containing the following columns:

            year
            day_of_year
            days_since_start
            t
            cones

        This data contains _all_ dates between the beginning and end of the
        observed dataset. Many dates have no measured values, and thus have a
        lot of nan values. This makes moving average computations easier later on.
    """
    fname = f"observed_site_{site}.csv"
    if pathlib.Path(fname).exists():
        console.log(f"[bold yellow]Loading existing data at {fname}")
        observed = read_data(fname)
    else:
        if pathlib.Path(weather_path).exists() and pathlib.Path(cones_path).exists():
            # Convert year+ordinal day of year to just day since the start of the dataset
            weather = read_data(weather_path)
            cones = read_data(cones_path)
        else:
            cones, weather = load_data()

        weather = weather.rename(columns={"tmean (degrees f)": "t"})
        weather = weather.loc[weather["site"] == site]
        cones = cones.loc[cones["site"] == site]

        # Note that we combine years differently than in analysis.py here (we are not using a crop
        # year). The crop year itself is a parameter of the model below.
        observed = weather.merge(cones, on="year")
        observed = add_days_since_start(observed, doy_col="day_of_year")[
            ["year", "day_of_year", "days_since_start", "t", "cones"]
        ]
        observed.to_csv(fname, index=False)

    obs = observed.rename(columns={"cones": "c"})

    if impute_time:
        return pd.DataFrame(
            {"days_since_start": np.arange(0, obs["days_since_start"].max() + 1)}
        ).merge(obs, on=["days_since_start"])

    return obs


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
    result[:-lag] = average[lag:]
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
    result[:-lag] = c[lag:]
    return result


def log_prior(theta: tuple[float, ...]) -> float:
    """Compute the log prior probability.

    Parameters
    ----------
    theta : tuple[float, ...]
        Parameters of the model

    Returns
    -------
    float
        Log prior probability
    """
    c0, alpha, beta, width_0, width_1, lag_0, lag_1, lag_2 = theta

    priors = [
        st.randint.pmf(np.floor(c0), low=0, high=1000),
        st.halfnorm.pdf(alpha, scale=10),
        st.halfnorm.pdf(beta, scale=10),
        st.randint.pmf(np.floor(width_0), low=1, high=100),
        st.randint.pmf(np.floor(width_1), low=1, high=100),
        st.randint.pmf(np.floor(lag_0), low=180, high=545),
        st.randint.pmf(np.floor(lag_1), low=550, high=910),
        st.randint.pmf(np.floor(lag_2), low=915, high=1275),
    ]

    prior = np.prod(priors)
    if prior <= 0 or np.isnan(prior):
        return -np.inf
    return np.log(prior)


def log_likelihood_vector(theta: tuple[float, ...], f: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute the log likelihood vector.

    The nansum of this vector returns the log likelihood. Data must be contiguous.

    Parameters
    ----------
    theta : tuple[float, ...]
        Parameters of the model
    f : np.ndarray
        Temperature data
    c : np.ndarray
        Cone data

    Returns
    -------
    np.ndarray
        Array containing log-likelihood for every data point for the given theta
    """
    c0, alpha, beta, width_0, width_1, lag_0, lag_1, lag_2 = theta

    # Each date has a different c_mu, so this vector is of shape == c.shape
    c_mu: np.ndarray = (
        c0 + alpha * mavg(f, width_0, lag_0) + beta * mavg(f, width_1, lag_1) - lagged(c, lag_2)
    )

    return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))


def log_probability(theta: tuple, f: np.ndarray, c: np.ndarray) -> tuple[float, np.ndarray]:
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

    Returns
    -------
    (float, np.ndarray)
        Log posterior and pointwise log likelihood
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, np.full((len(f),), -np.inf)

    log_likelihood_vect = log_likelihood_vector(theta, f, c)
    log_likelihood = np.nansum(log_likelihood_vect)

    return lp + log_likelihood, log_likelihood_vect


def run_sampler():
    """Run the sampler."""
    data = get_data(impute_time=True)

    f = data["t"].to_numpy()
    c = data["c"].to_numpy()

    np.random.default_rng(42)
    initial = np.array(
        [
            10,  # c0
            5,  # alpha
            5,  # beta
            10,  # width_0
            10,  # width_1
            200,  # lag_0
            600,  # lag_1
            1000,  # lag_2
        ]
    )

    nwalkers = 32
    ndim = len(initial)
    pos = (
        np.vstack(
            (
                st.randint.rvs(low=-5, high=5, size=nwalkers),
                st.norm.rvs(loc=10, scale=1, size=nwalkers),
                st.norm.rvs(loc=10, scale=1, size=nwalkers),
                st.randint.rvs(low=-10, high=10, size=nwalkers),
                st.randint.rvs(low=-10, high=10, size=nwalkers),
                st.randint.rvs(low=-10, high=10, size=nwalkers),
                st.randint.rvs(low=-10, high=10, size=nwalkers),
                st.randint.rvs(low=-10, high=10, size=nwalkers),
            )
        ).T
        + initial
    )

    with Pool(processes=10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob_fn=log_probability, pool=pool, args=(f, c)
        )
        sampler.run_mcmc(pos, 20000, progress=True)

    idata = az.from_emcee(
        sampler=sampler,
        var_names=["c0", "alpha", "beta", "width_0", "width_1", "lag_0", "lag_1", "lag_2"],
        # var_names=['c0', 'alpha', 'beta', 'gamma', 'width_0', 'width_1', 'lag_0', 'lag_1', 'lag_2'],
        arg_names=["T", "c"],
        blob_names=["log_likelihood"],
    )

    samples = sampler.get_chain()
    np.save("posterior_samples_nogamma.npy", samples)

    idata.to_zarr("posterior_samples_nogamma.zarr")


def sample_posterior_predictive(
    samples: np.ndarray,
    data: np.ndarray,
    n_predictions: int | None = None,
    burn_in: int = 6000,
) -> np.ndarray:
    """Sample from the posterior predictive distribution.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples
    data : np.ndarray
        Observed data
    n_predictions : int | None
        Number of predictions to make
    burn_in : int | None
        Number of posterior samples to ignore from the beginning of the chains

    Returns
    -------
    np.ndarray
        Posterior predictive samples for the number of cones
    """
    f = data["t"].to_numpy()
    c = data["c"].to_numpy()

    samples = samples[burn_in:, :, :]
    _, _, ndim = samples.shape

    samples = samples.reshape((-1, ndim))

    if n_predictions is None:
        n_predictions = samples.shape[0]
    elif n_predictions > samples.shape[0]:
        raise ValueError(
            "Number of predictions is greater than the number of samples from the posterior distribution."
        )

    n_steps = data.shape[0]
    posterior_predictive = np.zeros((n_steps, n_predictions), dtype=float)

    for i in tqdm.trange(n_predictions, desc="Sampling the posterior predictive distribution..."):
        c0, alpha, beta, width_0, width_1, lag_0, lag_1, lag_2 = samples[i, :]
        # c0, alpha, beta, gamma, width_0, width_1, lag_0, lag_1, lag_2 = samples[i, :]

        for step in tqdm.trange(n_steps, desc=f"Sampling timesteps for sample {i}..."):
            slice0 = slice(step - lag_0 - width_0, step - lag_0 + width_0)
            slice1 = slice(step - lag_1 - width_1, step - lag_1 + width_1)

            if (
                (slice0.start < 0 or slice0.stop > n_steps)
                or (slice1.start < 0 or slice1.stop > n_steps)
                or (step - lag_2 < 0)
            ):
                posterior_predictive[step, i] = np.nan
                continue

            mu = c0 + alpha * f[slice0].mean() + beta * f[slice1].mean() - c[step - lag_2]
            # mu = c0 + alpha*f[slice0].mean() + beta*f[slice1].mean() - gamma*c[step - lag_2]

            if np.isnan(mu):
                posterior_predictive[step, i] = np.nan
                continue

            posterior_predictive[step, i] = st.poisson.rvs(mu, size=1)

    np.save("posterior_predictive_samples.npy", posterior_predictive)


def plot_chains(
    chains: str | np.ndarray = "posterior_samples_nogamma.npy",
    fig: plt.Figure | None = None,
    burn_in: int = 6000,
) -> plt.Figure:
    """Plot the MCMC chains.

    Parameters
    ----------
    chains : str | np.ndarray
        Sample chains
    fig : plt.Figure | None
        Figure in which to plot; if None, a new figure is returned
    burn_in :  int
        Number of steps to ignore from the front of the dataset

    Returns
    -------
    plt.Figure
        The plot of the chains for each dimension
    """
    if fig is None:
        fig = plt.figure()

    if isinstance(chains, str):
        chains = np.load(chains)

    tarmac.walker_trace(
        fig,
        chains[burn_in:, :, :],
        labels=[
            "c0",
            "alpha",
            "beta",
            # "gamma",
            "width_0",
            "width_1",
            "lag_0",
            "lag_1",
            "lag_2",
        ],
    )
    return fig


def plot_corner(
    chains: str | np.ndarray = "posterior_samples_nogamma.npy",
    fig: plt.Figure | None = None,
    burn_in: int = 6000,
) -> plt.Figure:
    """Generate a corner plot.

    Parameters
    ----------
    chains : str | np.ndarray
        MCMC sample chains
    fig : plt.Figure | None
        Figure in which to plot; if None, a new figure is generated
    burn_in : int
        Number of samples to ignore from the front of the dataset

    Returns
    -------
    plt.Figure
        The corner plot
    """
    if fig is None:
        fig = plt.figure()

    if isinstance(chains, str):
        chains = np.load(chains)

    tarmac.corner_plot(
        fig,
        chains[burn_in:, :, :],
        labels=[
            "c0",
            "alpha",
            "beta",
            # "gamma",
            "width_0",
            "width_1",
            "lag_0",
            "lag_1",
            "lag_2",
        ],
    )
    return fig


if __name__ == "__main__":
    run_sampler()
    plot_chains(burn_in=0)
    plot_corner(burn_in=16000)
    plt.show()
