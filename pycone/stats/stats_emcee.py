import functools
import pathlib
import sys
import warnings
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

from ..preprocess import load_data
from ..util import add_days_since_start, read_data

az.style.use("default")
console = Console()


if not sys.warnoptions:
    warnings.simplefilter("ignore", RuntimeWarning)


class Model:
    """Container for probability functions for a given model."""

    name: str = ""
    labels: list[str] = []

    def __init__(self):
        """Create a Model."""
        self.ndim = len(self.labels)
        self.data = None

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        raise NotImplementedError

    def log_prior(self, theta: tuple[float, ...]) -> float:
        """Calculate the log prior of the model.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        float
            Log probability for the given model parameters
        """
        raise NotImplementedError

    def log_likelihood_vector(
        self,
        theta: tuple[float, ...],
        f: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """Calculate the log likelihood for each data point (f, c) given the model parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone count

        Returns
        -------
        np.ndarray
            Individual probabilities of observing the data points given the model
            parameters; the shape is the same as the shape of the input data `f` and `c`.
        """
        raise NotImplementedError

    def posterior_predictive(
        self,
        theta: tuple[float, ...],
        f: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """Generate a sample from the posterior predictive distribution.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone count

        Returns
        -------
        np.ndarray
            Array of data generated from the posterior predictive distribution; has the
            same shape as the input data `f` and `c`

        """
        raise NotImplementedError


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


def log_probability(
    theta: tuple, f: np.ndarray, c: np.ndarray, model: Model
) -> tuple[float, np.ndarray]:
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
        return -np.inf, np.full((len(f),), -np.inf)

    log_likelihood_vect = model.log_likelihood_vector(theta, f, c)
    log_likelihood = np.nansum(log_likelihood_vect)

    return lp + log_likelihood, log_likelihood_vect


def run_sampler(model: Model, nwalkers: int = 32, nsamples: int = 20000):
    """Run the sampler."""
    data = get_data(impute_time=True)

    model.data = data

    f = data["t"].to_numpy()
    c = data["c"].to_numpy()

    sampler_path = f"{model.name}_sampler.h5"
    backend = emcee.backends.HDFBackend(f"{model.name}_sampler.h5")

    if pathlib.Path(sampler_path).exists():
        console.log(
            f"Existing emcee sampler loaded from {model.name}_sampler.h5. "
            f"Existing samples: {backend.get_chain().shape[0]}"
        )
    else:
        console.log(f"No emcee sampler found at {model.name}_sampler.h5; starting new sampler.")
        backend.reset(nwalkers, model.ndim)

    np.random.default_rng(42)
    with Pool(processes=10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            model.ndim,
            log_prob_fn=functools.partial(log_probability, model=model),
            pool=pool,
            args=(f, c),
            backend=backend,
        )
        sampler.run_mcmc(model.initialize(nwalkers), nsamples, progress=True)


def load_az_idata(model: Model, chains: np.ndarray | str | None = None) -> az.InferenceData:
    """Load the existing emcee backend into an Arviz InferenceData object.

    Parameters
    ----------
    model : Model
        Model for which the MCMC samples were generated
    chains : str | None
        MCMC sampler backend path; if None, the default backend path is checked

    Returns
    -------
    az.InferenceData
        Inference data object
    """
    if chains is None:
        backend = emcee.backends.HDFBackend(f"{model.name}_sampler.h5")
    elif isinstance(chains, str):
        backend = emcee.backends.HDFBackend(chains)

    nsamples, nwalkers, ndim = backend.get_chain().shape

    return az.from_emcee(
        sampler=emcee.EnsembleSampler(nwalkers, ndim, lambda: None, backend=backend),
        var_names=model.labels,
        blob_names=["log_likelihood"],
        arg_names=["T", "c"],
        arg_groups=["observed_data", "observed_data"],
    )


def sample_posterior_predictive(
    data: pd.DataFrame,
    chains: np.ndarray | str | None = None,
    n_predictions: int | None = None,
    burn_in: int = 16000,
) -> np.ndarray:
    """Sample from the posterior predictive distribution.

    This uses entirely self-generated data - i.e. it initializes with real cone data,
    but all subsequent samples are built on top of other prior predictive samples.

    Parameters
    ----------
    data : pd.DataFrame
        Observed data; `t` is the temperature, `c` is the cone count. Must be for a single site
    chains : np.ndarray | str | None
        Posterior samples array of shape (nsamples, nwalkers, ndim); or a path to an emcee backend
        containing samples; if None, the default backend path is checked
    model_name : str
        Name of the model
    n_predictions : int | None
        Number of predictions to make
    burn_in : int | None
        Number of posterior samples to ignore from the beginning of the chains

    Returns
    -------
    np.ndarray
        Posterior predictive samples for the number of cones
    """
    if chains is None:
        samples = emcee.backends.HDFBackend(f"{model.name}_sampler.h5").get_chain()
    elif isinstance(chains, str):
        samples = emcee.backends.HDFBackend(chains).get_chain()
    else:
        samples = chains

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
    posterior_predictive = np.zeros((n_predictions, n_steps), dtype=int)
    for i in tqdm.trange(n_predictions, desc="Sampling the posterior predictive distribution..."):
        posterior_predictive[i, :] = model.posterior_predictive(samples[i, :], f, c)

    np.save(f"posterior_predictive_{model.name}.npy", posterior_predictive)
    return posterior_predictive


class ThreeYearsPreceedingModel(Model):
    """Model with terms from temperature contributions for three years before the cone crop."""

    name = "three_years_preceeding"
    labels = [
        "c0",
        "alpha",
        "beta",
        "gamma",
        "width_alpha",
        "width_beta",
        "width_gamma",
        "lag_alpha",
        "lag_beta",
        "lag_gamma",
        "lag_last_cone",
    ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        initial = np.array(
            [
                10,  # c0
                5,  # alpha
                5,  # beta
                5,  # gamma
                10,  # width_alpha
                10,  # width_beta
                10,  # width_gamma
                365,  # lag_alpha
                730,  # lag_beta
                1095,  # lag_gamma
                1000,  # lag_last_cone
            ]
        )

        return (
            np.vstack(
                (
                    st.randint.rvs(low=-5, high=5, size=nwalkers),  # c0
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),  # alpha
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),  # beta
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),  # gamma
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # width_alpha
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # width_beta
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # width_gamma
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # lag_alpha
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # lag_beta
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # lag_gamma
                    st.randint.rvs(low=-10, high=10, size=nwalkers),  # lag_last_cone
                )
            ).T
            + initial
        )

    def log_prior(self, theta: tuple[float, ...]) -> float:
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
        (
            c0,
            alpha,
            beta,
            gamma,
            width_alpha,
            width_beta,
            width_gamma,
            lag_alpha,
            lag_beta,
            lag_gamma,
            lag_last_cone,
        ) = theta

        priors = [
            st.randint.pmf(np.floor(c0), low=0, high=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.halfnorm.pdf(gamma, scale=10),
            st.randint.pmf(np.floor(width_alpha), low=1, high=100),
            st.randint.pmf(np.floor(width_beta), low=1, high=100),
            st.randint.pmf(np.floor(width_gamma), low=1, high=100),
            st.randint.pmf(np.floor(lag_alpha), low=185, high=545),
            st.randint.pmf(np.floor(lag_beta), low=550, high=910),
            st.randint.pmf(np.floor(lag_gamma), low=915, high=1275),
            st.randint.pmf(np.floor(lag_last_cone), low=915, high=1275),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(
        self, theta: tuple[float, ...], f: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
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
        (
            c0,
            alpha,
            beta,
            gamma,
            width_alpha,
            width_beta,
            width_gamma,
            lag_alpha,
            lag_beta,
            lag_gamma,
            lag_last_cone,
        ) = theta

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))


class ScaledThreeYearsPreceedingModel(Model):
    """A model with terms for 1, 2, and 3 years preceeding a cone crop."""

    name = "scaled_three_year"
    labels = [
        "c0",
        "alpha",
        "beta",
        "gamma",
        "width_alpha",
        "width_beta",
        "width_gamma",
        "lag_alpha",
        "lag_beta",
        "lag_gamma",
        "lag_last_cone",
    ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        initial = np.array(
            [
                10,  # c0
                5,  # alpha
                5,  # beta
                5,  # gamma
                0.1,  # width_alpha
                0.1,  # width_beta
                0.1,  # width_gamma
                1,  # lag_alpha
                2,  # lag_beta
                3,  # lag_gamma
                3,  # lag_last_cone
            ]
        )

        return (
            np.vstack(
                (
                    st.uniform.rvs(loc=-5, scale=5, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                )
            ).T
            + initial
        )

    def log_prior(self, theta: tuple[float, ...]) -> float:
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
        (
            c0,
            alpha,
            beta,
            gamma,
            width_alpha,
            width_beta,
            width_gamma,
            lag_alpha,
            lag_beta,
            lag_gamma,
            lag_last_cone,
        ) = theta

        priors = [
            st.uniform.pdf(c0, loc=0, scale=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.halfnorm.pdf(gamma, scale=10),
            st.uniform.pdf(width_alpha, loc=0, scale=1),
            st.uniform.pdf(width_beta, loc=0, scale=1),
            st.uniform.pdf(width_gamma, loc=0, scale=1),
            st.uniform.pdf(lag_alpha, loc=0.5, scale=1),
            st.uniform.pdf(lag_beta, loc=1.5, scale=1),
            st.uniform.pdf(lag_gamma, loc=2.5, scale=1),
            st.uniform.pdf(lag_last_cone, loc=2.5, scale=1),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(
        self, theta: tuple[float, ...], f: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
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
        (
            c0,
            alpha,
            beta,
            gamma,
            width_alpha,
            width_beta,
            width_gamma,
            lag_alpha,
            lag_beta,
            lag_gamma,
            lag_last_cone,
        ) = theta

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * self.mavg(f, width_alpha, lag_alpha)
            + beta * self.mavg(f, width_beta, lag_beta)
            + gamma * self.mavg(f, width_gamma, lag_gamma)
            - self.lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def mavg(self, f: np.ndarray, width: float | int, lag: float | int) -> np.ndarray:
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
        width = int(width * 100)
        lag = int(lag * 365)

        window = 2 * width + 1
        average = np.convolve(f, np.ones(shape=(window,), dtype=float), mode="same") / window

        # Mask off the convolution at the edge
        average[:width] = np.nan
        average[-width:] = np.nan

        result = np.full_like(f, np.nan, dtype=float)
        result[:-lag] = average[lag:]
        return result

    def lagged(self, c: np.ndarray, lag: int | float) -> np.ndarray:
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
        lag = int(lag * 365)

        result = np.full_like(c, np.nan, dtype=float)
        result[:-lag] = c[lag:]
        return result

    def posterior_predictive(
        self,
        theta: tuple[float, ...] | np.ndarray,
        f: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone number

        Returns
        -------
        np.ndarray
            Time series (same shape as `f` and `c`) of independent cone predictions
        """
        (
            c0,
            alpha,
            beta,
            gamma,
            width_alpha,
            width_beta,
            width_gamma,
            lag_alpha,
            lag_beta,
            lag_gamma,
            lag_last_cone,
        ) = theta

        c_mu: np.ndarray = (
            c0
            + alpha * self.mavg(f, width_alpha, lag_alpha)
            + beta * self.mavg(f, width_beta, lag_beta)
            + gamma * self.mavg(f, width_gamma, lag_gamma)
            - self.lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)


class ScaledTwoYearsPreceedingModel(Model):
    """A model with terms for 1 and 2 years preceeding a cone crop."""

    name = "scaled_two_year"
    labels = [
        "c0",
        "alpha",
        "beta",
        "width_alpha",
        "width_beta",
        "lag_alpha",
        "lag_beta",
        "lag_last_cone",
    ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        initial = np.array(
            [
                10,  # c0
                5,  # alpha
                5,  # beta
                0.1,  # width_alpha
                0.1,  # width_beta
                1,  # lag_alpha
                2,  # lag_beta
                3,  # lag_last_cone
            ]
        )

        return (
            np.vstack(
                (
                    st.uniform.rvs(loc=-5, scale=5, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                )
            ).T
            + initial
        )

    def log_prior(self, theta: tuple[float, ...]) -> float:
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
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        priors = [
            st.uniform.pdf(c0, loc=0, scale=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.uniform.pdf(width_alpha, loc=0, scale=1),
            st.uniform.pdf(width_beta, loc=0, scale=1),
            st.uniform.pdf(lag_alpha, loc=0.5, scale=1),
            st.uniform.pdf(lag_beta, loc=1.5, scale=1),
            st.uniform.pdf(lag_last_cone, loc=2.5, scale=1),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(
        self, theta: tuple[float, ...], f: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
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
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * self.mavg(f, width_alpha, lag_alpha)
            + beta * self.mavg(f, width_beta, lag_beta)
            - self.lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def mavg(self, f: np.ndarray, width: float | int, lag: float | int) -> np.ndarray:
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
        width = int(width * 100)
        lag = int(lag * 365)

        window = 2 * width + 1
        average = np.convolve(f, np.ones(shape=(window,), dtype=float), mode="same") / window

        # Mask off the convolution at the edge
        average[:width] = np.nan
        average[-width:] = np.nan

        result = np.full_like(f, np.nan, dtype=float)
        result[:-lag] = average[lag:]
        return result

    def lagged(self, c: np.ndarray, lag: int | float) -> np.ndarray:
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
        lag = int(lag * 365)

        result = np.full_like(c, np.nan, dtype=float)
        result[:-lag] = c[lag:]
        return result

    def posterior_predictive(
        self,
        theta: tuple[float, ...],
        f: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone number

        Returns
        -------
        np.ndarray
            Time series (same shape as `f` and `c`) of independent cone predictions
        """
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        c_mu: np.ndarray = (
            c0
            + alpha * self.mavg(f, width_alpha, lag_alpha)
            + beta * self.mavg(f, width_beta, lag_beta)
            - self.lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)


class TwoYearsPreceedingModel(Model):
    """A model with terms for 1 and 2 years preceeding a cone crop."""

    name = "no_gamma"
    labels = [
        "c0",
        "alpha",
        "beta",
        "width_alpha",
        "width_beta",
        "lag_alpha",
        "lag_beta",
        "lag_last_cone",
    ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        initial = np.array(
            [
                10,  # c0
                5,  # alpha
                5,  # beta
                10,  # width_alpha
                10,  # width_beta
                200,  # lag_alpha
                600,  # lag_beta
                1000,  # lag_last_cone
            ]
        )

        return (
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

    def log_prior(self, theta: tuple[float, ...]) -> float:
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
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        priors = [
            st.randint.pmf(np.floor(c0), low=0, high=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.randint.pmf(np.floor(width_alpha), low=1, high=100),
            st.randint.pmf(np.floor(width_beta), low=1, high=100),
            st.randint.pmf(np.floor(lag_alpha), low=180, high=545),
            st.randint.pmf(np.floor(lag_beta), low=550, high=910),
            st.randint.pmf(np.floor(lag_last_cone), low=915, high=1275),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(
        self, theta: tuple[float, ...], f: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
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
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def posterior_predictive(
        self,
        theta: tuple[float, ...],
        f: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone number

        Returns
        -------
        np.ndarray
            Time series (same shape as `f` and `c`) of independent cone predictions
        """
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            - lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)


def plot_chains(
    model: Model,
    chains: str | np.ndarray | None = None,
    fig: plt.Figure | None = None,
    burn_in: int = 6000,
) -> plt.Figure:
    """Plot the MCMC chains.

    Parameters
    ----------
    model : Model
        Model to plot
    chains : str | np.ndarray | None
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
    if chains is None:
        samples = emcee.backends.HDFBackend(f"{model.name}_sampler.h5").get_chain()
    elif isinstance(chains, str):
        samples = emcee.backends.HDFBackend(chains).get_chain()
    else:
        samples = chains

    if fig is None:
        fig = plt.figure()

    tarmac.walker_trace(fig, samples[burn_in:, :, :], labels=model.labels)
    return fig


def plot_corner(
    model: Model,
    chains: str | np.ndarray | None = None,
    fig: plt.Figure | None = None,
    burn_in: int = 6000,
) -> plt.Figure:
    """Generate a corner plot.

    Parameters
    ----------
    model : Model
        Model to plot
    chains : str | np.ndarray | None
        Posterior sample chains
    fig : plt.Figure | None
        Figure in which to plot; if None, a new figure is generated
    burn_in : int
        Number of samples to ignore from the front of the dataset

    Returns
    -------
    plt.Figure
        The corner plot
    """
    if chains is None:
        samples = emcee.backends.HDFBackend(f"{model.name}_sampler.h5").get_chain()
    elif isinstance(chains, str):
        samples = emcee.backends.HDFBackend(chains).get_chain()
    else:
        samples = chains

    if fig is None:
        fig = plt.figure()

    tarmac.corner_plot(fig, samples[burn_in:, :, :], labels=model.labels)
    return fig


def plot_posterior_predictive(
    model: Model,
    posterior_predictive: np.ndarray | str | None = None,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot the posterior predictive samples.

    Parameters
    ----------
    model : Model
        Model for which posterior predictive samples were generated
    posterior_predictive : np.ndarray | str | None
        Samples from the posterior predictive distribution
    fig : plt.Figure | None
        Figure in which the samples should be plotted

    Returns
    -------
    plt.Figure
        Figure containing plots of posterior predictive samples

    """
    if fig is None:
        fig = plt.figure()

    if posterior_predictive is None:
        posterior_predictive = f"posterior_predictive_{model.name}.npy"

    if isinstance(posterior_predictive, str):
        samples = np.load(posterior_predictive)
    else:
        samples = posterior_predictive

    npredictions, nsteps = samples.shape

    axes = fig.subplots(npredictions, 1)
    if npredictions == 1:
        axes = [axes]

    for i in range(npredictions):
        axes[i].plot(samples[i, :], "-r", label="predicted")
        axes[i].plot(model.data["c"], "-k", label="measured")

    axes[0].legend()
    return fig


def plot_figures(
    model: Model,
    chains: str | np.ndarray | None = None,
    burn_in: int = 16000,
):
    """Plot the walker trace and corner plot.

    Parameters
    ----------
    model : Model
        Model to display
    chains : str | np.ndarray | None
        Posterior sample chains
    burn_in : int
        Number of samples to ignore from the front of the dataset
    """
    if chains is None:
        chains = emcee.backends.HDFBackend(f"{model.name}_sampler.h5").get_chain()
    elif isinstance(chains, str):
        chains = emcee.backends.HDFBackend(chains).get_chain()

    plot_chains(model, burn_in=0)
    plot_corner(model, burn_in=burn_in)


if __name__ == "__main__":
    model = ScaledThreeYearsPreceedingModel()
    # run_sampler(model, nwalkers=32, nsamples=20000)
    plot_figures(model, burn_in=16000)
    plt.show()
