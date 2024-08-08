import functools
import pathlib
import sys
import warnings
from multiprocessing import Pool

import arviz as az
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tarmac
import tqdm
from rich.console import Console

from ..preprocess import load_data
from ..util import add_days_since_start, read_data
from .model import Model, ThreeYearsPreceedingModel
from .transform import StandardizeNormal

az.style.use("default")
console = Console()


if not sys.warnoptions:
    warnings.simplefilter("ignore", RuntimeWarning)


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


def log_probability(theta: tuple, model: Model) -> tuple[float, np.ndarray]:
    """Calculate the log posterior.

    See https://python.arviz.org/en/stable/getting_started/ConversionGuideEmcee.html
    for why two values are returned here. In short it is to keep track of the log
    likelihood at each sample, allowing us to unlock sample stats.

    Parameters
    ----------
    theta : tuple
        Model parameter vector
    model : Model
        Model for which the log probability is to be calculated

    Returns
    -------
    (float, np.ndarray)
        Log posterior and pointwise log likelihood
    """
    lp = model.log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, np.full((len(model.raw_data),), -np.inf)

    log_likelihood_vect = model.log_likelihood_vector(theta)
    log_likelihood = np.nansum(log_likelihood_vect)

    return lp + log_likelihood, log_likelihood_vect


def run_sampler(model: Model, nwalkers: int = 32, nsamples: int = 20000):
    """Run the sampler."""
    sampler_path = f"{model.name}_sampler.h5"

    if pathlib.Path(sampler_path).exists():
        # Get the number of existing runs in the file
        with h5py.File(sampler_path, "r") as file:
            runs = list(file.keys())

        backend = emcee.backends.HDFBackend(
            f"{model.name}_sampler.h5",
            name=f"mcmc_{len(runs)}",
        )
        old_backend = emcee.backends.HDFBackend(
            f"{model.name}_sampler.h5",
            name=f"mcmc_{len(runs) - 1}",
        )
        console.log(
            f"Existing emcee sampler loaded from {model.name}_sampler.h5;"
            f"Existing samples: {old_backend.iteration}"
        )
    else:
        backend = emcee.backends.HDFBackend(
            f"{model.name}_sampler.h5",
            name="mcmc_0",
        )
        console.log(f"No emcee sampler found at {model.name}_sampler.h5; starting new sampler.")
        backend.reset(nwalkers, model.ndim)

    np.random.default_rng(42)
    with Pool(processes=10) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            model.ndim,
            log_prob_fn=functools.partial(log_probability, model=model),
            pool=pool,
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
    model: Model,
    chains: np.ndarray | str | None = None,
    n_predictions: int | None = None,
    burn_in: int = 16000,
) -> np.ndarray:
    """Sample from the posterior predictive distribution.

    This uses entirely self-generated data - i.e. it initializes with real cone data,
    but all subsequent samples are built on top of other prior predictive samples.

    Parameters
    ----------
    model : Model
        Model for which posterior predictive samples are to be generated
    chains : np.ndarray | str | None
        Posterior samples array of shape (nsamples, nwalkers, ndim); or a path to an emcee backend
        containing samples; if None, the default backend path is checked
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
        sampler_path = f"{model.name}_sampler.h5"
        # Get the number of existing runs in the file
        with h5py.File(sampler_path, "r") as file:
            runs = list(file.keys())

        sampler = emcee.backends.HDFBackend(
            f"{model.name}_sampler.h5",
            name=f"mcmc_{len(runs) - 1}",
        )

        samples = sampler.get_chain()
    elif isinstance(chains, str):
        samples = emcee.backends.HDFBackend(chains, name="mcmc_0").get_chain()
    else:
        samples = chains

    samples = samples[burn_in:, :, :]
    _, _, ndim = samples.shape

    samples = samples.reshape((-1, ndim))

    if n_predictions is None:
        n_predictions = samples.shape[0]
    elif n_predictions > samples.shape[0]:
        raise ValueError(
            "Number of predictions is greater than the number of samples from the posterior distribution."
        )

    n_steps = len(model.raw_data)

    # Allocate float array because some values are np.nan (near edge of measured datapoints)
    posterior_predictive = np.zeros((n_predictions, n_steps), dtype=float)
    for i in tqdm.trange(n_predictions, desc="Sampling the posterior predictive distribution..."):
        posterior_predictive[i, :] = model.predictive(samples[i, :])

    # `model.predictive` returns _transformed_ predictions. Invert the transformed
    # predictions here rather than inside `model.predictive` for efficiency.
    posterior_predictive = model.transforms["c"].inverse(posterior_predictive)

    np.save(f"posterior_predictive_{model.name}.npy", posterior_predictive)
    return posterior_predictive


def sample_prior_predictive(model: Model, n_predictions: int | None = None) -> np.ndarray:
    """Sample from the prior predictive distribution.

    Parameters
    ----------
    model : Model
        Model for which prior predictive samples are to be generated
    n_predictions : int | None
        Number of predictions to make

    Returns
    -------
    np.ndarray
        Prior predictive samples for the number of cones
    """
    n_steps = len(model.raw_data)

    prior_samples = np.zeros((n_predictions, model.ndim), dtype=float)
    for i in tqdm.trange(n_predictions, desc="Sampling the prior distribution..."):
        prior_samples[i, :] = model.sample_prior()

    # Allocate float array because some values are np.nan (near edge of measured datapoints)
    prior_predictive = np.zeros((n_predictions, n_steps), dtype=float)
    for i in tqdm.trange(n_predictions, desc="Sampling the prior predictive distribution..."):
        prior_predictive[i, :] = model.predictive(prior_samples[i, :])

    # `model.predictive` returns _transformed_ predictions. Invert the transformed
    # predictions here rather than inside `model.predictive` for efficiency.
    prior_predictive = model.transforms["c"].inverse(prior_predictive)

    np.save(f"prior_predictive_{model.name}.npy", prior_predictive)
    return prior_predictive


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

    tarmac.corner_plot(fig, samples[burn_in:, :, :], labels=model.labels, bins=40)
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

    axes = fig.subplots(npredictions, 1, sharex=True)
    if npredictions == 1:
        axes = [axes]

    for i in range(npredictions):
        axes[i].plot(samples[i, :], "-r", label="predicted")
        axes[i].plot(model.raw_data["c"], "-k", label="data")

    axes[0].legend()
    return fig


def plot_posterior_predictive_one_plot(
    model: Model,
    posterior_predictive: np.ndarray | str | None = None,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot the posterior predictive samples on one plot.

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
        Figure containing a plot of posterior predictive samples
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

    ax = fig.subplots(1, 1)

    for i in range(npredictions):
        if i == 0:
            ax.plot(samples[i, :], "-k", label="predicted", alpha=0.3)
        else:
            ax.plot(samples[i, :], "-k", alpha=0.3)
    ax.plot(model.raw_data["c"], "-r", label="data")

    return fig


def plot_posterior_predictive_density(
    model: Model,
    posterior_predictive: np.ndarray | str | None = None,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Plot the posterior predictive samples as a density.

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
        Figure containing a plot of posterior predictive samples
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

    ax = fig.subplots(1, 1)

    # Choose the bin size so that each cone number is one bin
    bins = int(np.nanmax(samples))
    c_range = np.array([np.nanmin(samples), bins])

    pdf = np.zeros((bins, nsteps), dtype=float)
    for i in tqdm.trange(nsteps, desc="Histogramming the posterior predictive distribution."):
        values, _ = np.histogram(samples[:, i], bins=bins, range=c_range, density=True)
        pdf[:, i] = values

    ax.imshow(
        pdf,
        extent=[0, nsteps, 0, bins],
        cmap="viridis",
        interpolation="none",
        origin="lower",
        aspect="auto",
    )

    ax.set_title("Posterior predictive distribution of cone count")
    ax.plot(model.raw_data["c"], "-r", label="observed")
    ax.legend()
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
        chains = emcee.backends.HDFBackend(f"{model.name}_sampler.h5", name="mcmc_0").get_chain()
    elif isinstance(chains, str):
        chains = emcee.backends.HDFBackend(chains).get_chain()

    plot_chains(model, chains, burn_in=0)
    plot_corner(model, chains, burn_in=burn_in)

    posterior_predictive_fname = f"posterior_predictive_{model.name}.npy"
    if pathlib.Path(posterior_predictive_fname).exists():
        posterior_predictive = np.load(posterior_predictive_fname)
        plot_posterior_predictive(model, posterior_predictive)


if __name__ == "__main__":
    model = ThreeYearsPreceedingModel(
        get_data(impute_time=True, site=1), transforms={"t": StandardizeNormal}
    )
    # run_sampler(model, nwalkers=32, nsamples=10000)
    sample_posterior_predictive(model, n_predictions=100000, burn_in=2000)
    # plot_figures(model, burn_in=2000)
    # plot_posterior_predictive_one_plot(model)
    plot_posterior_predictive_density(model)
    plt.show()
