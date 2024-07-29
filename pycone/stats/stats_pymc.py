import pathlib
import shutil
import subprocess

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.tensor import conv
from rich.console import Console

from ..preprocess import load_data
from ..util import add_days_since_start, df_to_rich, read_data

plt.style.use("dark_background")
console = Console()


def render_to_terminal(model: pm.Model):
    """Render the model to the terminal.

    The model is rendered as an svg, then displayed using the icat kitten. If kitty isn't installed,
    a log message is displayed instead and the image is not shown.

    Parameters
    ----------
    model : pm.Model
        PyMC model to display
    """
    if shutil.which("kitten") is None:
        console.log("[red]kitty is not installed; cannot display model graph in terminal.")
        return
    subprocess.run(
        ["kitten", "icat", "--align", "left"],
        input=model.to_graphviz().pipe(format="svg"),
    )


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
    int_width = pt.tensor.cast(width * 365, "int")
    int_lag = pt.tensor.cast(lag * 365, "int")

    window = 2 * width + 1
    average = np.convolve(f, pm.math.ones(shape=(window,), dtype=float), mode="same") / window

    # Mask off the convolution at the edge
    average[:int_width] = np.nan
    average[-int_width:] = np.nan

    result = np.full_like(f, np.nan, dtype=float)
    result[:-int_lag] = average[int_lag:]
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
    int_lag = pt.tensor.cast(lag * 365, "int")
    # lag = int(lag*365)

    result = np.full_like(c, np.nan, dtype=float)
    result[:-int_lag] = c[int_lag:]
    return result


def run_simple_model():
    # Likelihood is going to be poisson-distributed for each site; there's a waiting time
    # distribution for each cone appearing in the stand. There are few enough cones produced for
    # some species that summing them together (for the stand) will not produce a normal
    # distribution.
    data = get_data(impute_time=False)

    with pm.Model() as model:
        days_since_start_data = pm.Data("days_since_start_data", data["days_since_start"])
        t_data = pm.Data("t_data", data["t"])
        c_data = pm.Data("c_data", data["c"])

        # Uninformative uniform prior for the initial number c.
        c0 = pm.DiscreteUniform("c0", lower=0, upper=1000)

        alpha = pm.HalfNormal("alpha", sigma=10)
        # beta = pm.HalfNormal("beta", sigma=10)
        # gamma = pm.HalfNormal("gamma", sigma=10)
        half_width_0 = pm.DiscreteUniform("width_0", 1, 100)
        # half_width_1 = pm.DiscreteUniform("width_1", 1, 100)
        lag_0 = pm.DiscreteUniform("lag_0", lower=180, upper=545)
        # lag_1 = pm.DiscreteUniform("lag_1", lower=550, upper=910)
        # lag_2 = pm.DiscreteUniform("lag_2", lower=915, upper=1275)

        avg_t0 = pm.Deterministic("avg_t0", mavg(t_data, half_width_0, lag_0))
        # avg_t0 = pm.Deterministic("avg_t0", moving_average(t_data, half_width_0, lag_0))
        # avg_t1 = pm.Deterministic("avg_t1", moving_average(t_data, half_width_1, lag_1))
        # lagged_c = pm.Deterministic("lagged_c", lagged(c_data, lag_2))

        avg_t0_masked = pm.Deterministic(
            "avg_t0_masked",
            pm.math.switch(
                pt.tensor.isnan(avg_t0),
                pm.math.zeros_like(avg_t0),
                avg_t0,
            ),
        )

        c_mu = pm.Deterministic("c_mu", c0 + alpha * avg_t0_masked)

        c_model = pm.Poisson("c_model", mu=c_mu, observed=c_data)

        render_to_terminal(model)
        idata = pm.sample(discard_tuned_samples=False)
        # prior_samples = pm.sample_prior_predictive(1000)

    console.print(df_to_rich(az.summary(idata)))
    # render_to_terminal(model)

    az.plot_trace(idata)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    az.plot_dist(
        data["c"],
        kind="hist",
        color="C1",
        hist_kwargs={"alpha": 0.6},
        label="observed",
        ax=ax,
    )
    # az.plot_dist(
    #     prior_samples.prior_predictive["cones"],
    #     kind="hist",
    #     hist_kwargs={"alpha": 0.6},
    #     label="simulated",
    #     ax=ax,
    # )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    az.plot_ts(
        idata=idata,
        y="c_model",
        x="days_since_start_data",
        axes=ax,
    )

    plt.show()


def runmodel():
    data = get_data(impute_time=True)

    with pm.Model() as model:
        n_values = len(data)

        c_vals = data["c"].values
        f_vals = data["t"].values

        f_data = pm.MutableData("f_data", f_vals)
        c_data = pm.MutableData("c_data", c_vals)

        # Priors
        c0 = pm.Uniform("c0", lower=0, upper=1000)
        alpha = pm.HalfNormal("alpha", sigma=10)
        beta = pm.HalfNormal("beta", sigma=10)
        gamma = pm.HalfNormal("gamma", sigma=10)
        width_alpha = pm.Uniform("width_alpha", lower=0.002, upper=0.27)
        width_beta = pm.Uniform("width_beta", lower=0.002, upper=0.27)
        width_gamma = pm.Uniform("width_gamma", lower=0.002, upper=0.27)
        lag_alpha = pm.Uniform("lag_alpha", lower=0.5, upper=1.5)
        lag_beta = pm.Uniform("lag_beta", lower=1.5, upper=2.5)
        lag_gamma = pm.Uniform("lag_gamma", lower=2.5, upper=3.5)
        lag_last_cone = pm.Uniform("lag_last_cone", lower=2.5, upper=3.5)

        c_mu = pm.Deterministic(
            "c_mu",
            c0
            + alpha * mavg(f_data, width_alpha, lag_alpha)
            + beta * mavg(f_data, width_beta, lag_beta)
            + gamma * mavg(f_data, width_gamma, lag_gamma)
            - lagged(c_data, lag_last_cone),
        )

        c_model = pm.Poisson("c_model", mu=c_mu, observed=c_data)

        # render_to_terminal(model)
        idata = pm.sample()
        # prior_samples = pm.sample_prior_predictive(1000)

    console.print(df_to_rich(az.summary(idata)))

    az.plot_trace(idata)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    az.plot_dist(
        data["c"],
        kind="hist",
        color="C1",
        hist_kwargs={"alpha": 0.6},
        label="observed",
        ax=ax,
    )
    # az.plot_dist(
    #     prior_samples.prior_predictive["cones"],
    #     kind="hist",
    #     hist_kwargs={"alpha": 0.6},
    #     label="simulated",
    #     ax=ax,
    # )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    az.plot_ts(
        idata=idata,
        y="c_model",
        x="days_since_start_data",
        axes=ax,
    )

    plt.show()


if __name__ == "__main__":
    runmodel()
