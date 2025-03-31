import pathlib
import shutil
import subprocess

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
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
            c

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


def year_cumsum(data: pd.DataFrame) -> np.ndarray:
    """Take the yearly cumulative sum of the cone measurements.

    Cones are only measured once a year, but the values are copied to every measured day.
    This function takes the cone count for each year, cumsums it, then copies the cone
    cumsum to every day in the year. This column is then returned as the result.

    Parameters
    ----------
    data : pd.DataFrame
        Observed data containing "year" and "c" (cone count) columns

    Returns
    -------
    np.ndarray
        Cumulative cone crop count (summed by year), copied to each day of the year
    """
    first_c = data.groupby("year")["c"].first().cumsum()
    first_c.name = "c_year_cumsum"
    return data.merge(first_c, left_on="year", right_index=True)["c_year_cumsum"]


def runmodel():
    """Run the pymc model."""
    data = get_data(impute_time=True)

    data["c_year_cumsum"] = year_cumsum(data)

    with pm.Model() as model:
        t_obs = pm.Data("t_data", data["t"].to_numpy())
        c_obs = pm.Data("c_data", data["c"].to_numpy())
        c_cumsum_data = pm.Data("c_year_cumsum_data", data["c_year_cumsum"].to_numpy())

        # Priors
        c0 = pm.Uniform("c0", lower=0, upper=1000)
        alpha = pm.HalfNormal("alpha", sigma=10)
        n = pm.DiscreteUniform("n_buds", lower=0, upper=1000)
        k = pm.DiscreteUniform("n_cones", lower=0, upper=1000)

        # c_mu = c0 + alpha*cumsum(t) - year_cumsum(c)
        c_max = pm.Deterministic("c_mu", c0 + alpha * t_obs.cumsum() - c_cumsum_data)

        # p = pm.Deterministic("p", pm.math.invlogit(c_mu))

        # See https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/GLM-binomial-regression.html
        # pm.Binomial("c_model", n=n, p=p, observed=c_obs)
        pm.HyperGeometric("c_model", N=c_max, n=n, k=k, observed=c_obs)

        # model.debug(verbose=True)
        # render_to_terminal(model)
        print(model.initial_point())
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
