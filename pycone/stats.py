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

from .preprocess import load_data
from .util import add_days_since_start, df_to_rich, read_data

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


def mavg_conv(
    t: SharedVariable,
    half_width: pm.Continuous,
    lag: pm.Continuous,
) -> pt.tensor.TensorLike:
    """Compute the moving average temperature.

    Parameters
    ----------
    t : SharedVariable
        t data
    half_width : pm.Discrete
        Half-width of the moving average window
    lag : pm.Discrete
        Lag time for the moving average to be computed at

    Returns
    -------
    pt.tensor.TensorLike
        Lagged moving average temperature
    """
    half_width_int = pm.pytensorf.intX(half_width)
    width = 2 * half_width_int + 1
    # Reshape into 4d arrays to make conv2d happy; then flatten post-convolution to original dims
    kernel_4d = (pm.math.ones((width,), dtype=float) / width).reshape((1, 1, 1, -1))
    temperature_4d = t.reshape((1, 1, 1, -1))
    result = pm.math.full_like(t, np.nan, dtype=float)

    result = pt.tensor.set_subtensor(
        result[half_width_int:-half_width_int],
        conv.conv2d(temperature_4d, kernel_4d, border_mode="valid").flatten(),
    )
    return lagged(result, lag)


def lagged(data: SharedVariable, lag: pm.Continuous) -> pt.tensor.TensorLike:
    """Lag an array by some amount.

    Parameters
    ----------
    data : SharedVariable
        Cone counts
    lag : pm.Discrete
        Number of days to lag the dataset by

    Returns
    -------
    pt.tensor.TensorLike
        Lagged dataset
    """
    lagged_data = pm.math.full((data.shape[0],), np.nan, dtype=float)
    lag_int = pm.pytensorf.intX(lag)
    return pt.tensor.set_subtensor(lagged_data[:-lag_int], data[lag_int:])


def mavg_scan(
    t: SharedVariable,
    half_width: pm.Discrete,
    lag: pm.Discrete,
) -> pt.tensor.TensorLike:
    """Compute the (lagged) moving average on a dataset without imputed time.

    Parameters
    ----------
    t : SharedVariable
        t data; does not contain dates with nan-valued cones or temperature
    half_width : pm.Discrete
        Half-width of the moving average window
    lag : pm.Discrete
        Lag time for the moving average to be computed at

    Returns
    -------
    pt.tensor.TensorLike
        Lagged moving average temperature
    """
    width = 2 * half_width + 1

    # Output shape _must_ equal the shape of the temperature data `t`
    result, _updates = pt.scan(
        fn=forward_moving_average,
        outputs_info=None,
        sequences=[pt.tensor.arange(t.shape[0])],
        non_sequences=[t, width],
    )
    return result


def forward_moving_average(i: int, t: SharedVariable, width: pm.Discrete):
    return pm.math.switch(i + width < t.shape[0], t[i : i + width].mean(), np.nan)


def standardize_normal(series: pd.Series) -> pd.Series:
    """Standardize a normally distributed dataset.

    Parameters
    ----------
    series : pd.Series
        Dataset to standardize

    Returns
    -------
    pd.Series
        Standardized dataset
    """
    return (series - series.mean()) / series.std()


def runmodel():
    # Likelihood is going to be poisson-distributed for each site; there's a waiting time
    # distribution for each cone appearing in the stand. There are few enough cones produced for
    # some species that summing them together (for the stand) will not produce a normal
    # distribution.
    data = get_data(impute_time=False)

    with pm.Model() as model:
        d_data = pm.MutableData("days_since_start_data", data["days_since_start"])
        t_data = pm.MutableData("t_data", standardize_normal(data["t"]))
        c_data = pm.MutableData("c_data", data["c"])

        # Priors
        c0 = pm.Uniform("c0", lower=0, upper=1000)
        alpha = pm.HalfNormal("alpha", sigma=10)
        beta = pm.HalfNormal("beta", sigma=10)
        gamma = pm.HalfNormal("gamma", sigma=10)
        width_alpha = pm.Uniform("width_alpha", lower=1, upper=100)
        width_beta = pm.Uniform("width_beta", lower=1, upper=100)
        width_gamma = pm.Uniform("width_gamma", lower=1, upper=100)
        lag_alpha = pm.Uniform("lag_alpha", lower=180, upper=545)
        lag_beta = pm.Uniform("lag_beta", lower=550, upper=910)
        lag_gamma = pm.Uniform("lag_gamma", lower=915, upper=1275)
        lag_last_cone = pm.Uniform("lag_last_cone", lower=915, upper=1275)

        c_mu = pm.Deterministic(
            "c_mu",
            c0
            + alpha * pm.Deterministic("avg_t_alpha", mavg_conv(t_data, width_alpha, lag_alpha))
            + beta * pm.Deterministic("avg_t_beta", mavg_conv(t_data, width_beta, lag_beta))
            + gamma * pm.Deterministic("avg_t_gamma", mavg_conv(t_data, width_gamma, lag_gamma))
            - pm.Deterministic("last_cone", lagged(c_data, lag_last_cone)),
        )

        pm.Deterministic(
            "mask_prior",
            pm.math.switch(
                pm.math.or_(pt.tensor.math.isnan(c_mu), pt.tensor.math.isnan(c_data)),
                0,
                1,
            ),
        )

        # # simple poisson
        # pm.Poisson(
        #     "c_model",
        #     mu=c_mu,
        #     observed=c_data,
        # )

        pm.Poisson(
            "c_model",
            mu=pm.math.switch(
                pm.math.or_(pt.tensor.math.isnan(c_mu), pt.tensor.math.isnan(c_data)),
                0,
                c_mu,
            ),
            observed=c_data,
        )

        render_to_terminal(model)
        model.debug()
        return
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


def mavg_forward(data: pd.Series, width: pm.Discrete, lag: pm.Discrete):
    result, _updates = pt.scan(
        fn=_forward,
        outputs_info=None,
        sequences=[pt.tensor.arange(data.shape[0])],
        non_sequences=[data, width, lag],
    )
    return result


def _forward(i: int, data: pd.Series, width: pm.Discrete, lag: pm.Discrete):
    start = i - lag
    return pm.math.switch(
        pm.math.and_(pm.math.ge(start, 0), pm.math.lt(start + width, data.shape[0])),
        data[start : start + width].mean(),
        np.nan,
    )


if __name__ == "__main__":
    runmodel()
