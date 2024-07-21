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
    if pathlib.Path("observed.csv").exists():
        console.log(f"[bold yellow]Loading existing data at {weather_path}")
        observed = read_data("observed.csv")
    else:
        # Convert year+ordinal day of year to just day since the start of the dataset
        site = 1
        weather = read_data(weather_path).rename(columns={"tmean (degrees f)": "t"})
        cones = read_data(cones_path)

        weather = weather.loc[weather["site"] == site]
        cones = cones.loc[cones["site"] == site]

        # Note that we combine years differently than in analysis.py here (we are not using a crop
        # year). The crop year itself is a parameter of the model below.
        observed = weather.merge(cones, on="year")
        observed = add_days_since_start(observed, doy_col="day_of_year")[
            ["year", "day_of_year", "days_since_start", "t", "cones"]
        ]
        observed.to_csv("observed.csv", index=False)

    obs = observed.rename(columns={"cones": "c"})

    if impute_time:
        return pd.DataFrame(
            {"days_since_start": np.arange(0, obs["days_since_start"].max() + 1)}
        ).merge(obs, on=["days_since_start"])

    return obs


def moving_average(
    t: SharedVariable,
    half_width: pm.Discrete,
    lag: pm.Discrete,
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
    width = 2 * half_width + 1
    # Reshape into 4d arrays to make conv2d happy; then flatten post-convolution to original dims
    kernel_4d = (pm.math.ones((width,), dtype=float) / width).reshape((1, 1, 1, -1))
    temperature_4d = t.reshape((1, 1, 1, -1))
    result = pm.math.full_like(t, np.nan, dtype=float)

    result = pt.tensor.set_subtensor(
        result[half_width:-half_width],
        conv.conv2d(temperature_4d, kernel_4d, border_mode="valid").flatten(),
    )
    return result
    # return lagged(result, lag)


def lagged(data: SharedVariable, lag: pm.Discrete) -> pt.tensor.TensorLike:
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
    result = pm.math.full_like(data, -np.inf, dtype=float)
    return pt.tensor.set_subtensor(result[:-lag], data[lag:])


def mavg(
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


# def calc_mavg(data, half_width):
#     width = 2 * half_width + 1
#     # Reshape into 4d arrays to make conv2d happy; then flatten post-convolution to original dims
#     kernel_4d = (pm.math.ones((width,), dtype=float) / width).reshape((1, 1, 1, -1))
#     temperature_4d = data.reshape((1, 1, 1, -1))
#     result = pm.math.full_like(data, np.nan, dtype=float)
#
#     result = pt.tensor.set_subtensor(
#         result[half_width:-half_width],
#         conv.conv2d(temperature_4d, kernel_4d, border_mode="valid").flatten(),
#     )
#     return result


def make_coeffs(data: np.ndarray, width: pm.Discrete, lag: pm.Discrete) -> pt.tensor.TensorLike:
    coeffs = pm.math.zeros(
        shape=data.size,
        dtype=float,
    )
    return pt.tensor.set_subtensor(
        coeffs[lag : lag + width + 1], pm.math.full(shape=(lag + width,), fill_value=1 / width)
    )


def run_ar():
    data = get_data(impute_time=True)

    with pm.Model() as model:
        n_values = len(data)
        days_since_start_data = pm.Data("days_since_start_data", data["days_since_start"])
        t_data = pm.Data("t_data", data["t"])
        c_data = pm.Data("c_data", data["c"])

        c0 = pm.DiscreteUniform("c0", lower=0, upper=1000)
        alpha = pm.HalfNormal("alpha", sigma=10)
        # beta = pm.HalfNormal("beta", sigma=10)
        # gamma = pm.HalfNormal("gamma", sigma=10)
        width_0 = pm.DiscreteUniform("width_0", 1, 100)
        # width_1 = pm.DiscreteUniform("width_1", 1, 100)
        lag_0 = pm.DiscreteUniform("lag_0", lower=180, upper=545)
        # lag_1 = pm.DiscreteUniform("lag_1", lower=550, upper=910)
        # lag_2 = pm.DiscreteUniform("lag_2", lower=915, upper=1275)

        rho = make_coeffs(t_data, width_0, lag_0)
        # sigma = make_coeffs(t_data, width_1, lag_1)
        # eta = make_coeffs(c_data, 1, lag_2)

        ar_t_1 = pm.AR(
            "ar_t_1",
            rho=rho,
            sigma=0,
            ar_order=n_values,
            init_dist=pm.Normal.dist(60, 20),
            shape=(n_values,),
        )
        # ar_t_2 = pm.AR("ar_t_2", rho=sigma, sigma=0, ar_order=n_values, init_dist=pm.Normal.dist(60, 20), shape=(n_values,))
        # ar_c = pm.AR("ar_c", rho=eta, sigma=0, ar_order=n_values, init_dist=pm.Normal.dist(10, 1), shape=(n_values,))

        c_mu = pm.Deterministic(
            "c_mu",
            c0 + alpha * ar_t_1,  # + beta*ar_t_2 + gamma*ar_c
        )
        c_model = pm.Poisson("c_model", mu=c_mu, observed=c_data)

        render_to_terminal(model)
        idata = pm.sample()
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


def lag(data: pd.Series, lag: pm.Discrete):
    lagged_data = pm.math.full((data.shape[0],), np.nan, dtype=float)
    lagged_data = pt.tensor.set_subtensor(lagged_data[:-lag], data[lag:])
    return lagged_data


def mavg2(data: pd.Series, width: pm.Discrete, lag: pm.Discrete):
    result, _updates = pt.scan(
        fn=fmavg,
        outputs_info=None,
        sequences=[pt.tensor.arange(data.shape[0])],
        non_sequences=[data, width, lag],
    )
    return result


def fmavg(i: int, data: pd.Series, width: pm.Discrete, lag: pm.Discrete):
    start = i - lag
    return pm.math.switch(
        pm.math.and_(pm.math.ge(start, 0), pm.math.lt(start + width, data.shape[0])),
        data[start : start + width].mean(),
        np.nan,
    )


def runmodel():
    data = get_data(impute_time=True)

    # fig, ax = plt.subplots()
    # ax.plot(data['days_since_start'], data['t'], '-ow', alpha=0.4)
    # ax.plot(data['days_since_start'], data['c'], '-or')
    # plt.show()

    with pm.Model() as model:
        n_values = len(data)

        c_vals = data["c"].values
        f_vals = data["t"].values
        day_vals = data["days_since_start"].values

        days_since_start_data = pm.Data("days_since_start_data", day_vals)
        f_data = pm.Data("f_data", f_vals)
        c_data = pm.Data("c_data", c_vals)

        c0 = pm.DiscreteUniform("c0", lower=0, upper=1000)
        alpha = pm.HalfNormal("alpha", sigma=10)
        beta = pm.HalfNormal("beta", sigma=10)
        gamma = pm.HalfNormal("gamma", sigma=10)
        width_0 = pm.DiscreteUniform("width_0", 1, 100)
        width_1 = pm.DiscreteUniform("width_1", 1, 100)
        lag_0 = pm.DiscreteUniform("lag_0", lower=180, upper=545)
        lag_1 = pm.DiscreteUniform("lag_1", lower=550, upper=910)
        lag_2 = pm.DiscreteUniform("lag_2", lower=915, upper=1275)

        c_mu = pm.Deterministic(
            "c_mu",
            c0
            + alpha * mavg2(f_data, width_0, lag_0)
            + beta * mavg2(f_data, width_1, lag_1)
            + gamma * lag(c_data, lag_2),
        )

        c_model = pm.Poisson("c_model", mu=c_mu, observed=c_data)

        render_to_terminal(model)
        idata = pm.sample()
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


if __name__ == "__main__":
    runmodel()
