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
        console.log(
            "[red]kitty is not installed; cannot display model graph in terminal."
        )
        return
    subprocess.run(
        ["kitten", "icat", "--align", "left"],
        input=model.to_graphviz().pipe(format="svg"),
    )


def get_data(
    weather_path: str = "weather.csv", cones_path: str = "cones.csv"
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

    return (
        pd.DataFrame(
            {"days_since_start": np.arange(0, observed["days_since_start"].max() + 1)}
        )
        .merge(observed, on=["days_since_start"])
        .rename(columns={"cones": "c"})
    )


# Model 1: Delta-T model. Harder to model this because ΔT(t - τ_0) lags behind time t
# continuously, but n(t - τ_1) really doesn't; it's the time since the last crop. So the data
# preprocessing is harder to think about.

# Model 2: Discrete times
# \ n = ɑT[i] + βT[j] - ɣn[k]
#     = ɑT_k-2 + βT_k-1 - ɣn_k       (for pine species)

# Model 3: Continuous times <-- This is probably the best. Allows the most freedom; should be
# able to see pine species have different crop year gaps than other coniferous species
# \ n(t) = ɑT(t - τ_0) + βT(t - τ_1) - ɣn(t - τ_2) + c
# T ~ N(t_avg, sigma_t)
#
#
# Model is autoregressive:
# N(t) = N_0 + a*T_avg(t - tau_0) + b*T_avg(t - tau_1) - c*N(t - tau_2)
#
# where N(t) ~ Poisson(n_avg)
# and T_avg(t - tau) = avg(T(t - tau), duration=d)


def moving_average(
    t: SharedVariable,
    width: pm.Discrete,
    lag: pm.Discrete,
) -> pt.tensor.TensorLike:
    """Compute the moving average temperature.

    Parameters
    ----------
    t : SharedVariable
        t data
    width : pm.Discrete
        Width of the moving average window
    lag : pm.Discrete
        Lag time for the moving average to be computed at

    Returns
    -------
    pt.tensor.TensorLike
        Lagged moving average temperature
    """
    # Reshape into 4d arrays to make conv2d happy; then flatten post-convolution to original dims
    kernel_4d = (pm.math.ones((width,), dtype=float) / width).reshape((1, 1, 1, -1))
    temperature_4d = t.reshape((1, 1, 1, -1))
    result = pm.math.full_like(t, np.nan, dtype=float)

    pt.tensor.set_subtensor(
        result[width // 2 : -width // 2],
        conv.conv2d(temperature_4d, kernel_4d, border_mode="valid").flatten(),
    )
    return lagged(result, lag)


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
    result = pm.math.full_like(data, np.nan, dtype=float)
    pt.tensor.set_subtensor(result[:-lag], data[lag:])
    return result


def my_model(t_data, c_data, c0, a, b, g, w0, l0, w1, l1, l2):
    return (
        c0
        + a * moving_average(t_data, w0, l0)
        + b * moving_average(t_data, w1, l1)
        - g * lagged(c_data, l2)
    )


def main():
    # Likelihood is going to be poisson-distributed for each site; there's a waiting time
    # distribution for each cone appearing in the stand. There are few enough cones produced for
    # some species that summing them together (for the stand) will not produce a normal
    # distribution.
    data = get_data()

    with pm.Model() as model:
        t_data = pm.MutableData("t_data", data["t"])
        c_data = pm.MutableData("c_data", data["c"])

        # Uninformative uniform prior for the initial number c.
        c0 = pm.DiscreteUniform("c0", lower=0, upper=1000)

        alpha = pm.HalfNormal("alpha", sigma=10)
        beta = pm.HalfNormal("beta", sigma=10)
        gamma = pm.HalfNormal("gamma", sigma=10)
        width_0 = pm.DiscreteUniform("width_0", 1, 200)
        width_1 = pm.DiscreteUniform("width_1", 1, 200)
        lag_0 = pm.DiscreteUniform("lag_0", lower=180, upper=545)
        lag_1 = pm.DiscreteUniform("lag_1", lower=550, upper=910)
        lag_2 = pm.DiscreteUniform("lag_2", lower=915, upper=1275)

        avg_t0 = pm.Deterministic("avg_t0", moving_average(t_data, width_0, lag_0))
        avg_t1 = pm.Deterministic("avg_t1", moving_average(t_data, width_1, lag_1))
        lagged_c = pm.Deterministic("lagged_c", lagged(c_data, lag_2))

        c_mu = pm.Deterministic(
            "c_mu",
            c0 + alpha * avg_t0 + beta * avg_t1 - gamma * lagged_c,
        )

        c_likelihood = pm.Poisson("c_likelihood", mu=c_mu, observed=c_data)

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
        y="c",
        x="days_since_start_data",
        axes=ax,
    )

    plt.show()


if __name__ == "__main__":
    main()
