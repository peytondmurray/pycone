import pathlib

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from rich.console import Console

from .util import add_days_since_start, df_to_rich, read_data

plt.style.use("dark_background")
console = Console()

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


def get_data(
    weather_path: str = "weather.csv", cones_path: str = "cones.csv"
) -> pd.DataFrame:
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
        # year). This is because we are letting τ_0, τ_1, τ_2, τ_3 vary as (discrete) parameters.
        observed = weather.merge(cones, on="year")
        observed = add_days_since_start(observed, doy_col="day_of_year")[
            ["year", "day_of_year", "days_since_start", "t", "cones"]
        ]
        observed.to_csv("observed.csv", index=False)

    return observed


def main():
    # Likelihood is going to be poisson-distributed for each site; there's a waiting time
    # distribution for each cone appearing in the stand. There are few enough cones produced for
    # some species that summing them together (for the stand) will not produce a normal
    # distribution.
    data = get_data()

    coords = {}
    with pm.Model(coords=coords) as model:
        dss_data = pm.ConstantData("days_since_start_data", data["days_since_start"])
        t_data = pm.ConstantData("t_data", data["t"])
        cones_data = pm.ConstantData("cones_data", data["cones"])

        cones_mu = pm.DiscreteUniform("cones_0", lower=0, upper=20)
        cones = pm.Poisson("cones", mu=cones_mu, observed=cones_data)

        idata = pm.sample(discard_tuned_samples=False)
        prior_samples = pm.sample_prior_predictive(1000)

    #
    model.to_graphviz(save="model.pdf")
    console.print(df_to_rich(az.summary(idata)))

    az.plot_trace(idata)
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    az.plot_dist(
        data["cones"],
        kind="hist",
        color="C1",
        hist_kwargs={"alpha": 0.6},
        label="observed",
        ax=ax,
    )
    az.plot_dist(
        prior_samples.prior_predictive["cones"],
        kind="hist",
        hist_kwargs={"alpha": 0.6},
        label="simulated",
        ax=ax,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    az.plot_ts(
        idata=idata,
        y="cones",
        x="days_since_start_data",
        axes=ax,
    )

    plt.show()


if __name__ == "__main__":
    main()
