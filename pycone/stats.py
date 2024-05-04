import datetime

import arviz as az
import numpy as np
import pymc as pm

import pycone

# Model 1: Delta-T model. Harder to model this because ΔT(t - τ_0) lags behind time t continuously,
# but n(t - τ_1) really doesn't; it's the time since the last crop. So the data preprocessing is
# harder to think about.

# Model 2: Discrete times
# \ n = ɑT_i + βT_j - ɣn_k
#    = ɑT_k-2 + βT_k-1 - ɣn_k       (for pine species)

# Model 3: Continuous times <-- This is probably the best. Allows the most freedom; should be
# able to see pine species have different crop year gaps than other coniferous species
# \ n(t) = ɑT(t - τ_0) + βT(t - τ_2) - ɣn(t - τ_3)

# Convert year+ordinal day of year to just day since the start of the dataset
mean_t = pycone.util.read_data("mean_t.csv")
cones = pycone.util.read_data("cones.csv")

# Note that we combine years differently than in analysis.py here (we are not using a crop year).
# This is because we are letting τ_0, τ_1, τ_2, τ_3 vary as (discrete) parameters.
observed = pycone.util.add_days_since_start(mean_t.merge(cones, on="year"))

site = 1
# years = sorted(mean_t["year"].unique())

site_data = observed.loc[observed["site"] == site]
# year_h = site_data.loc[site_data["year"].isin(years[:-3])]
# year_i = site_data.loc[site_data["year"].isin(years[1:-2])]
# year_j = site_data.loc[site_data["year"].isin(years[2:-1])]
# year_k = site_data.loc[site_data["year"].isin(years[3:])]


# Likelihood is going to be poisson-distributed for each site; there's a waiting time distribution
# for each cone appearing in the stand. There are few enough cones produced for some species that
# summing them together (for the stand) will not produce a normal distribution.

with pm.Model() as model:
    # Define priors for the model
    # α: Coefficient for the T_i-1 term, the pollination year
    # β: Coefficient for the T_i-2 term, the priming year
    # ɣ: Coefficient of the term corresponding to the previous cone crop. Must exist in range
    #    [0, 1], otherwise the term doesn't make sense in terms of energy conservation.
    # σ_n: Statistical variation in the number of cones. HalfCauchy, because that's what more
    #      experienced people are doing with this kind of parameter
    # σ_t: Statistical variation in the (mean daily) temperature
    alpha = pm.Uniform("alpha", lower=-1000, upper=1000)
    beta = pm.Uniform("beta", lower=-1000, upper=1000)
    gamma = pm.Uniform("gamma", lower=0, upper=1)
    sigma_n = pm.HalfCauchy("sigma_n", beta=10)
    sigma_t = pm.HalfCauchy("sigma_t", beta=100)
    t_avg = pm.Normal("t_avg", mu=50, sigma=30)
    tau_0 = pm.Uniform("tau_0", lower=0, upper=1460)
    tau_1 = pm.Uniform("tau_1", lower=0, upper=1460)
    tau_2 = pm.Uniform("tau_2", lower=0, upper=1460)
    tau_3 = pm.Uniform("tau_3", lower=0, upper=1460)

    likelihood = pm.Poisson(
        "p_cones",
        mu=alpha*site_data["mean_t"] + beta*year_j
    )

    # likelihood = pm.Normal(
    #     "p_cones",
    #     mu=alpha * year_i["mean_t"] + beta * year_j["mean_t"] - gamma * year_h["cones"],
    #     sigma=sigma_n,
    #     observed=year_k["cones"],
    # ) * pm.Normal("p_temp", mu=t_avg, sigma=sigma_t, observed=year_k["mean_t"])

model.to_graphviz()
