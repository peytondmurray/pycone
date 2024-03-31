import datetime

import arviz as az
import numpy as np
import pymc as pm

import pycone

# Model 1:
# n(t) = ɑΔT(t - τ_0) - βn(t - τ_1)

# Model 2:
# n(t) = ɑΔT(t - τ_0) + βΔT(t - τ_1) - ɣn(t - τ_2)


observed = pycone.util.to_day_since_start(pycone.util.read_data("mean_t.csv"))

# Convert year+ordinal day of year to just day since the start of the dataset


with pm.Model() as model:
    # Define priors for the model
    # σ_n: Statistical variation in the number of cones
    # α: Coefficient for the ΔT term; not sure about this, as ΔT can be negative.
    # Probably we should reconsider this and instead use a "total solar flux" term
    # with a coefficient that can only be positive.
    # β: Coefficient of the term corresponding to the previous cone crop. Must exist in range
    # [0, 1], otherwise the term doesn't make sense in terms of energy conservation.
    alpha = pm.Uniform("alpha", lower=-1000, upper=1000)
    beta = pm.Uniform("beta", lower=0, upper=1)
    sigma_n = pm.HalfCauchy("sigma_n", beta=10)
    sigma_delta_t = pm.HalfCauchy("sigma_delta_t", beta=100)
    delta_t_bar = pm.Normal("delta_t_bar", mu=50, sigma=30)

    likelihood = pm.Normal("n", mu=alpha)
