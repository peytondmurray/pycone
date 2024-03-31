import datetime

import arviz as az
import numpy as np
import pymc as pm

import pycone

# Model 1: Delta-T model. Harder to model this because ΔT(t - τ_0) lags behind time t continuously,
# but n(t - τ_1) really doesn't; it's the time since the last crop. So the data preprocessing is
# harder to think about.

# Model 2: Continuous times
#\ n(t) = ɑT(t - τ_0) + βT(t - τ_2) - ɣn(t - τ_3)

# Model 2: Discrete times
#\ n = ɑT_i + βT_j - ɣn_k
#    = ɑT_k-2 + βT_k-1 - ɣn_k       (for pine species)

# Convert year+ordinal day of year to just day since the start of the dataset
mean_t = pycone.util.read_data("mean_t.csv")
cones = pycone.util.read_data('cones.csv')

# Note that we combine years differently than in analysis.py here (we are not using a crop year).
observed = pycone.util.add_days_since_start(
    mean_t.merge(cones, on="year")
)

site = 1
years = sorted(site_data['year'].unique())

site_data = observed.loc[observed['site'] == site]
year_i = site_data.loc[site_data['year'].isin(years[:-2])]
year_j = site_data.loc[site_data['year'].isin(years[1:-1])]
year_k = site_data.loc[site_data['year'].isin(years[2:])]



with pm.Model() as model:
    # Define priors for the model
    # σ_n: Statistical variation in the number of cones
    # α: Coefficient for the T_i-1 term, the differentiation year
    # β: Coefficient for the T_i-2 term, the differentiate
    # ɣ: Coefficient of the term corresponding to the previous cone crop. Must exist in range
    # [0, 1], otherwise the term doesn't make sense in terms of energy conservation.
    alpha = pm.Uniform("alpha", lower=-1000, upper=1000)
    beta = pm.Uniform("beta", lower=0, upper=1)
    sigma_n = pm.HalfCauchy("sigma_n", beta=10)
    sigma_t = pm.HalfCauchy("sigma_t", beta=100)
    t_avg = pm.Normal("t_avg", mu=50, sigma=30)

    likelihood = pm.Normal("n", mu=alpha*observed[])
