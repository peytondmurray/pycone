import arviz as az
import numpy as np
import pymc as pm

import pycone


def main():
    # Model 1: Delta-T model. Harder to model this because ΔT(t - τ_0) lags behind time t continuously,
    # but n(t - τ_1) really doesn't; it's the time since the last crop. So the data preprocessing is
    # harder to think about.

    # Model 2: Discrete times
    # \ n = ɑT[i] + βT[j] - ɣn[k]
    #     = ɑT_k-2 + βT_k-1 - ɣn_k       (for pine species)

    # Model 3: Continuous times <-- This is probably the best. Allows the most freedom; should be
    # able to see pine species have different crop year gaps than other coniferous species
    # \ n(t) = ɑT(t - τ_0) + βT(t - τ_1) - ɣn(t - τ_2) + c
    # T ~ N(t_avg, sigma_t)

    # Convert year+ordinal day of year to just day since the start of the dataset
    site = 1
    mean_t = pycone.util.read_data("mean_t.csv")
    cones = pycone.util.read_data("cones.csv")

    mean_t = mean_t.loc[mean_t["site"] == site]
    cones = cones.loc[cones["site"] == site]

    # Note that we combine years differently than in analysis.py here (we are not using a crop year).
    # This is because we are letting τ_0, τ_1, τ_2, τ_3 vary as (discrete) parameters.
    observed = mean_t.merge(cones, on="year")

    mean_t_vals = observed["mean_t"].to_numpy()
    cone_vals = observed["cones"].to_numpy()

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
        # alpha = pm.Uniform("alpha", lower=-1000, upper=1000)
        # beta = pm.Uniform("beta", lower=-1000, upper=1000)
        # gamma = pm.Uniform("gamma", lower=0, upper=1)
        sigma_t = pm.HalfCauchy("sigma_t", beta=100)
        t_avg = pm.Normal("t_avg", mu=50, sigma=30)
        c = pm.Uniform("n0", lower=0, upper=1000)

        # Can use pm.math.stack to transform a list of floats/tensor_like to something pymc can use for
        # sampling. Here we just use normal distributions; but probably we should use noninformative
        # Uniform distributions here, as we do for alpha/beta above.
        rho_t = []
        rho_n = []
        for i in range(0, 1460):
            if i in [2 * 365, 3 * 365]:
                rho_t.append(pm.HalfNormal(f"rho_t_{i}", sigma=100))
            else:
                rho_t.append(0)

            if i == 4 * 365:
                rho_n.append(pm.Uniform("rho_n", lower=0, upper=1))
            else:
                rho_n.append(0)

        shape = len(rho_t) - 1
        rho_t = pm.math.stack(rho_t)
        rho_n = pm.math.stack(rho_n)

        # Mean number of cones due to temperature term
        ar_t = pm.AR(
            "ar_t",
            rho=rho_t,
            tau=0,
            constant=False,
            init_dist=pm.Uniform.dist(0, 1),
            shape=shape,
        )
        ar_n = pm.AR(
            "ar_n",
            rho=rho_n,
            tau=0,
            constant=False,
            init_dist=pm.HalfNormal.dist(100),
            shape=shape,
        )

        mean_n = ar_t + ar_n + c

        likelihood = pm.Poisson(
            "p_cones",
            mu=mean_n,
            observed=cone_vals,
        ) * pm.Normal(
            "t",
            mu=t_avg,
            sigma=sigma_t,
            observed=mean_t_vals,
        )

        idata = pm.sample()

    # model.to_graphviz(save="model.pdf")

    az.summary(idata)
    az.plot_trace(idata)


if __name__ == "__main__":
    main()
