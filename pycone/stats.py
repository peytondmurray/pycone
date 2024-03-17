import pymc as pm

model = pm.Model()

with model:
    # Define priors for the model
    # σ_n: Statistical variation in the number of cones
    # α: Coefficient for the ΔT term; not sure about this, as ΔT can be negative.
    # Probably we should reconsider this and instead use a "total solar flux" term
    # with a coefficient that can only be positive.
    # β: Coefficient of the term corresponding to the previous cone crop. Must exist in range
    # [0, 1], otherwise the term doesn't make sense in terms of energy conservation.
    sigma_n = pm.HalfNormal("sigma_n", sigma=1)
    alpha = pm.Normal("alpha", sigma=10)
    beta = pm.Uniform(0, 1, value=0.5)
