= Background

- $T_0$: Vegetative priming
- $T_1$: Floral initiation
- $T_2$: Pollination
- $T_3$: Seed maturation (_Pinus_ only)


For 2 year reproductive cycles:

+ T0 occurs in early April; buds become dormant by November.
+ T1 occurs in early April; intial stages of sexual cone primordium formation, with differentiation
occuring through the spring of T1. These developing buds are sensitive to temperatures during
development, and environmental conditions have an effect on their developmental path through May and
June of T1. In unfavorable conditions, buds will abort. By mid-July of T1, vegetative and
reproductive buds are differentiated; by October, they are morphologically identifiable. By
December, buds have gone dormant.
+ Warm spring temperatures in T2 break dormancy and trigger pollen development. Fertilization and
embryonic development occurs from April through September. In autumn of T2, mature cones open and
shed seeds until late November.

#pagebreak()

= Symbols

$
x(t)
y(t)
tau
hat(R)
beta
$

$
n(t) = alpha Delta T underbrace((t - tau_0), "\"crop year\ngap\"") - beta n underbrace((t - tau_1), "\"crop\ninterval\"")
$
$
y(t) &= x(t) && + beta y(t-tau) \
n(t) &= alpha Delta T(t - tau_0) && - beta n(t - tau_1)
$

$
frak(L)(f(t)) = tilde(f)(s) = integral_0^(infinity) f(t)e^(-s t) d t &
$

where $s = sigma + i omega$

where $s$ is complex


#pagebreak()

= Modeling

So far we've talked a lot about $Delta T$ as a quantity that is correlated with cone production, but
as you've seen in our data, the degree to which it actually is a good predictor of cone crop varies
with the intervals you choose to analyze; so making a principled choice about what intervals
to analyze (start, offset, and duration) is difficult. You might be able to make some arguments
about when vegetative priming is happening or when certain species are dormant, or when
differentiation occurs, or something like that but _a priori_ we don't know the intervals that are
relevant here. An effective approach requires that you already know what these intervals should be,
which is not very simple to do. And just choosing the intervals which yield the highest correlation
opens up a huge can of worms due to look elsewhere effects that are pretty hard to control.

A different issue that is also something to be concerned with is that $Delta T$ does not _explain_
the reproductive processes we want to study. So I've wanted for a while to push forward on a cone
production model that comes from first principles, and my progress so far is what I'll be telling
you about today.

== First principles modeling

Energy to produce cones comes from sunlight. It's hard to measure solar flux, but let's make the
crude assumption that the energy available from sunlight is some function of the temperature. Now
for the sake of connecting this to previous work that was done, and to our discussion of $Delta T$
previously, I'm writing down a model which includes $Delta T$, and for the moment I'm going
to push to the back of our minds questions we might have about doing that. I'll talk about the
validity of that choice later on when we discuss other models.

In a simple model that includes $Delta T$, let's model the energy available to produce some number
of cones in a stand as proportional to $Delta T$ at a time in the past #footnote[During the
differentiation, or possibly during vegetative priming or some other important time during the
reproductive cycle. The fact that it's hard to talk about the meaning of $Delta T$ _at a specific
time_ is probably an indication here that there's a more appropriate model], but that energy budget
is reduced by the amount of energy used to produce cones in the previous crop some time $tau_1$ ago:

$
n(t) = alpha Delta T(t - tau_0) - beta n(t - tau_1)
$

Recurrence (or difference) equations very similar to this have been studied before in the context of
signal processing. In that context, there's a close analogy called the comb filter:

#figure(
    image("images/feedback_comb_filter.png", width: 80%),
)

Here, the output signal $y(t)$ is composed of an input signal $x(t)$ plus a part of the output
signal from $tau$ time ago $y(t-tau)$:

$
y(t) = x(t) + beta y(t-tau)
$

These systems are usually studied not in the time domain but rather in the frequency domain, and
there's a good reason for that: the fact that you're adding a delayed version of a signal to itself
means that there will be constructive and destructive interference. So we should naturally expect
systems which behave this way to have interesting behavior in the frequency domain.

So applying the laplace transform we have:


$
tilde(y)(s) = tilde(x)(s) + beta e^(-s tau) tilde(y)(s)
$

Then taking the ratio of the transformed output to the transformed input we get the so-called
Transfer Function, which tells us the frequency response of the system:

$
H(s) = frac(tilde(y)(s), tilde(x)(s)) = frac(1, 1-beta e^(-s tau))
$

This is a complex function whose magnitude has peaks and valleys when the denominator is minimized
and maximized for different input frequencies $s$. If we plot the magnitude as a function of
frequency:

#figure(
    image("images/comb_filter_spectrum.png", width: 80%),
)

The frequency response depends on the value of $beta$; negative values show peaks at
$omega tau = (2n+1)pi$, while positive values of $beta$ results in peaks at $omega tau = 2 pi n$.

Let's return our cone crop model and apply the same mathematics used to analyze these circuits:

$
n(t) = alpha Delta T(t - tau_0) - beta n(t - tau_1)
$

Apply the transform:

$
tilde(n)(s) = alpha e^(-s tau_0) tilde(Delta T)(s) - beta e^(-s tau_1)n(s)
$

Calculate the transfer function:

$
H(s) = frac(alpha e^(-s tau_0), 1 + beta e^(-s tau_1))
$

where $beta <= 0 <= 1$. Let's look at the magnitude of the transfer function for $s = i omega$:

$
|H(i omega)| = frac(alpha, sqrt(1 + beta^2 + 2 beta cos(omega tau_1)))
$

Plotting the magnitude of the transfer function vs frequency, we get peaks very similar to what we
saw with the comb filter:

#figure(
    image("images/cone_filter_spectrum.png", width: 80%),
)

Importantly, $beta$ cannot take on negative values here, so the only peaks appear for
$omega tau_1 = (2n + 1)pi$.

#pagebreak()

== Numerical Analysis

=== General questions

- Why is Pearson's correlation coefficient insufficient?

In general, we don't expect there to be a simple linear relationship between $n$ and $T$.
- Even the simplest models such as the crude one introduced here aren't linear
- We know that masting events happen. These clearly are nonlinear processes because the yield does
  not scale linearly with the tempearture or with $Delta T$.

- Why aren't you just using method X?

This bayesian approach is just one method. You could do X. You could do whatever. This is just one
method. I think it has some nice features that make it useful, so that's why I chose it.


=== Model

==== Likelihood

$
n(t) = alpha Delta T(t - tau_0) - beta n(t - tau_1)
$

Write down a posterior distribution. According to Bayes:

$
cal(P)(theta | D) = frac(cal(L)(D | theta) P(theta), P(D))
$

Denominator is a difficult integral, but it works out to a normalization constant so we don't really
need to deal with it:

$
P(D) = integral_theta cal(L)(D | theta) P(theta) d theta
$

Instead, we focus on the likelihood $cal(L)$ and the priors $P$. The likelihood is just the
probability of observing our data given the model. The probability of observing an individual data
point $(n_i)$ given our model is:

$
P(n_i | overline(n), sigma_n) = frac(1, sqrt(2 pi sigma_n^2)) exp[-frac((overline(n) - n_i)^2, 2 sigma_n^2)]
$

So the probability of observing all ${n_i}$ is the product of these individual probabilities:

$
P({n_i} | overline(n), sigma_n) = product_i frac(1, sqrt(2 pi sigma_n^2)) exp[-frac((overline(n) - n_i)^2, 2 sigma_n^2)]
$

But we already know what the expected value of $overline(n)$ is - it's our model! Substitute this in
and doing some algebra:

$
P({n_i} | alpha, beta, sigma_n) = frac(1, sqrt(2 pi sigma_n^2)) exp[-frac(1, 2) sum_i frac((alpha Delta T_(i-tau_0) - beta n_(i-tau_1) - n_i)^2,  sigma_n^2)]
$

Where $alpha$, $beta$, and $sigma_n$ are the parameters of our model. If you include statistical
uncertanties in our $Delta T$ measurements, the likelihood is:

$
cal(L)({n_i, Delta T_(i-tau_0)} | alpha, beta, sigma_n, sigma_(Delta T)) =

frac(1, 2 pi sigma_n sigma_(Delta T)) exp[-frac(1, 2) sum_i frac((alpha Delta T_(i-tau_0) - beta
n_(i-tau_1))^2, sigma_n^2) + frac((overline(Delta T)_(i-tau_0) - Delta T_(i-tau_0))^2, sigma_(Delta T)^2)]
$

where $overline(Delta T)_(i-tau_0)$ is a fit parameter to be marginalized over for each measurement.

==== Priors

- $P(alpha)$: Noninformative prior $(-infinity, infinity)$
- $P(beta)$: Noninformative prior $(0, 1)$
- $P(sigma_n)$: Noninformative (Jeffrey's) prior $∝ 1/sigma_n$
- $P(sigma_(Delta T))$: Noninformative (Jeffrey's) prior $∝ 1/sigma_(Delta T)$
- $P(overline(Delta T))$: $cal(N)("historical data")$

=== How do we know if our analysis is working?

+ Convergence diagnostics
    - Look at sample traces. The MCMC sampler produces a Markov Chain that samples from the posterior
      distribution. If the chain has converged, i.e. the mean has stabilized, the variance remains
      constant - then we know we've converged.
    - There are no divergences: a chain doesn't wander off into regions of low probability.
    - Bayesian fraction of missing information (BFMI). The latest gradient-based samplers like what
      you find in modern probabilistic programming packages use a family of solvers called Hamiltonian
      Monte Carlo methods to sample from the posterior distribution. These methods use Hamilton's
      equations of motion to do that - in essence, they're simulating how particles move around in a
      potential well, and they're generating samples from the posterior distribution from the position
      of the particles at various timesteps. And it turns out you can look at the energy at each
      step relative to the overall energy, you can estimate how well a sampler has been exploring
      the posterior distribution.
    - Similarly you can look at a plot of the energy transitions at each step for your chains, and
      compare that to the overall energy after fitting.
    - You can also look at something called the Potential Scale Reduction Factor (R-hat), which is
      basically an analysis of variance of different traces. Essentially you compare the variation
      in an individual chain to the variation between chains. And if they're the same, you've
      probably got a good fit. If your chains are in different spots, the between variance is wider
      than the variance within.

+ Goodness of fit
    - The best way to do this is to compare the output of your model to the data that was used to
      fit the model. Remember - we’re writing down an actual PDF that we think represents the
      generative processes that we think produced the ΔT and n measurements. So if we draw a bunch
      of samples from that model, ourdata should look just like more samples from that model.
      Otherwise we can't really claim that we're modeling the process that generated our data.

      The way that you can do this is using something called Posterior Predictive Checks, which
      involves sampling from the Posterior Predictive Distribution. The Posterior Predictive
      Distribution gives the probability of obtaining some new data $D_"new"$ given existing data
      $D$. So recall that for a fixed value of our model $theta$, our data $D$ follows the
      likelihood distribution:

      $
      cal(L)(D|theta)
      $

      and our posterior distribution tells us the probability of our model given our data:

      $
      cal(P)(theta|D)
      $

      However, the true value of our model $theta$ is uncertain, so we should average over the
      possible values of $theta$ to get the distribution of $D_"new"$ _given_ the existing data $D$.


      $
      p(D_"new"|D) = integral cal(L)(D_"new"|theta) cal(P)(theta|D) d theta
      $

      Here, $D_"new"$ is hypothetical new data that would be expected, taking into account the
      posterior uncertainty in the model parameters. Here $cal(P)(theta|D)$ is the posterior
      distribution - the probability of the model given the data that we've calculated numerically,
      and $cal(L)(D_"new"|theta)$ is our likelihood - we're drawing values from our likelihood.

      So what this does is include the residual uncertainty in our parameters in addition to the
      stochastic sampling uncertainty that comes from drawing from a distribution.

      Modern probabilistic programming packages will give you this for free, so it can actually be
      really straightforward to carry out this kind of analysis.
