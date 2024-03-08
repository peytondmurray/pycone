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
$

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
crude assumption that the temperature is proportional to the energy available from sunlight.

In a simple model that includes $Delta T$, the energy available to produce some number of cones in a
stand is proportional to $Delta T$ at a time in the past #footnote[During the differentiation, or possibly
during vegetative priming or some other important time during the reproductive cycle. The fact that
it's hard to talk about the meaning of $Delta T$ _at a specific time_ is probably an indication here
that there's a more appropriate model], but that energy budget is reduced by the amount of energy
used to produce cones in the previous crop some time $tau_1$ ago:

$
n(t) = alpha Delta T(t - tau_0) - beta n(t - tau_1)
$

Recurrence (or difference) equations very similar to this have been studied before in the context of
signal processing. In that context, there closest analogy is called a comb filter:

#figure(
    image("images/feedback_comb_filter.png", width: 80%),
)

Here, the output signal $y(t)$ is composed of an input signal $x(t)$ plus a part of the output
signal from $tau$ time ago $y(t-tau)$:

$
y(t) = x(t) + alpha y(t-tau)
$

These systems are usually studied not in the time domain but rather in the frequency domain, and
there's a good reason for that: the fact that you're adding a delayed version of a signal to itself
means that there will be constructive and destructive interference. So we should naturally expect
systems which behave this way to have interesting behavior in the frequency domain.

Let's return our cone crop model and apply the same mathematics used to analyze these circuits:
