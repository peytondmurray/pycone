cimport numpy
import numpy as np
import scipy.special as ss
import scipy.stats as st

from .fast_stats_emcee cimport lagged, mavg

cdef class Model:
    """Container for probability functions for a given model."""

    cdef str _name
    cdef list[str] _labels

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def ndim(self):
        return len(self.labels)

    cpdef double[:, :] initialize(self, int nwalkers = 32):
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        raise NotImplementedError

    cpdef double log_prior(self, double[:] theta):
        """Calculate the log prior of the model.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        float
            Log probability for the given model parameters
        """
        raise NotImplementedError

    cpdef double[:] log_likelihood_vector(
        self,
        double[:] theta,
        double[:] f,
        long[:] c,
    ):
        """Calculate the log likelihood for each data point (f, c) given the model parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone count

        Returns
        -------
        np.ndarray
            Individual probabilities of observing the data points given the model
            parameters; the shape is the same as the shape of the input data `f` and `c`.
        """
        raise NotImplementedError

    cpdef posterior_predictive(
        self,
        double[:] theta,
        double[:] f,
        long[:] c,
    ):
        """Generate a sample from the posterior predictive distribution.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters
        f : np.ndarray
            Temperature
        c : np.ndarray
            Cone count

        Returns
        -------
        np.ndarray
            Array of data generated from the posterior predictive distribution; has the
            same shape as the input data `f` and `c`

        """
        raise NotImplementedError


cdef class ThreeYearsPreceedingModel(Model):
    """Model with terms from temperature contributions for three years before the cone crop."""

    cdef list priors

    def __init__(self):
        super().__init__()
        self.name = "three_years_preceeding"
        self.labels = [
            "c0",
            "alpha",
            "beta",
            "gamma",
            "width_alpha",
            "width_beta",
            "width_gamma",
            "lag_alpha",
            "lag_beta",
            "lag_gamma",
            "lag_last_cone",
        ]

        self.priors = [
            st.uniform(loc=0, scale=1000),
            st.halfnorm(scale=10),
            st.halfnorm(scale=10),
            st.halfnorm(scale=10),
            st.uniform(loc=1, scale=100),
            st.uniform(loc=1, scale=100),
            st.uniform(loc=1, scale=100),
            st.uniform(loc=185, scale=365),
            st.uniform(loc=550, scale=365),
            st.uniform(loc=915, scale=365),
            st.uniform(loc=915, scale=365),
        ]


    cpdef double[:, :] initialize(self, int nwalkers = 32):
        """Generate initial positions for the MCMC walkers.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        return np.vstack(
            (
                st.norm.rvs(loc=20, scale=5, size=nwalkers),  # c0
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # alpha
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # beta
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # gamma
                st.norm.rvs(loc=30, scale=5, size=nwalkers),  # width_alpha
                st.norm.rvs(loc=30, scale=5, size=nwalkers),  # width_beta
                st.norm.rvs(loc=30, scale=5, size=nwalkers),  # width_gamma
                st.norm.rvs(loc=365, scale=5, size=nwalkers),  # lag_alpha
                st.norm.rvs(loc=730, scale=5, size=nwalkers),  # lag_beta
                st.norm.rvs(loc=1095, scale=5, size=nwalkers),  # lag_gamma
                st.norm.rvs(loc=1095, scale=5, size=nwalkers),  # lag_last_cone
            )
        ).T

    cpdef double log_prior(self, double[:] theta):
        """Compute the log prior probability.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

        Returns
        -------
        float
            Log prior probability
        """
        cdef double[:] priors = np.array(
            [self.priors[i].pdf(parameter) for i, parameter in enumerate(theta)]
        )

        cdef double prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    cpdef double[:] log_likelihood_vector(
        self,
        double[:] theta,
        double[:] f,
        long[:] c,
    ):
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model
        f : np.ndarray
            Temperature data
        c : np.ndarray
            Cone data

        Returns
        -------
        np.ndarray
            Array containing log-likelihood for every data point for the given theta
        """
        (
            c0,
            alpha,
            beta,
            gamma,
            width_alpha,
            width_beta,
            width_gamma,
            lag_alpha,
            lag_beta,
            lag_gamma,
            lag_last_cone,
        ) = theta

        # Each date has a different c_mu, so this vector is of shape == c.shape
        cdef double[:] c_mu = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    cpdef double log_probability(self, double[:] theta, double[:] f, long[:] c):
        cdef double lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        cdef double log_likelihood = np.nansum(self.log_likelihood_vector(theta, f, c))

        return lp + log_likelihood
