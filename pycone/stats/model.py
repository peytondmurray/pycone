import numpy as np
import pandas as pd
import scipy.special as ss
import scipy.stats as st

from .math import lagged, mavg
from .transform import IdentityTransform


class Model:
    """Container for probability functions for a given model."""

    name: str = ""
    labels: list[str] = []

    def __init__(self, data: pd.DataFrame, transforms: dict[str, type]):
        """Create a Model."""
        self.ndim = len(self.labels)
        self.raw_data = data
        self.transforms = {}

        transformed_data = {}
        for col in data.columns:
            self.transforms[col] = transforms.get(col, IdentityTransform)()
            transformed_data[col] = self.transforms[col].transform(data[col])

        self.transformed_data = pd.DataFrame(transformed_data)

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
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

    def log_prior(self, theta: tuple[float, ...]) -> float:
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

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> np.ndarray:
        """Calculate the log likelihood for each data point given the model parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        np.ndarray
            Individual probabilities of observing the data points given the model
            parameters; has the same shape as the raw data `self.raw_data['c']`
        """
        raise NotImplementedError

    def posterior_predictive(self, theta: tuple[float, ...]) -> np.ndarray:
        """Generate a sample from the posterior predictive distribution.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        np.ndarray
            Array of data generated from the posterior predictive distribution; has the
            same shape as the raw data `self.raw_data['c']`

        """
        raise NotImplementedError


class ThreeYearsPreceedingModel(Model):
    """Model with terms from temperature contributions for three years before the cone crop."""

    name = "three_years_preceeding"
    labels = [
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

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
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
                st.norm.rvs(loc=30, scale=10, size=nwalkers),  # c0
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # alpha
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # beta
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # gamma
                st.norm.rvs(loc=30, scale=15, size=nwalkers),  # width_alpha
                st.norm.rvs(loc=30, scale=15, size=nwalkers),  # width_beta
                st.norm.rvs(loc=30, scale=15, size=nwalkers),  # width_gamma
                st.norm.rvs(loc=365, scale=15, size=nwalkers),  # lag_alpha
                st.norm.rvs(loc=730, scale=15, size=nwalkers),  # lag_beta
                st.norm.rvs(loc=1095, scale=15, size=nwalkers),  # lag_gamma
                st.norm.rvs(loc=1095, scale=15, size=nwalkers),  # lag_last_cone
            )
        ).T

    def log_prior(self, theta: tuple[float, ...]) -> float:
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

        priors = [
            st.uniform.pdf(c0, loc=0, scale=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.halfnorm.pdf(gamma, scale=10),
            st.uniform.pdf(width_alpha, loc=1, scale=100),
            st.uniform.pdf(width_beta, loc=1, scale=100),
            st.uniform.pdf(width_gamma, loc=1, scale=100),
            st.uniform.pdf(lag_alpha, loc=185, scale=365),
            st.uniform.pdf(lag_beta, loc=550, scale=365),
            st.uniform.pdf(lag_gamma, loc=915, scale=365),
            st.uniform.pdf(lag_last_cone, loc=915, scale=1095),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> np.ndarray:
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

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

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def posterior_predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector

        Returns
        -------
        np.ndarray
            Time series (same shape as `self.raw_data['f']`) of independent cone predictions.
            Note that this returns _transformed_ predictions, not the predictions in the original
            units.
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

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)


class ScaledThreeYearsPreceedingModel(Model):
    """A model with terms for 1, 2, and 3 years preceeding a cone crop."""

    name = "scaled_three_year"
    labels = [
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

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
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

    def log_prior(self, theta: tuple[float, ...]) -> float:
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

        priors = [
            st.uniform.pdf(c0, loc=0, scale=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.halfnorm.pdf(gamma, scale=10),
            st.uniform.pdf(width_alpha, loc=0.01, scale=0.3),
            st.uniform.pdf(width_beta, loc=0.01, scale=0.3),
            st.uniform.pdf(width_gamma, loc=0.01, scale=0.3),
            st.uniform.pdf(lag_alpha, loc=0.5, scale=1),
            st.uniform.pdf(lag_beta, loc=1.5, scale=1),
            st.uniform.pdf(lag_gamma, loc=2.5, scale=1),
            st.uniform.pdf(lag_last_cone, loc=1, scale=5),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> np.ndarray:
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

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

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def posterior_predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector

        Returns
        -------
        np.ndarray
            Time series (same shape as `f` and `c`) of independent cone predictions
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

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)


class ScaledTwoYearsPreceedingModel(Model):
    """A model with terms for 1 and 2 years preceeding a cone crop."""

    name = "scaled_two_year"
    labels = [
        "c0",
        "alpha",
        "beta",
        "width_alpha",
        "width_beta",
        "lag_alpha",
        "lag_beta",
        "lag_last_cone",
    ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
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
        initial = np.array(
            [
                10,  # c0
                5,  # alpha
                5,  # beta
                0.1,  # width_alpha
                0.1,  # width_beta
                1,  # lag_alpha
                2,  # lag_beta
                3,  # lag_last_cone
            ]
        )

        return (
            np.vstack(
                (
                    st.uniform.rvs(loc=-5, scale=5, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.1, scale=0.3, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                    st.uniform.rvs(loc=-0.5, scale=1, size=nwalkers),
                )
            ).T
            + initial
        )

    def log_prior(self, theta: tuple[float, ...]) -> float:
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
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        priors = [
            st.uniform.pdf(c0, loc=0, scale=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.uniform.pdf(width_alpha, loc=0, scale=1),
            st.uniform.pdf(width_beta, loc=0, scale=1),
            st.uniform.pdf(lag_alpha, loc=0.5, scale=1),
            st.uniform.pdf(lag_beta, loc=1.5, scale=1),
            st.uniform.pdf(lag_last_cone, loc=2.5, scale=1),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> np.ndarray:
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

        Returns
        -------
        np.ndarray
            Array containing log-likelihood for every data point for the given theta
        """
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def posterior_predictive(self, theta: tuple[float, ...]) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector

        Returns
        -------
        np.ndarray
            Time series (same shape as `f` and `c`) of independent cone predictions
        """
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta
        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            - lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)


class TwoYearsPreceedingModel(Model):
    """A model with terms for 1 and 2 years preceeding a cone crop."""

    name = "no_gamma"
    labels = [
        "c0",
        "alpha",
        "beta",
        "width_alpha",
        "width_beta",
        "lag_alpha",
        "lag_beta",
        "lag_last_cone",
    ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
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
        initial = np.array(
            [
                10,  # c0
                5,  # alpha
                5,  # beta
                10,  # width_alpha
                10,  # width_beta
                200,  # lag_alpha
                600,  # lag_beta
                1000,  # lag_last_cone
            ]
        )

        return (
            np.vstack(
                (
                    st.randint.rvs(low=-5, high=5, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.norm.rvs(loc=10, scale=1, size=nwalkers),
                    st.randint.rvs(low=-10, high=10, size=nwalkers),
                    st.randint.rvs(low=-10, high=10, size=nwalkers),
                    st.randint.rvs(low=-10, high=10, size=nwalkers),
                    st.randint.rvs(low=-10, high=10, size=nwalkers),
                    st.randint.rvs(low=-10, high=10, size=nwalkers),
                )
            ).T
            + initial
        )

    def log_prior(self, theta: tuple[float, ...]) -> float:
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
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        priors = [
            st.randint.pmf(np.floor(c0), low=0, high=1000),
            st.halfnorm.pdf(alpha, scale=10),
            st.halfnorm.pdf(beta, scale=10),
            st.randint.pmf(np.floor(width_alpha), low=1, high=100),
            st.randint.pmf(np.floor(width_beta), low=1, high=100),
            st.randint.pmf(np.floor(lag_alpha), low=180, high=545),
            st.randint.pmf(np.floor(lag_beta), low=550, high=910),
            st.randint.pmf(np.floor(lag_last_cone), low=915, high=1275),
        ]

        prior = np.prod(priors)
        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> np.ndarray:
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

        Returns
        -------
        np.ndarray
            Array containing log-likelihood for every data point for the given theta
        """
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        # Each date has a different c_mu, so this vector is of shape == c.shape
        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            - lagged(c, lag_last_cone)
        )

        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def posterior_predictive(
        self,
        theta: tuple[float, ...],
    ) -> np.ndarray:
        """Generate a set of independent posterior predictive samples.

        Parameters
        ----------
        theta : tuple[float, ...]
            Model parameter vector

        Returns
        -------
        np.ndarray
            Time series (same shape as `f` and `c`) of independent cone predictions
        """
        c0, alpha, beta, width_alpha, width_beta, lag_alpha, lag_beta, lag_last_cone = theta

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        c_mu: np.ndarray = (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            - lagged(c, lag_last_cone)
        )
        return st.poisson.rvs(c_mu)
