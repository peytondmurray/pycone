import numpy as np
import pandas as pd

# import scipy.optimize as so
import scipy.special as ss
import scipy.stats as st

from .. import gsl
from .math import backward_integral, lagged, mavg
from .transform import IdentityTransform


class Model:
    """Container for probability functions for a given model."""

    name: str = ""
    labels: list[str] = []

    def __init__(
        self,
        data: pd.DataFrame,
        preprocess: dict[str, type] | None = None,
        transforms: dict[str, type] | None = None,
    ):
        """Create a Model."""
        self.ndim = len(self.labels)
        self.priors: list = []
        self.raw_data = data

        if preprocess is None:
            preprocess = {}

        if transforms is None:
            transforms = {}

        self.preprocesses = {}
        preprocessed_data = {}
        for col in data.columns:
            self.preprocesses[col] = preprocess.get(col, IdentityTransform)()
            preprocessed_data[col] = self.preprocesses[col].transform(self.raw_data[col])
        self.preprocessed_data = pd.DataFrame(preprocessed_data)

        self.transforms = {}
        transformed_data = {}
        for col in self.preprocessed_data.columns:
            self.transforms[col] = transforms.get(col, IdentityTransform)()
            transformed_data[col] = self.transforms[col].transform(self.preprocessed_data[col])
        self.transformed_data = pd.DataFrame(transformed_data)

        self._transformed_data: dict[str, np.ndarray] = {}

    def get_transformed_data(self, col: str) -> np.ndarray:
        """Get the transformed data for a column.

        If cached, that data is returned; otherwise, cache it first before returning it.

        Parameters
        ----------
        col : str
            Column to get

        Returns
        -------
        np.ndarray
            Column data as an array
        """
        if col in self._transformed_data:
            return self._transformed_data[col]

        self._transformed_data[col] = self.transformed_data[col].to_numpy()
        return self._transformed_data[col]

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

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> tuple[np.ndarray, ...]:
        """Calculate the log likelihood for each data point given the model parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        tuple[np.ndarray, ...]
            The first element is the individual probabilities of observing the data points given
            the model parameters; has the same shape as the raw data `self.raw_data['c']`. The
            rest of the elements in this tuple are emcee blobs to be stored for later.
        """
        raise NotImplementedError

    def sample_prior(self) -> np.ndarray:
        """Generate a sample from the prior distribution.

        Returns
        -------
        np.ndarray
            Array of length self.ndim containing a single prior sample
        """
        return np.array([dist.rvs() for dist in self.priors])

    def predictive(self, theta: tuple[float, ...]) -> np.ndarray:
        """Generate a sample from the prior or posterior predictive distributions.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters. Can be either drawn from the posterior distribution,
            in which case the output will be samples of the posterior predictive distribution,
            or from the priors themselves, which will yield the prior predictive distribution

        Returns
        -------
        np.ndarray
            Array of data generated from the prior or posterior predictive distribution; has the
            same shape as the raw data `self.raw_data['c']`
        """
        raise NotImplementedError


class RAModel(Model):
    """Model which sums the temperature (in Kelvin) cumulatively.

    c_mu = c0 + alpha*cumsum(T) - cumsum(c)
    """

    name = "ramodel"
    labels = [
        "c0",
        "alpha",
    ]
    blobs_dtype = [("t_contribution", float), ("cone_contribution", float)]

    def __init__(
        self,
        data: pd.DataFrame,
        preprocess: dict[str, type] | None = None,
        transforms: dict[str, type] | None = None,
    ):
        """Init an RAModel."""
        super().__init__(data, preprocess, transforms)

        self.priors = [
            st.uniform(loc=0, scale=100),  # c0
            st.halfnorm(loc=0, scale=100),  # alpha
        ]

        self.labels = self.labels
        self.ndim = len(self.priors)

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Init in a gaussian ball around the max of the likelihood.

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
                st.norm.rvs(loc=10, scale=1, size=nwalkers),  # c0
                st.norm.rvs(loc=10, scale=1, size=nwalkers),  # alpha
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
        # prior = np.prod([dist.pdf(param) for dist, param in zip(self.priors, theta, strict=True)])
        prior = gsl.halfnorm_pdf([theta[0]], 10) * gsl.halfnorm_pdf([theta[1]], 10)

        if np.isnan(prior):
            return -np.inf

        return np.log(prior)

    def compute_c_mu(self, theta: tuple[float, ...] | np.ndarray) -> tuple[np.ndarray, ...]:
        """Compute the expected number of cones from the given parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        tuple[np.ndarray, ...]
            Expected number of cones for the entire length of time spanned
            by the dataset; should be of length self.raw_data.shape[0]
        """
        c0, alpha = theta

        c_trans = self.get_transformed_data("c")
        t_trans = self.get_transformed_data("t")

        t_contribution = alpha * t_trans
        cone_contribution = c_trans
        c_mu = c0 + t_contribution - cone_contribution
        return c_mu, t_contribution, cone_contribution

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> tuple[np.ndarray, ...]:
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.
        Each date has a different c_mu, so this vector is of shape == c.shape.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

        Returns
        -------
        np.ndarray
            Array containing log-likelihood for every data point for the given theta
        """
        c = self.get_transformed_data("c")
        c_mu, t_contribution, c_contribution = self.compute_c_mu(theta)

        # Anywhere c_mu <= 0 we reject the sample by setting the probability to -np.inf
        ll = c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))
        # ll[ll <= 0] = -np.inf

        return (
            ll,
            t_contribution,
            c_contribution,
        )

    def log_likelihood(self, theta: tuple[float, ...]) -> float:
        c = self.get_transformed_data("c")
        c_mu, t_contrib, c_contrib = self.compute_c_mu(theta)

        ll = c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

        # Anywhere c_mu <= 0 we reject the sample by setting the probability to -np.inf
        # ll[ll <= 0] = -np.inf

        return np.nansum(ll)

    def predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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
        c_mu, *_ = self.compute_c_mu(theta)

        # Mask off the bad data points. These arise because in the
        # log_probability function, we take `np.nansum` of the log
        # probabilities to ignore points near to regions where there
        # is no measured data.
        mask = (c_mu < 0) | np.isnan(c_mu)
        c_mu[mask] = 0
        c_pred = st.poisson.rvs(c_mu).astype(float)
        c_pred[mask] = np.nan
        return c_pred


class SumModel(Model):
    """Model which sums the temperature (in Kelvin) cumulatively.

    c_mu = c0 + alpha*cumsum(T) - cumsum(c)
    """

    name = "sum"
    labels = [
        "c0",
        "alpha",
    ]
    blobs_dtype = [("t_contribution", float), ("cone_contribution", float)]

    def __init__(
        self,
        data: pd.DataFrame,
        preprocess: dict[str, type] | None = None,
        transforms: dict[str, type] | None = None,
    ):
        """Init an SumModel."""
        super().__init__(data, preprocess, transforms)

        self.nR = len(data) // 365

        self.priors = [
            st.halfnorm(loc=0, scale=10),  # c0
            st.halfnorm(loc=0, scale=1),  # alpha
        ] + [st.halfnorm(loc=0, scale=10) for _ in range(self.nR)]

        # Set the labels and ndims for this model, since they depend on the data
        self.labels = self.labels + [f"R_{i}" for i in range(self.nR)]
        self.ndim = len(self.priors)

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Init in a gaussian ball around the max of the likelihood.

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
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # c0
                st.norm.rvs(loc=10, scale=5, size=nwalkers),  # alpha
                *(st.halfnorm.rvs(loc=0, scale=10, size=nwalkers) for _ in range(self.nR)),
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
        # prior = np.prod([dist.pdf(param) for dist, param in zip(self.priors, theta, strict=True)])

        prior = (
            gsl.halfnorm_pdf([theta[0]], 10)
            * gsl.halfnorm_pdf([theta[1]], 1)
            * np.prod([gsl.halfnorm_pdf([param], 10) for param in theta])
        )

        if np.isnan(prior):
            return -np.inf

        return np.log(prior)

    def compute_c_mu(self, theta: tuple[float, ...] | np.ndarray) -> tuple[np.ndarray, ...]:
        """Compute the expected number of cones from the given parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        tuple[np.ndarray, ...]
            Expected number of cones for the entire length of time spanned
            by the dataset; should be of length self.raw_data.shape[0]
        """
        (c0, alpha, *r) = theta

        c_trans = self.get_transformed_data("c")
        t_trans = self.get_transformed_data("t")

        r_arr = np.repeat(np.cumsum(np.array(r)), self.nR * 365)[: len(c_trans)]

        t_contribution = alpha * t_trans
        cone_contribution = c_trans
        c_mu = c0 + t_contribution - cone_contribution - r_arr
        return c_mu, t_contribution, cone_contribution

    def log_likelihood_vector(self, theta: tuple[float, ...]) -> tuple[np.ndarray, ...]:
        """Compute the log likelihood vector.

        The nansum of this vector returns the log likelihood. Data must be contiguous.
        Each date has a different c_mu, so this vector is of shape == c.shape.

        Parameters
        ----------
        theta : tuple[float, ...]
            Parameters of the model

        Returns
        -------
        np.ndarray
            Array containing log-likelihood for every data point for the given theta
        """
        c = self.get_transformed_data("c")
        c_mu, t_contribution, c_contribution = self.compute_c_mu(theta)

        # Anywhere c_mu <= 0 we reject the sample by setting the probability to -np.inf
        ll = c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))
        ll[ll <= 0] = -np.inf

        return (
            ll,
            t_contribution,
            c_contribution,
        )

    def predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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
        c_mu, *_ = self.compute_c_mu(theta)

        # Mask off the bad data points. These arise because in the
        # log_probability function, we take `np.nansum` of the log
        # probabilities to ignore points near to regions where there
        # is no measured data.
        mask = (c_mu < 0) | np.isnan(c_mu)
        c_mu[mask] = 0
        c_pred = st.poisson.rvs(c_mu).astype(float)
        c_pred[mask] = np.nan
        return c_pred


class ITKModel(Model):
    """Model which Integrates the Temperature (in Kelvin).

    c_mu = c0 + alpha*integral_{t0}^{t}(T) - c[t - lag]
    """

    name = "itk"
    labels = [
        "c0",
        "alpha",
        "t0",
        "lag_last_cone",
    ]

    def __init__(self, *args, **kwargs):
        """Init an ITKModel."""
        super().__init__(*args, **kwargs)
        self.priors = [
            st.halfnorm(loc=0, scale=10),  # c0
            st.halfnorm(loc=0, scale=10),  # alpha
            st.uniform(loc=365, scale=1095),  # t0
            st.uniform(loc=915, scale=1095),  # lag_last_cone
        ]

    def initialize(self, nwalkers: int = 32) -> np.ndarray:
        """Generate initial positions for the MCMC walkers.

        Init in a gaussian ball around the max of the likelihood.

        Parameters
        ----------
        nwalkers : int
            Number of walkers to generate positions for

        Returns
        -------
        np.ndarray
            Array of shape (nwalkers, self.ndim)
        """
        # # Minimize the negative log likelihood
        # res = so.minimize(
        #     lambda theta: -np.nansum(self.log_likelihood_vector(theta)),
        #     x0=[5, 5, 720, 1095],
        #     bounds=[
        #         (0, 1000),
        #         (0, 100),
        #         (365, 2000),
        #         (915, 5000),
        #     ]
        # )
        # print(f"Max log likelihood: {res.x}")
        # return res.x + st.norm.rvs(loc=0, scale=1, size=(nwalkers, self.ndim))

        return np.vstack(
            (
                st.norm.rvs(loc=10, scale=2, size=nwalkers),  # c0
                st.norm.rvs(loc=10, scale=5, size=nwalkers),  # alpha
                st.norm.rvs(loc=720, scale=15, size=nwalkers),  # t0
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
        prior = np.prod([dist.pdf(param) for dist, param in zip(self.priors, theta, strict=True)])

        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def compute_c_mu(self, theta: tuple[float, ...]) -> np.ndarray:
        """Compute the expected number of cones from the given parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        np.ndarray
            Expected number of cones for the entire length of time spanned
            by the dataset; should be of length self.raw_data.shape[0]
        """
        (
            c0,
            alpha,
            t0,
            lag_last_cone,
        ) = theta

        f = self.transformed_data["t"].to_numpy()
        c = self.transformed_data["c"].to_numpy()

        return c0 + alpha * backward_integral(f, t0) - lagged(c, lag_last_cone)

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
        c = self.transformed_data["c"].to_numpy()

        c_mu = self.compute_c_mu(theta)

        # Each date has a different c_mu, so this vector is of shape == c.shape
        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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
        c_mu = self.compute_c_mu(theta)

        # Mask off the bad data points. These arise because in the
        # log_probability function, we take `np.nansum` of the log
        # probabilities to ignore points near to regions where there
        # is no measured data.
        mask = (c_mu < 0) | np.isnan(c_mu)
        c_mu[mask] = 0
        c_pred = st.poisson.rvs(c_mu).astype(float)
        c_pred[mask] = np.nan
        return c_pred


class TYPKelvinModel(Model):
    """Model with terms from temperature contributions for three years before the cone crop."""

    name = "typ_kelvin"
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

    def __init__(self, *args, **kwargs):
        """Instantiate a TYPKelvinModel."""
        super().__init__(*args, **kwargs)
        self.priors = [
            # st.uniform(loc=0, scale=1000), # c0
            st.halfnorm(loc=0, scale=10),  # c0 - priors are way more controlled with this one
            st.halfnorm(loc=0, scale=10),  # alpha
            st.halfnorm(loc=0, scale=10),  # beta
            st.halfnorm(loc=0, scale=10),  # gamma
            st.uniform(loc=1, scale=100),  # width_alpha
            st.uniform(loc=1, scale=100),  # width_beta
            st.uniform(loc=1, scale=100),  # width_gamma
            st.uniform(loc=185, scale=365),  # lag_alpha
            st.uniform(loc=550, scale=365),  # lag_beta
            st.uniform(loc=915, scale=365),  # lag_gamma
            st.uniform(loc=915, scale=1095),  # lag_last_cone
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
                st.norm.rvs(loc=10, scale=5, size=nwalkers),  # alpha
                st.norm.rvs(loc=10, scale=5, size=nwalkers),  # beta
                st.norm.rvs(loc=10, scale=5, size=nwalkers),  # gamma
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
        prior = np.prod([dist.pdf(param) for dist, param in zip(self.priors, theta, strict=True)])

        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def compute_c_mu(self, theta: tuple[float, ...]) -> np.ndarray:
        """Compute the expected number of cones from the given parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        np.ndarray
            Expected number of cones for the entire length of time spanned
            by the dataset; should be of length self.raw_data.shape[0]
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

        return (
            c0
            + alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
            - lagged(c, lag_last_cone)
        )

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
        c = self.transformed_data["c"].to_numpy()

        c_mu = self.compute_c_mu(theta)

        # Each date has a different c_mu, so this vector is of shape == c.shape
        return c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))

    def predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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
        c_mu = self.compute_c_mu(theta)

        # Mask off the bad data points. These arise because in the
        # log_probability function, we take `np.nansum` of the log
        # probabilities to ignore points near to regions where there
        # is no measured data.
        mask = (c_mu < 0) | np.isnan(c_mu)
        c_mu[mask] = 0
        c_pred = st.poisson.rvs(c_mu).astype(float)
        c_pred[mask] = np.nan
        return c_pred


class ThreeYearsPreceedingModel(Model):
    """Model with terms from temperature contributions for three years before the cone crop."""

    name = "three_years_preceeding"
    labels = [
        "c0",
        "alpha_0",
        "alpha_1",
        "alpha_2",
        "width_alpha_0",
        "width_alpha_1",
        "width_alpha_2",
        "lag_alpha_0",
        "lag_alpha_1",
        "lag_alpha_2",
        "lag_3",
    ]

    def __init__(self, *args, **kwargs):
        """Instantiate a ThreeYearsPreceedingModel."""
        super().__init__(*args, **kwargs)
        self.priors = [
            st.halfnorm(loc=0, scale=100),  # c0 - priors are way more controlled with this one
            st.halfnorm(loc=0, scale=20),  # alpha
            st.halfnorm(loc=0, scale=20),  # beta
            st.halfnorm(loc=0, scale=20),  # gamma
            st.uniform(loc=1, scale=100),  # width_alpha
            st.uniform(loc=1, scale=100),  # width_beta
            st.uniform(loc=1, scale=100),  # width_gamma
            st.uniform(loc=185, scale=365),  # lag_alpha
            st.uniform(loc=550, scale=365),  # lag_beta
            st.uniform(loc=915, scale=365),  # lag_gamma
            st.uniform(loc=915, scale=365),  # lag_last_cone
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
                st.norm.rvs(loc=5, scale=1, size=nwalkers),  # alpha
                st.norm.rvs(loc=5, scale=1, size=nwalkers),  # beta
                st.norm.rvs(loc=5, scale=1, size=nwalkers),  # gamma
                st.norm.rvs(loc=30, scale=10, size=nwalkers),  # width_alpha
                st.norm.rvs(loc=30, scale=10, size=nwalkers),  # width_beta
                st.norm.rvs(loc=30, scale=10, size=nwalkers),  # width_gamma
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
        prior = np.prod([dist.pdf(param) for dist, param in zip(self.priors, theta, strict=True)])

        if prior <= 0 or np.isnan(prior):
            return -np.inf
        return np.log(prior)

    def compute_c_mu(self, theta: tuple[float, ...]) -> tuple[np.ndarray, ...]:
        """Compute the expected number of cones from the given parameters.

        Parameters
        ----------
        theta : tuple[float, ...]
            Tuple of model parameters

        Returns
        -------
        np.ndarray
            Expected number of cones for the entire length of time spanned
            by the dataset; should be of length self.raw_data.shape[0]
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

        f = self.get_transformed_data("t")
        c = self.get_transformed_data("c")

        t_contrib = (
            alpha * mavg(f, width_alpha, lag_alpha)
            + beta * mavg(f, width_beta, lag_beta)
            + gamma * mavg(f, width_gamma, lag_gamma)
        )
        c_contrib = lagged(c, lag_last_cone)

        return (c0 + t_contrib - c_contrib), t_contrib, c_contrib

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
        c = self.get_transformed_data("c")
        c_mu, t_contrib, c_contrib = self.compute_c_mu(theta)

        # Anywhere c_mu <= 0 we reject the sample by setting the probability to -np.inf
        ll = c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))
        # ll[ll <= 0] = -np.inf

        # Each date has a different c_mu, so this vector is of shape == c.shape
        return ll, t_contrib, c_contrib

    def log_likelihood(self, theta: tuple[float, ...]) -> float:
        c = self.get_transformed_data("c")
        c_mu, t_contrib, c_contrib = self.compute_c_mu(theta)

        # Anywhere c_mu <= 0 we reject the sample by setting the probability to -np.inf
        ll = c * np.log(c_mu) - c_mu - np.log(ss.factorial(c))
        # ll[ll <= 0] = -np.inf

        # Each date has a different c_mu, so this vector is of shape == c.shape
        return np.nansum(ll)

    def predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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
        c_mu, _, _ = self.compute_c_mu(theta)

        # Mask off the bad data points. These arise because in the
        # log_probability function, we take `np.nansum` of the log
        # probabilities to ignore points near to regions where there
        # is no measured data.
        mask = (c_mu < 0) | np.isnan(c_mu)
        c_mu[mask] = 0
        c_pred = st.poisson.rvs(c_mu).astype(float)
        c_pred[mask] = np.nan
        return c_pred


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

    def predictive(self, theta: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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

    def predictive(self, theta: tuple[float, ...]) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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

    def predictive(
        self,
        theta: tuple[float, ...],
    ) -> np.ndarray:
        """Generate a set of independent prior or posterior predictive samples.

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
