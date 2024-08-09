import numpy as np
import pandas as pd


class Transform:
    """Class which transforms raw data."""

    def __init__(self):
        """Init the Transform class."""
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform raw data to a new space.

        Parameters
        ----------
        data : pd.Series
            Data to transform

        Returns
        -------
        pd.Series
            Transformed data
        """
        raise NotImplementedError

    def inverse(self, data: pd.Series) -> pd.Series:
        """Invert the standardized dataset to recover data in original units.

        Parameters
        ----------
        data : pd.Series
            Dataset to denormalize

        Returns
        -------
        pd.Series
            Unstandardized dataset (i.e. in original units)
        """
        raise NotImplementedError


class IdentityTransform(Transform):
    """The identity transformation (no transformation)."""

    def transform(self, data: pd.Series) -> pd.Series:
        """Transform the raw data; this is the identity, so this does nothing.

        Parameters
        ----------
        data : pd.Series
            Data to transform

        Returns
        -------
        pd.Series
            Transformed data
        """
        return data

    def inverse(self, data: pd.Series) -> pd.Series:
        """Invert the dataset; this is the identity, so this does nothing.

        Parameters
        ----------
        data : pd.Series
            Dataset to transform

        Returns
        -------
        pd.Series
            The same dataset
        """
        return data


class StandardizeNormal(Transform):
    """Transformation which converts normally distributed data to data with mean 0, std 1."""

    def __init__(self):
        """Init the StandardizeNormal class."""
        self.mean = np.nan
        self.std = np.nan

    def transform(self, data: pd.Series) -> pd.Series:
        """Standardize a normally distributed dataset.

        nan-values are ignored.

        Parameters
        ----------
        data : pd.Series
            Dataset to standardize

        Returns
        -------
        pd.Series
            Standardized dataset
        """
        self.mean = np.nanmean(data)
        self.std = np.nanstd(data)
        return (data - self.mean) / self.std

    def inverse(self, data: pd.Series) -> pd.Series:
        """Invert the standardized dataset to recover data in original units.

        Parameters
        ----------
        data : pd.Series
            Dataset to denormalize

        Returns
        -------
        pd.Series
            Unstandardized dataset (i.e. in original units)
        """
        if np.isnan(self.mean):
            raise ValueError("Cannot invert a transform without applying a transform first.")
        return data * self.std + self.mean


class ToKelvinBeforeStandardizeHalfNorm(Transform):
    """Transformation which converts fahrenheit temperatures to kelvin before a half-norm transformation."""

    def __init__(self):
        """Init the ToKelvinBeforeStandardizeHalfNorm class."""
        self.std = np.nan
        self.min = np.nan

    def fahrenheit_to_kelvin(self, data: pd.Series) -> pd.Series:
        """Conver fahrenheit to kelvin.

        Parameters
        ----------
        data : pd.Series
            Fahrenheit temperatures

        Returns
        -------
        pd.Series
            Kelvin temperatures
        """
        return (data - 32) * 5 / 9 + 273.15

    def kelvin_to_fahrenheit(self, data: pd.Series) -> pd.Series:
        """Convert kelvin to fahrenheit.

        Parameters
        ----------
        data : pd.Series
            Kelvin temperatures

        Returns
        -------
        pd.Series
            Fahrenheit temperatures
        """
        return (data - 273.15) * 9 / 5 + 32

    def transform(self, data: pd.Series) -> pd.Series:
        """Standardize the temperature data.

        Parameters
        ----------
        data : pd.Series
            Temperature dataset

        Returns
        -------
        pd.Series
            Standardized temperature; guaranteed to be positive
        """
        kelvin = self.fahrenheit_to_kelvin(data)

        self.min = kelvin.min()
        self.std = kelvin.std()
        return (kelvin - self.min) / self.std

    def inverse(self, data: pd.Series) -> pd.Series:
        """Invert the half-normed data to recover data in original units.

        Parameters
        ----------
        data : pd.Series
            Dataset to invert

        Returns
        -------
        pd.Series
            Unstandardized dataset (i.e. in fahrenheit)
        """
        kelvin = data * self.std + self.min
        return self.kelvin_to_fahrenheit(kelvin)
