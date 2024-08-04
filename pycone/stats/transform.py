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
