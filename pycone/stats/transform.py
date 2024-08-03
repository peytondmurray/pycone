import numpy as np
import pandas as pd


class Transform:
    def __init__(self):
        pass

    def transform(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError

    def inverse(self, data: pd.Series) -> pd.Series:
        raise NotImplementedError


class IdentityTransform(Transform):
    def transform(self, data: pd.Series) -> pd.Series:
        return data

    def inverse(self, data: pd.Series) -> pd.Series:
        return data


class StandardizeNormal(Transform):
    def __init__(self):
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
