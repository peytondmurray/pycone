import datetime
import multiprocessing as mp
import pathlib
import re
from collections.abc import Iterable
from enum import Enum
from multiprocessing.pool import AsyncResult
from types import TracebackType
from typing import Any, overload

import numpy as np
import pandas as pd
import rich.progress as rp


class CorrelationType(Enum):
    """Correlation calculation types.

    Represents correlations between

        DEFAULT: n and ΔT
        EXP_DT: n and exp(ΔT)
        EXP_DT_OVER_N: n and exp(ΔT/n)

    where n is the cone count and ΔT is the temperature difference.
    """

    DEFAULT = "dt_vs_n"
    EXP_DT = "exp_dt_vs_n"
    EXP_DT_OVER_N = "exp_dt_over_n_vs_n"


# Set to be as small as possible to save memory and increase performance
DTYPES = {
    "mean_t": np.single,
    "start": np.short,
    "duration": np.short,
    "site": np.short,
    "year": np.short,
    "cones": int,
    "tmean (degrees f)": np.single,
    "day_of_year": np.short,
    "delta_t": np.single,
}

SITE_CODES = {
    "10ABAM_OR": 1,
    "11ABAM_OR": 2,
    "12ABGR_OR": 3,
    "13ABMAS_OR": 4,
    "14ABPR_OR": 5,
    "15ABPR_OR": 6,
    "16PIMO3_OR": 7,
    "17TSME_OR": 8,
    "18TSME_OR": 9,
    "19TSME_OR": 10,
    "1ABAM_OR": 11,
    "21PIEN_WA": 12,
    "22ABAM_WA": 13,
    "23ABLA_WA": 14,
    "25TSME_WA": 15,
    "26TSME_WA": 16,
    "27PIMO3_WA": 17,
    "28ABAM_WA": 18,
    "2ABCO_OR": 19,
    "30ABAM_WA": 20,
    "31ABAM_WA": 21,
    "33ABAM_WA": 22,
    "35ABAM_WA": 23,
    "36ABAM_WA": 24,
    "37ABAM_WA": 25,
    "39ABGR_WA": 26,
    "3ABCO_OR": 27,
    "40ABGR_WA": 28,
    "43ABMAS_WA": 29,
    "44ABPR_WA": 30,
    "45ABPR_WA": 31,
    "46ABPR_WA": 32,
    "48ABPR_WA": 33,
    "49ABPR_WA": 34,
    "4ABLA_OR": 35,
    "50ABPR_WA": 36,
    "54PIMO3_WA": 37,
    "55PIMO3_WA": 38,
    "57PIMO3_WA": 39,
    "58TSME_WA": 40,
    "59TSME_WA": 41,
    "5ABMA_OR": 42,
    "60TSME_WA": 43,
    "6ABMA_OR": 44,
    "7PILA_OR": 45,
    "8TSME_OR": 46,
    "9TSME_OR": 47,
}

SITE_CODES_INVERSE = {val: key for key, val in SITE_CODES.items()}


def get_species(site: str | int | np.integer) -> str:
    """Get the species given the site.

    Parameters
    ----------
    site : str | int
        Site or site code

    Returns
    -------
    str
        Species name extracted from the site name
    """
    if isinstance(site, int | np.integer):
        site_str = code_to_site(site)
    else:
        site_str = site

    matched = re.match(r"\d+(?P<abbrev>\w+)_[A-Z]+", site_str)
    if not matched:
        raise ValueError("No match found for site code.")

    return matched.group("abbrev")


def get_crop_year_gap(site: str | int) -> int:
    """Get the crop year gap for the given site.

    Parameters
    ----------
    site : str | int
        Site or site code

    Returns
    -------
    int
        Crop year gap for the given site. Most species have 1 year between when the
        cones are pollinated and when they are mature, but PIMO and PILA have 2.
    """
    species = get_species(site)
    if string_contains(species, ["PIMO", "PILA"]):
        return 2
    return 1


def write_data(df: pd.DataFrame, path: str | pathlib.Path):
    """Write data to a csv file.

    If it isn't already, site data will be converted to numeric values
    to save space and to make io faster. This saves ~25% of the total
    size of the data just by converting this one column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be written
    path : str | pathlib.Path
        Path to write the data to
    """
    if "site" in df.columns and df["site"].dtype == "object":
        df = df.replace({"site": SITE_CODES})

    with open(path, "w") as f:
        df.to_csv(f, index=False)


def read_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Read data from a file.

    Importantly, this function sets the appropriate dtypes for the columns
    to ensure best performance.

    Parameters
    ----------
    path : str | pathlib.Path
        File to be read

    Returns
    -------
    pd.DataFrame
        Data read from the file
    """
    with rp.open(path, "rb") as f:
        return pd.read_csv(f, dtype=DTYPES)


class ParallelExecutor:
    """Context manager which provides a process pool with a managed dict and a progress bar."""

    def __init__(self, overall_description: str, processes: int | None = None):
        """Process pool which manages status updates from workers."""
        self.progress = rp.Progress(
            "[progress.description]{task.description}",
            rp.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            rp.TimeRemainingColumn(),
            rp.TimeElapsedColumn(),
            refresh_per_second=30,
        )
        self.manager = mp.Manager()
        self.pool = mp.Pool(processes=processes)
        self.overall_description = overall_description
        self.results: list[AsyncResult] = []

    def __enter__(self):
        self.overall_progress_task = self.progress.add_task(self.overall_description)
        self.worker_status = self.manager.dict()
        self.progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType,
    ):
        self.progress.stop()

    def add_task(self, description: str, **kwargs) -> int:
        """Add a task to the progress bar.

        Parameters
        ----------
        description : str
            Description to display for the progress bar
        **kwargs
            Additional kwargs passed to rich.progress.Progress.add_task

        Returns
        -------
        int
            Task ID for the new task bar
        """
        return self.progress.add_task(description, **kwargs)

    def wait_for_results(self) -> list[Any]:
        """Monitor worker progress until all jobs finish and display the progress.

        Returns
        -------
        list[Any]
            List of return values from the functions executed by the pool.
            The order of this list matches the order in which functions and
            arguments were called with ParallelExecutor.apply_async.
        """
        try:
            # Monitor worker progress by checking worker_status until all jobs finish
            n_complete = 0
            while n_complete < len(self.results):
                self.progress.update(
                    self.overall_progress_task,
                    completed=n_complete,
                    total=len(self.results),
                )
                for task_id, status in self.worker_status.items():
                    completed = status["items_completed"]
                    total = status["total"]
                    self.progress.update(
                        task_id,
                        completed=completed,
                        total=total,
                        visible=completed < total,
                    )
                n_complete = sum(result.ready() for result in self.results)

            # Clean up any progress bars that aren't hidden already
            for task_id, status in self.worker_status.items():
                completed = status["items_completed"]
                total = status["total"]
                self.progress.update(
                    task_id,
                    completed=completed,
                    total=total,
                    visible=False,
                )

            # Set the main progress bar to 100%
            self.progress.update(
                self.overall_progress_task,
                completed=len(self.results),
                total=len(self.results),
            )
            return [result.get() for result in self.results]
        finally:
            self.pool.close()
            self.pool.join()

    def apply_async(self, *args, **kwargs):
        """Apply a function asynchronously.

        The result of the function call is captured on `self.results` as an AsyncResult.
        Wait for all processes to finish with `ParallelExecutor.wait_for_results`.

        Parameters
        ----------
        *args
            Arguments to be passed to `multiprocessing.Pool.apply_async`
        **kwargs
            Keyword arguments to be passed to `multiprocessing.Pool.apply_async`

        """
        self.results.append(self.pool.apply_async(*args, **kwargs))


@overload
def site_to_code(site: str) -> int:
    ...


@overload
def site_to_code(site: list[str]) -> list[int]:
    ...


def site_to_code(site: str | list[str]) -> int | list[int]:
    """Convert a string site name to an integer code for performance.

    Parameters
    ----------
    site : str | Iterable[str]
        Name of the site to convert

    Returns
    -------
    int | list[int]
        Unique integer site code
    """
    if isinstance(site, str):
        return SITE_CODES[site]
    return [SITE_CODES[s] for s in site]


@overload
def code_to_site(site: int) -> str:
    ...


@overload
def code_to_site(site: list[int]) -> list[str]:
    ...


def code_to_site(site: int | list[int]) -> str | list[str]:
    """Map the site code to the corresponding site name.

    Parameters
    ----------
    site : int | Iterable[int]
        Unique site integer to convert back into a site name

    Returns
    -------
    str | list[str]
        Original site name corresponding to the site code
    """
    if isinstance(site, int | np.integer):
        return SITE_CODES_INVERSE[site]
    return [SITE_CODES_INVERSE[s] for s in site]


def make_pixel_map(
    x_meas: np.ndarray,
    y_meas: np.ndarray,
    z_meas: np.ndarray,
    extent: tuple[int, int, int, int],
) -> np.ndarray:
    """Reorder the z-data into a 2D array, with data populated based on the x and y variables.

    Parameters
    ----------
    x_meas : np.ndarray
        X data
    y_meas : np.ndarray
        Y data
    z_meas : np.ndarray
        Z data to plot as a colormap
    extent : tuple[int, int, int, int]
        [xmin, xmax, ymin, ymax] to use for the dataset; will be used by imshow to display this
        pixel map

    Returns
    -------
    np.ndarray
        The intput data as a 2D array ready to be plotted by imshow. Missing values are set to
        np.nan
    """
    x = np.arange(extent[0], extent[1] + 1)
    y = np.arange(extent[2], extent[3] + 1)
    xx, yy = np.meshgrid(x, y)
    zz = np.full_like(xx, np.nan, dtype=float)

    x_sorter = np.argsort(x)
    xi = x_sorter[np.searchsorted(x, x_meas, sorter=x_sorter)]

    y_sorter = np.argsort(y)
    yi = y_sorter[np.searchsorted(y, y_meas, sorter=y_sorter)]

    zz[yi, xi] = z_meas
    return zz


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the columns of df which match items in DTYPES to those types.

    Parameters
    ----------
    df : pd.DataFrame
        Data table for which dtypes should be casted

    Returns
    -------
    pd.DataFrame
        Casted dataframe
    """
    return df.astype({key: val for key, val in DTYPES.items() if key in df.columns})


def separate_pine_sites(sites: Iterable[str]) -> tuple[list[str], list[str]]:
    """Separate a list of sites into two groups: pines, and other species.

    Parameters
    ----------
    sites : Iterable[str | int]
        List of sites to separate out.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        [pine sites, other sites]
    """
    pine_sites, other_sites = [], []

    for site in sites:
        if isinstance(site, int):
            site = code_to_site(site)

        if string_contains(site.lower(), ["pila", "pimo"]):
            pine_sites.append(site)
        else:
            other_sites.append(site)

    return pine_sites, other_sites


def split_pine_sites(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset with a `sites` column into two: the pines and the others.

    A site is considered to be a pine site if it has either 'pila' or 'pimo' in the site name.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at least one column: `sites`

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (Pine data, other species data)
    """
    pine_sites = []
    other_sites = []

    for site in SITE_CODES:
        if string_contains(site.lower(), ["pila", "pimo"]):
            pine_sites.append(site)
        else:
            other_sites.append(site)

    return (
        df.loc[df["site"].isin(site_to_code(pine_sites))],
        df.loc[df["site"].isin(site_to_code(other_sites))],
    )


def string_contains(string: str, substr: Iterable[str]) -> bool:
    """Test whether a string contains any of the specified substrings.

    Parameters
    ----------
    string : str
        String to check the presence of substrings for
    substr : Iterable[str]
        Substrings to check

    Returns
    -------
    bool
        True if at least one of the substrings is present in the string
    """
    return any(s in string for s in substr)


class Group:
    """Class which stores information about site groupings for correlations.

    A Group can represent a single site or multiple sites grouped together.
    """

    def __init__(self, name: str, sites: list[int] | None = None, **correlation_kwargs):
        """Instantiate a Group.

        Parameters
        ----------
        name : str
            Name of the group
        sites : list[int] | None
            Site codes which are members of the group
        **correlation_kwargs
            Other kwargs to be passed to analysis.correlation_group
        """
        self.name = name
        self.sites = sites if sites else []
        self.correlation_kwargs = correlation_kwargs if correlation_kwargs else {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(vars(self))


def to_day_since_start(data: pd.DataFrame) -> pd.DataFrame:
    first_data = data.loc[
        (data["year"] == data["year"].min()) & (data["start"] == data["start"].min())
    ]

    first_date = datetime.date(data["year"], 1, 1) + datetime.timedelta(
        days=first_data["start"] - 1
    )

    data["days_since_start"] = (
        datetime.date(data["year"], 1, 1) + datetime.timedelta(days=data["start"])
    ) - first_date

    data["days_since_start"] = pd.to_dateime
