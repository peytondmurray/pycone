import multiprocessing as mp
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import rich.progress as rp

from ._pycone_main import fast_delta_t_site_duration


def calculate_delta_t(
    mean_t: pd.DataFrame, duration: int | Iterable[int] | None = None, year_gap: int = 1
) -> pd.DataFrame:
    """Calculate ΔT from the mean temperature data.

    Parameters
    ----------
    mean_t : pd.DataFrame
        Mean temperature data
    year_gap : int
        Time gap between T_year1 and T_year2; specified in years
    duration : int | Iterable
        Duration(s) to calculate data for; if None, all possible durations are calculated.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing ΔT values:

            site: Site code
            year1: Year in which the first interval lies
            year2: Year in which the second interval lies
            start1: Starting day of year for the first interval
            start2: Starting day of year for the second inverval
            duration: Duration of the interval [days]
            delta_t: Difference in the average temperatures for the two intervals [°F]
    """
    progress = rp.Progress(
        "[progress.description]{task.description}",
        rp.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
        refresh_per_second=30,
    )

    with progress:
        overall_progress_task = progress.add_task("[green]Calculating ΔT:")

        # Start a manager so that we can use a dictionary to share worker status info
        with mp.Manager() as status_manager:
            results = []
            worker_status = status_manager.dict()

            with mp.Pool(processes=4) as pool:
                for site, df in mean_t.groupby(by="site"):
                    task_id = progress.add_task(
                        f"Site {site}:",
                        visible=False,
                    )

                    results.append(
                        pool.apply_async(
                            calculate_delta_t_site_fast,
                            (df, site),
                            {
                                "duration": duration,
                                "task_id": task_id,
                                "worker_status": worker_status,
                                "year_gap": year_gap,
                            },
                        )
                    )

                # Monitor worker progress by checking worker_status until all jobs finish
                n_complete = 0
                while n_complete < len(results):
                    progress.update(
                        overall_progress_task,
                        completed=n_complete,
                        total=len(results),
                    )
                    for task_id, status in worker_status.items():
                        completed = status["items_completed"]
                        total = status["total"]
                        progress.update(
                            task_id,
                            completed=completed,
                            total=total,
                            visible=completed < total,
                        )
                    n_complete = sum(result.ready() for result in results)

                progress.update(
                    overall_progress_task,
                    completed=len(results),
                    total=len(results),
                )
    return pd.concat(result.get() for result in results)


def calculate_delta_t_site_fast(
    df: pd.DataFrame,
    site: str,
    duration: int | Iterable | None = None,
    task_id: int | None = None,
    worker_status: dict[int, Any] | None = None,
    year_gap: int = 1,
) -> pd.DataFrame:
    """Calculate the difference in temperature for the given site.

    Data is grouped by duration here because the compiled delta-t calculation is so fast
    we don't get good process utilization if we groupby(site, duration) in the parent process.

    The heavy lifting is done by calculate_delta_t_site_duration_fast; this function wraps that
    calculation, managing progress reporting and appending additional data to the result.

    Parameters
    ----------
    df : pd.DataFrame
        Mean temperature data for the given site
    site : str
        Site where the temperature data was recorded
    duration : int | Iterable
        Duration(s) to calculate data for; if None, all possible durations are calculated.
    task_id : int
        Task ID returned by ``rich.progress.Progress.add_task``, used for reporting progress to the
        main process
    worker_status : dict[int, Any]
        Dictionary where worker status information can be written. This is a multiprocessing-safe
        object shared across all workers.
    year_gap : int
        Gap [years] between year1 and year2. Certain tree species have 3 year reproductive cycles,
        but most have 2 year cycles (1 year gap).

    Returns
    -------
    pd.DataFrame
        Data containing the difference in average temperature for all possible start dates of the
        intervals, for all years of data, for all possible durations. Columns:

            site: Site code
            year1: Year in which the first interval lies
            year2: Year in which the second interval lies
            start1: Starting day of year for the first interval
            start2: Starting day of year for the second inverval
            duration: Duration of the interval [days]
            delta_t: Difference in the average temperatures for the two intervals [°F]
    """
    years = np.sort(df["year"].unique())[:-year_gap]

    if isinstance(duration, int):
        durations, dfs = (duration,), (df.loc[df["duration"] == duration],)
    elif isinstance(duration, Iterable):
        gb = df.loc[df["duration"].isin(duration)].groupby(by="duration")
        durations, dfs = tuple(zip(*tuple(gb), strict=True))
    else:
        gb = df.groupby(by="duration")
        durations, dfs = tuple(zip(*tuple(gb), strict=True))

    is_subprocess = worker_status is not None and task_id is not None
    if is_subprocess:
        worker_status[task_id] = {"items_completed": 0, "total": gb.ngroups}  # type: ignore

    results = []
    for i, (duration, duration_df) in enumerate(zip(durations, dfs, strict=True)):
        result = calculate_delta_t_site_duration_fast(
            duration_df,
            years,
            year_gap,
        )
        result["duration"] = np.full(len(result), duration, dtype=np.short)
        results.append(result)

        if is_subprocess:
            worker_status[task_id] = {"items_completed": i, "total": gb.ngroups}  # type: ignore

    result_df = pd.concat(results, axis=0)
    result_df["site"] = np.full(len(result_df), site, dtype=np.short)
    return result_df


def calculate_delta_t_site_duration_fast(
    df: pd.DataFrame,
    years: Iterable,
    year_gap: int,
) -> pd.DataFrame:
    """Calculate the difference in temperature ΔT for the given site and duration.

    Parameters
    ----------
    df : pd.DataFrame
        Mean temperature data for a given site and duration.
    years : Iterable
        Years over which the data change in mean temperature is to be calculated.
        Each year must be unique.
    year_gap : int
        Gap between years to use for the ΔT calculation.

    Returns
    -------
    pd.DataFrame
        Table of ΔT data for the given site.
    """
    results = defaultdict(list)

    for year1 in years:
        year2 = year1 + year_gap

        df_year1 = df.loc[df["year"] == year1]
        df_year2 = df.loc[df["year"] == year2]

        delta_t, start1, start2 = fast_delta_t_site_duration(
            df_year1["start"].values,
            df_year1["mean_t"].values,
            df_year2["start"].values,
            df_year2["mean_t"].values,
        )

        results["start1"].append(start1)
        results["start2"].append(start2)
        results["delta_t"].append(delta_t)
        results["year1"].append(np.full_like(start1, year1, dtype=np.short))
        results["year2"].append(np.full_like(start1, year2, dtype=np.short))
    return pd.DataFrame({key: np.concatenate(items) for key, items in results.items()})


def calculate_mean_t(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean temperature for all sites for each year.

    Parameters
    ----------
    weather_data : pd.DataFrame
        Weather data for different sites on different years

    Returns
    -------
    pd.DataFrame
        Mean temperature for all sites and years. Columns:

            site
            year
            start
            duration
            mean_t
    """
    progress = rp.Progress(
        "[progress.description]{task.description}",
        rp.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
        refresh_per_second=30,
    )

    with progress:
        overall_progress_task = progress.add_task(
            "[green]Calculating Mean Temperature:"
        )

        # Start a manager so that we can use a dictionary to share worker status info
        with mp.Manager() as manager:
            results = []
            worker_status = manager.dict()
            with mp.Pool() as pool:
                for (year, site), df in weather_data.groupby(by=["year", "site"]):
                    task_id = progress.add_task(
                        f"Year {year}, site {site}:",
                        visible=False,
                    )

                    # Initiate mean t calculation
                    results.append(
                        pool.apply_async(
                            calculate_mean_t_site_year,
                            (df, year, site),
                            {"task_id": task_id, "worker_status": worker_status},
                        )
                    )

                # Monitor worker progress by checking worker_status until all jobs finish
                n_complete = 0
                while n_complete < len(results):
                    progress.update(
                        overall_progress_task,
                        completed=n_complete,
                        total=len(results),
                    )
                    for task_id, status in worker_status.items():
                        completed = status["items_completed"]
                        total = status["total"]
                        progress.update(
                            task_id,
                            completed=completed,
                            total=total,
                            visible=completed < total,
                        )
                    n_complete = sum(result.ready() for result in results)

                progress.update(
                    overall_progress_task,
                    completed=len(results),
                    total=len(results),
                )

    return pd.concat(result.get() for result in results)


def calculate_mean_t_site_year(
    df: pd.DataFrame,
    year: int,
    site: str,
    task_id: int,
    worker_status: dict[int, Any],
    doy_col: str = "day_of_year",
    t_col: str = "tmean (degrees f)",
) -> pd.DataFrame:
    """Compute the mean temperature for the given year data at the given site.

    The input data is grouped by site, and average temperatures for all possible start days and
    durations is calculated.

    Parameters
    ----------
    df : pd.DataFrame
        Weather data for a year
    year : int
        Year for which the temperature data was recorded
    site : str
        Site where the temperature data was recorded
    task_id : int
        Task ID returned by ``rich.progress.Progress.add_task``, used for reporting progress to the
        main process
    worker_status : dict[int, Any]
        Dictionary where worker status information can be written. This is a multiprocessing-safe
        object shared across all workers.
    doy_col : str
        Column name of the day of year for a given measurement
    t_col : str
        Column name for the temperature for a given measurement

    Returns
    -------
    pd.DataFrame
        DataFrame containing mean temperature for all possible starting dates and durations for the
        given site and year
    """
    result = defaultdict(list)

    # Pull out the numpy arrays before looping - it's way faster (20x) than using
    # pandas .loc to filter rows
    doy = df[doy_col].to_numpy()
    temperature = df[t_col].to_numpy()
    min_doy = doy.min()
    max_doy = doy.max()

    start_range = range(min_doy, max_doy)

    if worker_status is not None:
        worker_status[task_id] = {"items_completed": 0, "total": len(start_range)}

    for i, start in enumerate(start_range, start=1):
        for duration in range(1, max_doy - start):
            temp = temperature[(start <= doy) & (doy < start + duration)]
            result["mean_t"].append(np.nan if temp.size == 0 else temp.mean())
            result["start"].append(start)
            result["duration"].append(duration)

        if worker_status is not None:
            worker_status[task_id] = {"items_completed": i, "total": len(start_range)}

    mean_t_df = pd.DataFrame(result)
    mean_t_df["site"] = site
    mean_t_df["year"] = year

    return mean_t_df


def compute_correlation(
    delta_t: pd.DataFrame,
    cones: pd.DataFrame,
) -> pd.DataFrame:
    r"""Compute the Pearson's correlation coefficient between the temperature and cone data.

    For a given site, duration, start1, and start2 dates, compute the Pearson's correlation
    coefficient between the number of cones and the difference in mean temperature across all years
    of data:

    ΔT(site, duration, start1, start2)_i: Difference in mean temperature for a given site,
    duration, ordinal start date of interval 1, ordinal start date of interval 2, duration, for a
    given year i.
    C_j: Cone count for a given year i.

    The two dataframes are joined by site and year2, so that the cone crop is for the later of the
    years that were used for calculating the mean temperature. In other words, the weather in the
    first year is assumed to influence cone crop in the second year.

    .. math::

        \rho_{C\DeltaT} = \frac{\text{cov}(C, \DeltaT)}{\sigma_C\sigma_{\DeltaT}}

    Parameters
    ----------
    delta_t : pd.DataFrame
        Difference in mean temperature for the intervals
        [start1, start1+duration] and [start2, start2+duration] on year1 and year2.
    cones : pd.DataFrame
        Number of cones as a function of year at a given site.

    Returns
    -------
    pd.DataFrame
        Correlation coefficient for a given site, duration, start1, and start2 for all years of
        data.
    """
    # join the weather and ΔT data into a single dataframe

    gb = (
        cones["site", "year", "cones"]
        .merge(
            delta_t,
            how="inner",
            left_on=["site", "year"],
            right_on=["site", "year2"],
        )
        .groupby(by=["site", "duration", "start1", "start2"])
    )

    results = defaultdict(list)

    for (site, duration, start1, start2), df in gb:
        results["site"].append(site)
        results["duration"].append(duration)
        results["start1"].append(start1)
        results["start2"].append(start2)
        results["correlation"].append(df.corr(method="pearson"))

    return pd.DataFrame(results)
