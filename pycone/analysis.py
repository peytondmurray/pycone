from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from . import util
from ._pycone_main import fast_delta_t_site_duration


def calculate_delta_t(
    mean_t: pd.DataFrame,
    duration: int | Iterable[int] | None = None,
    delta_t_year_gap: int = 1,
    output_path: str | None = None,
) -> pd.DataFrame | None:
    """Calculate ΔT from the mean temperature data.

    Parameters
    ----------
    mean_t : pd.DataFrame
        Mean temperature data. Must contain columns

            site
            year
            start
            duration
            mean_t

    duration : int | Iterable
        Duration(s) to calculate data for; if None, all possible durations are calculated.
    delta_t_year_gap : int
        Time gap between T_year1 and T_year2; specified in years
    output_path : str | None
        Optional path where the data should be written. If specified, None is returned.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame containing ΔT values:

            site: Site code
            year1: Year in which the first interval lies
            year2: Year in which the second interval lies
            start1: Starting day of year for the first interval
            start2: Starting day of year for the second inverval
            duration: Duration of the interval [days]
            delta_t: Difference in the average temperatures for the two intervals [°F]

        If an output_path is specified, None is returned, and data is written to the path instead.
    """
    with util.ParallelExecutor("[green]Calculating ΔT:", processes=8) as pe:
        for site, df in mean_t.groupby(by="site"):
            task_id = pe.add_task(
                f"Site {site}:",
                visible=False,
            )
            pe.apply_async(
                calculate_delta_t_site_fast,
                (df, site),
                {
                    "duration": duration,
                    "task_id": task_id,
                    "worker_status": pe.worker_status,
                    "year_gap": delta_t_year_gap,
                    "output_path": output_path,
                },
            )

        results = pe.wait_for_results()

    if output_path is None:
        return pd.concat(results)
    return None


def calculate_delta_t_site_fast(
    df: pd.DataFrame,
    site: int,
    duration: int | Iterable | None = None,
    task_id: int | None = None,
    worker_status: dict[int, Any] | None = None,
    year_gap: int = 1,
) -> pd.DataFrame | None:
    """Calculate the difference in temperature for the given site.

    Data is grouped by duration here because the compiled delta-t calculation is so fast
    we don't get good process utilization if we groupby(site, duration) in the parent process.

    The heavy lifting is done by calculate_delta_t_site_duration_fast; this function wraps that
    calculation, managing progress reporting and appending additional data to the result.

    Parameters
    ----------
    df : pd.DataFrame
        Mean temperature data for the given site
    site : int
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
    pd.DataFrame | None
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
        worker_status[task_id] = {"items_completed": 0, "total": len(durations)}  # type: ignore

    results = []
    for i, (duration, duration_df) in enumerate(
        zip(durations, dfs, strict=True), start=1
    ):
        result = calculate_delta_t_site_duration_fast(
            duration_df,
            years,
            year_gap,
        )
        result["duration"] = np.full(len(result), duration, dtype=np.short)
        results.append(result)

        if is_subprocess:
            worker_status[task_id] = {"items_completed": i, "total": len(durations)}  # type: ignore

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
        Table of ΔT data for the given site:

            start1
            start2
            delta_t
            year1
            year2
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
        Weather data for different sites on different years. Must have columns

            site
            year
            day_of_year
            tmean (degrees f)

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
    with util.ParallelExecutor("[green]Calculating Mean Temperature:") as pe:
        for (year, site), df in weather_data.groupby(by=["year", "site"]):
            task_id = pe.add_task(
                f"Year {year}, site {site}:",
                visible=False,
            )
            pe.apply_async(
                calculate_mean_t_site_year,
                (df, year, site),
                {"task_id": task_id, "worker_status": pe.worker_status},
            )
        return pd.concat(pe.wait_for_results())


def calculate_mean_t_site_year(
    df: pd.DataFrame,
    year: int,
    site: str,
    task_id: int | None = None,
    worker_status: dict[int, Any] | None = None,
    doy_col: str = "day_of_year",
    t_col: str = "tmean (degrees f)",
) -> pd.DataFrame:
    """Compute the mean temperature for the given year data at the given site.

    The input data is grouped by site, and average temperatures for all possible start days and
    durations is calculated.

    The mean of the data is calculated using the trapezoid rule (rather than a simple arithmetic
    mean), so gaps in the data are correctly accounted for.

    Parameters
    ----------
    df : pd.DataFrame
        Weather data for a year
    year : int
        Year for which the temperature data was recorded
    site : str
        Site where the temperature data was recorded
    task_id : int | None
        Task ID returned by ``rich.progress.Progress.add_task``, used for reporting progress to the
        main process
    worker_status : dict[int, Any] | None
        Dictionary where worker status information can be written. This is a multiprocessing-safe
        object shared across all workers. If None, no progress is reported.
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
    # Preallocate to cover the edge cases where data is recorded for 1 or 2 days in the year
    result: dict[str, list[float | int]] = {
        "mean_t": [],
        "start": [],
        "duration": [],
    }

    # Pull out the numpy arrays before looping - it's way faster (20x) than using
    # pandas .loc to filter rows
    doy = df[doy_col].to_numpy()
    temperature = df[t_col].to_numpy()

    is_subprocess = worker_status is not None and task_id is not None
    if is_subprocess:
        worker_status[task_id] = {"items_completed": 0, "total": len(doy) - 1}  # type: ignore

    for i, start in enumerate(doy):
        for end in doy[i + 1 :]:
            doy_interval = doy[(start <= doy) & (doy <= end)]
            temperature_interval = temperature[(start <= doy) & (doy <= end)]
            duration = end - start

            if temperature_interval.size == 0:
                mean_temp = np.nan
            else:
                if duration == 1:
                    # Trapezoid rule really only makes sense for 2 or more points; just
                    # use the temperature of the single day in this case.
                    mean_temp = temperature_interval[0]
                else:
                    mean_temp = (
                        np.trapz(y=temperature_interval, x=doy_interval) / duration
                    )

            result["mean_t"].append(mean_temp)
            result["start"].append(start)
            result["duration"].append(duration)

        if is_subprocess:
            worker_status[task_id] = {"items_completed": i, "total": len(doy) - 1}  # type: ignore

    mean_t_df = pd.DataFrame(result)
    mean_t_df["site"] = site
    mean_t_df["year"] = year
    return mean_t_df


def compute_correlation_from_mean(
    mean_t: pd.DataFrame,
    cones: pd.DataFrame,
    delta_t_year_gap: int = 1,
    crop_year_gap: int = 1,
) -> pd.DataFrame:
    """Compute the correlation between ΔT and the number of cones from mean temperature.

    This function skips the intermediate step of storing ΔT, which can take up huge amounts of
    memory.

    Parameters
    ----------
    mean_t : pd.DataFrame
        Mean temperature data
    cones : pd.DataFrame
        Cone crop data
    delta_t_year_gap : int
        Gap between years for which ΔT is calculated
    crop_year_gap : int
        Gap between the second year used for calculating ΔT and the year in which the cone crop is
        correlated

    Returns
    -------
    pd.DataFrame
        Correlation DataFrame for all sites; columns:

            start1
            start2
            correlation
            site
            duration
    """
    with util.ParallelExecutor("[green]Calculating correlation:") as pe:
        for site, df in mean_t.groupby(by="site"):
            task_id = pe.add_task(
                f"Site {site}:",
                visible=False,
            )
            pe.apply_async(
                calculate_correlation_from_mean_site,
                (df, site, cones.loc[cones["site"] == site]),
                {
                    "task_id": task_id,
                    "worker_status": pe.worker_status,
                    "delta_t_year_gap": delta_t_year_gap,
                    "crop_year_gap": crop_year_gap,
                },
            )

        return pd.concat(pe.wait_for_results())


def calculate_correlation_from_mean_site(
    df: pd.DataFrame,
    site: int,
    cones: pd.DataFrame,
    durations: int | Iterable | None = None,
    task_id: int | None = None,
    worker_status: dict[int, Any] | None = None,
    delta_t_year_gap: int = 1,
    crop_year_gap: int = 1,
) -> pd.DataFrame:
    """Calculate the cone crop correlation from mean temperature data per-site.

    Parameters
    ----------
    df : pd.DataFrame
        Mean temperature data for the given site
    site : int
        Site code
    cones : pd.DataFrame
        Cone data for the given site
    durations : int | Iterable | None
        Duration(s) to calculate data for; if None, all possible durations are calculated
    task_id : int | None
        Task ID returned by ``rich.progress.Progress.add_task``, used for reporting progress to the
        main process
    worker_status : dict[int, Any] | None
        Dictionary where worker status information can be written. This is a multiprocessing-safe
        object shared across all workers
    delta_t_year_gap : int
        Gap [years] between year1 and year2. Certain tree species have 3 year reproductive cycles,
        but most have 2 year cycles (1 year gap)
    crop_year_gap : int
        Gap between the second year used for calculating ΔT and the year in which the cone crop is
        correlated

    Returns
    -------
    pd.DataFrame
        Correlation DataFrame for the given site with columns

            start1
            start2
            correlation
            site
            duration
    """
    years = np.sort(df["year"].unique())[:-delta_t_year_gap]

    if isinstance(durations, int):
        durations, dfs = (durations,), (df.loc[df["duration"] == durations],)
    elif isinstance(durations, Iterable):
        gb = df.loc[df["duration"].isin(durations)].groupby(by="duration")
        durations, dfs = tuple(zip(*tuple(gb), strict=True))
    else:
        gb = df.groupby(by="duration")
        durations, dfs = tuple(zip(*tuple(gb), strict=True))

    is_subprocess = worker_status is not None and task_id is not None
    if is_subprocess:
        worker_status[task_id] = {"items_completed": 0, "total": len(durations)}  # type: ignore

    results = []
    for i, (duration, duration_df) in enumerate(
        zip(durations, dfs, strict=True), start=1
    ):
        delta_t = calculate_delta_t_site_duration_fast(
            duration_df, years, year_gap=delta_t_year_gap
        )
        delta_t["site"] = site
        delta_t["crop_year"] = delta_t["year2"] + crop_year_gap

        dt_cone_df = cones[["site", "year", "cones"]].merge(
            delta_t,
            how="inner",
            left_on=["site", "year"],
            right_on=["site", "crop_year"],
        )
        results.append(compute_correlation_site_duration(dt_cone_df, site, duration))
        if is_subprocess:
            worker_status[task_id] = {"items_completed": i, "total": len(durations)}  # type: ignore

    return pd.concat(results)


def compute_correlation(
    delta_t: pd.DataFrame,
    cones: pd.DataFrame,
    crop_year_gap: int = 1,
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
    crop_year_gap: int
        Time gap between year2 and the year of the expected cone crop. (year2+crop_year_gap) is the
        year that is used to join the temperature differences with cone crop data before
        correlations are calculated.

    Returns
    -------
    pd.DataFrame
        Correlation coefficient for a given site, duration, start1, and start2 for all years of
        data.
    """
    delta_t["crop_year"] = delta_t["year2"] + crop_year_gap
    with util.ParallelExecutor("[green]Calculating correlation:") as pe:
        gb = (
            cones[["site", "year", "cones"]]
            .merge(
                delta_t,
                how="inner",
                left_on=["site", "year"],
                right_on=["site", "crop_year"],
            )
            .groupby(by=["site", "duration"])
        )

        for (site, duration), df in gb:
            task_id = pe.add_task(f"Site {site}, duration {duration}:", visible=False)
            pe.apply_async(
                compute_correlation_site_duration,
                (df, site, duration),
                {"task_id": task_id, "worker_status": pe.worker_status},
            )

        return pd.concat(pe.wait_for_results())


def compute_correlation_site_duration(
    data: pd.DataFrame,
    site: int,
    duration: int,
    task_id: int | None = None,
    worker_status: dict[int, Any] | None = None,
    dt_col: str = "delta_t",
    cones_col: str = "cones",
) -> pd.DataFrame:
    """Compute the correlation for a given site and duration.

    Parameters
    ----------
    data : pd.DataFrame
        Merged delta-T and cone crop data. Must have the following columns:

            delta_t
            cones
            start1
            start2

    site : int
        Integer corresponding to the site from which the data was taken
    duration : int
        Duration of the intervals used to calculate the mean temperature (and therefore delta-T)
    task_id : int | None
        Task ID returned by ``rich.progress.Progress.add_task``, used for reporting progress to the
        main process
    worker_status : dict[int, Any]
        Dictionary where worker status information can be written. This is a multiprocessing-safe
        object shared across all workers.
    dt_col : str
        Name of the column containing delta-T data
    cones_col : str
        Name of the column containing cone count data

    Returns
    -------
    pd.DataFrame
        Cones/delta-T Pearson's correlation coefficient for all years for the given site and
        duration, for each value of start1 and start2 in the input data. Has columns:

            start1
            start2
            correlation
            site
            duration
    """
    results = defaultdict(list)
    gb = data.groupby(["start1", "start2"])

    is_subprocess = worker_status is not None and task_id is not None
    if is_subprocess:
        worker_status[task_id] = {"items_completed": 0, "total": gb.ngroups}  # type: ignore

    for i, ((start1, start2), df) in enumerate(gb, start=1):
        results["start1"].append(start1)
        results["start2"].append(start2)
        results["correlation"].append(
            df[[dt_col, cones_col]].corr(method="pearson")[dt_col][cones_col]
        )

        if is_subprocess:
            worker_status[task_id] = {"items_completed": i, "total": gb.ngroups}  # type: ignore

    result = pd.DataFrame(results)
    result["site"] = site
    result["duration"] = duration
    return result
