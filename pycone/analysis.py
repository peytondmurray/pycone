import multiprocessing
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import rich.progress as rp


def calculate_delta_t(mean_t: pd.DataFrame, year_gap: int = 1) -> pd.DataFrame:
    """Calculate ΔT from the mean temperature data.

    Parameters
    ----------
    year_gap : int
        Time gap between T_year1 and T_year2; specified in years
    mean_t : pd.DataFrame
        Mean temperature data

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
        with multiprocessing.Manager() as manager:
            results = []
            worker_status = manager.dict()

            with multiprocessing.Pool() as pool:
                for (site, duration), df in mean_t.groupby(["site", "duration"]):
                    task_id = progress.add_task(
                        f"Site {site}, duration {duration}:",
                        visible=False,
                    )

                    # Initiate delta t calculation
                    results.append(
                        pool.apply_async(
                            calculate_delta_t_site_duration,
                            (df, site, duration),
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


def calculate_delta_t_site_duration(
    df: pd.DataFrame,
    site: str,
    duration: int,
    task_id: int,
    worker_status: dict[int, Any],
    year_gap: int = 1,
) -> pd.DataFrame:
    """Calculate the difference in temperature for the given site and duration.

    Parameters
    ----------
    df : pd.DataFrame
        Mean temperature data for the given site and duration
    site : str
        Site where the temperature data was recorded
    duration : int
        Duration of the interval over which averages were calculated [days]
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
        intervals, for all years of data. Columns:

            site: Site code
            year1: Year in which the first interval lies
            year2: Year in which the second interval lies
            start1: Starting day of year for the first interval
            start2: Starting day of year for the second inverval
            duration: Duration of the interval [days]
            delta_t: Difference in the average temperatures for the two intervals [°F]
    """
    result = defaultdict(list)

    year = df["year"].values
    start = df["start"].values
    mean_t = df["mean_t"].values

    years = np.arange(year.min(), year.max() - year_gap)
    n_years = len(years)
    worker_status[task_id] = {"items_completed": 0, "total": n_years}

    for i, year1 in enumerate(years, start=1):
        # Before iterating the inner loops, precompute as much as we can.
        year2 = year1 + year_gap
        year1_mask = year == year1
        year2_mask = year == year2
        year1_start = np.unique(start[year1_mask])
        year2_start = np.unique(start[year2_mask])
        result["year1"].extend([year1] * len(year1_start) * len(year2_start))
        result["year2"].extend([year2] * len(year2_start) * len(year2_start))

        # For each value in start1 we iterate over start2; this is exactly
        # what meshgrid does. Precompute here to save time.
        y1s, y2s = np.meshgrid(year1_start, year2_start)
        result["start1"].extend(y1s.T.flatten().tolist())
        result["start2"].extend(y2s.T.flatten().tolist())

        for start1 in year1_start:
            mean_t_start1 = mean_t[(year1_mask) & (start == start1)]

            for start2 in year2_start:
                # Guaranteed to be a single value
                result["delta_t"].append(
                    mean_t[(year2_mask) & (start == start2)] - mean_t_start1
                )

        worker_status[task_id] = {"items_completed": i, "total": n_years}

    result_df = pd.DataFrame(result)
    result_df["site"] = site
    result_df["duration"] = duration

    return result_df


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
        with multiprocessing.Manager() as manager:
            results = []
            worker_status = manager.dict()
            with multiprocessing.Pool() as pool:
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
    doy = df[doy_col].values
    temperature = df[t_col].values
    min_doy = doy.min()
    max_doy = doy.max()

    start_range = range(min_doy, max_doy)
    for i, start in enumerate(start_range):
        for duration in range(1, max_doy - start):
            temp = temperature[(start <= doy) & (doy < start + duration)]
            result["mean_t"].append(np.nan if temp.size == 0 else temp.mean())
            result["start"].append(start)
            result["duration"].append(duration)
        worker_status[task_id] = {"items_completed": i, "total": len(start_range)}

    mean_t_df = pd.DataFrame(result)
    mean_t_df["site"] = site
    mean_t_df["year"] = year
    worker_status[task_id] = {
        "items_completed": len(start_range),
        "total": len(start_range),
    }

    return mean_t_df
