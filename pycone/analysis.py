import multiprocessing
from collections import defaultdict
from typing import Any

import pandas as pd
import rich.progress as rp


def delta_t_2_pass(weather_data: pd.DataFrame) -> dict[Any, Any]:
    """Calculate delta-T in two steps:

    1. Iterate over each year, computing mean T for every combination of start1, start2, and
    duration
    2. Iterate over each year again, subtracting the mean T values

    Parameters
    ----------
    weather_data : pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    progress = rp.Progress(
        "[progress.description]{task.description}",
        rp.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
        refresh_per_second=1,
    )

    with progress:
        overall_progress_task = progress.add_task(
            "[green]Calculating Mean Temperature:"
        )
        with multiprocessing.Manager() as manager:
            results = []
            worker_status = manager.dict()
            with multiprocessing.Pool() as pool:
                for year, df in weather_data.groupby(by="year"):
                    task_id = progress.add_task(
                        f"Year {year}:",
                        visible=False,
                    )
                    results.append(
                        pool.apply_async(
                            calculate_mean_t,
                            (df, year),
                            {"task_id": str(task_id), "worker_status": worker_status},
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

    return dict(result.get() for result in results)


def calculate_mean_t(
    df: pd.DataFrame,
    year: int,
    task_id: str,
    worker_status: dict[str, Any],
    doy_col: str = "day_of_year",
    t_col: str = "tmean (degrees f)",
) -> tuple[int, dict[str, pd.DataFrame]]:
    """Compute the mean temperature for the given year data.

    The input data is grouped by site name, and average temperatures for all possible start days and
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
    tuple[int, dict[str, pd.DataFrame]]
        Mapping between {site name: mean temperature data}
    """

    grouped = df.groupby("name")

    mean_t = {}

    for site_number, (site, site_df) in enumerate(grouped):
        min_doy = site_df[doy_col].min()
        max_doy = site_df[doy_col].max()

        mean_t_at_site = defaultdict(list)
        for start in range(min_doy, max_doy):
            for duration in range(0, max_doy - start):
                end = start + duration
                day_of_year = site_df[doy_col]

                mean_t_at_site["start"].append(start)
                mean_t_at_site["duration"].append(duration)
                mean_t_at_site["mean_t"].append(
                    site_df.loc[
                        (start <= day_of_year) & (day_of_year < end), t_col
                    ].mean()
                )

        mean_t[site] = pd.DataFrame(mean_t_at_site)

        worker_status[task_id] = {"items_completed": site_number, "total": len(grouped)}

    worker_status[task_id] = {"items_completed": len(grouped), "total": len(grouped)}
    return year, mean_t
