import multiprocessing
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import rich.progress as rp


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

            name
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
                for (year, name), df in weather_data.groupby(by=["year", "name"]):
                    task_id = progress.add_task(
                        f"Year {year}, site {name}:",
                        visible=False,
                    )

                    # Initiate mean t calculation
                    results.append(
                        pool.apply_async(
                            calculate_mean_t_site_year,
                            (df, year, name),
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
    name: str,
    task_id: int,
    worker_status: dict[int, Any],
    doy_col: str = "day_of_year",
    t_col: str = "tmean (degrees f)",
) -> tuple[int, dict[str, pd.DataFrame]]:
    """Compute the mean temperature for the given year data at the given site.

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
    mean_t_df["name"] = name
    mean_t_df["year"] = year
    worker_status[task_id] = {
        "items_completed": len(start_range),
        "total": len(start_range),
    }

    return mean_t_df
