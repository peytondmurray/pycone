import pathlib

import pandas as pd
from rich.console import Console
from rich.style import Style

from . import analysis, output, preprocess, util

console = Console()


def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the analysis to generate correlation data, or fetch it from disk if it exists already.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (cones, weather, mean_t, correlation) data
    """
    cones_path = pathlib.Path("cones.csv")
    weather_path = pathlib.Path("weather.csv")
    mean_t_path = pathlib.Path("mean_t.csv")
    correlation_path = pathlib.Path("correlation.csv")

    console.rule(
        "[bold yellow]Load weather and cone data", style=Style(color="dark_red")
    )
    if cones_path.exists():
        console.log(f"{cones_path} already exists in the current directory; skipping.")
        cones = util.read_data(cones_path)
    else:
        cones = preprocess.load_cone_crop(cones_path)
        util.write_data(cones, cones_path)

    if weather_path.exists():
        console.log(
            f"{weather_path} already exists in the current directory; skipping."
        )
        weather = util.read_data(weather_path)
    else:
        weather = preprocess.load_weather(weather_path)
        util.write_data(weather, weather_path)

    console.rule(
        "[bold yellow]Compute the mean temperature", style=Style(color="dark_red")
    )
    if mean_t_path.exists():
        console.log(f"{mean_t_path} already exists in the current directory; skipping.")
        mean_t = util.read_data(mean_t_path)
    else:
        mean_t = analysis.calculate_mean_t(weather)
        util.write_data(mean_t, mean_t_path)

    console.rule("[bold yellow]Compute the correlation", style=Style(color="dark_red"))
    if correlation_path.exists():
        console.log(
            f"{correlation_path} already exists in the current directory; skipping."
        )
        correlation = util.read_data(correlation_path)
    else:
        correlation = analysis.compute_correlation_from_mean(mean_t, cones)
        util.write_data(correlation, correlation_path)

    return cones, weather, mean_t, correlation


def run_analysis():
    """Run the pycone analysis.

    This function will load the weather and cone data and do some preprocessing. The resulting
    dataframes will be written to the current directory as 'weather.csv' and 'cones.csv',
    respectively.

    Then, the weather data will be used to calculate the mean temperature for all intervals that fit
    in the data for each year and site. The resulting mean temperature data is written to the
    current directory as 'mean_t.csv'.

    The mean temperature data is then used to calculate the ΔT for each site, for every possible
    start date and duration of the intervals on every pair of years. This data is so large that it
    is impractical to write to disk, so at this point we just calculate the correlation between ΔT
    and the number of cones for the site.

    """
    cones, weather, mean_t, correlation = get_data()
    output.plot_correlation_duration_grids(
        correlation,
        nrows=18,
        ncols=12,
        figsize=(40, 60),
        extent=[50, 280, 50, 280],
        filename="site_{}_correlations.svg",
    )
