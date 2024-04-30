import pathlib

import pandas as pd
from rich.console import Console
from rich.style import Style

from . import analysis, output, preprocess, util

console = Console()


def load_cones() -> pd.DataFrame:
    """Load the preprocessed cones.

    If the preprocessed data exists in the current directory, that data is loaded. Otherwise the
    original cone data is loaded, preprocessed, saved, and returned.

    Returns
    -------
    pd.DataFrame
        Preprocessed cone data
    """
    cones_path = pathlib.Path("cones.csv")
    console.rule("[bold yellow]Load cone data", style=Style(color="dark_red"))
    if cones_path.exists():
        console.log(f"{cones_path} already exists in the current directory; skipping.")
        cones = util.read_data(cones_path)
    else:
        cones = preprocess.load_cone_crop(cones_path)
        util.write_data(cones, cones_path)

    return cones


def load_weather() -> pd.DataFrame:
    """Load the preprocessed weather data.

    If the preprocessed data exists in the current directory, that data is loaded. Otherwise the
    original weather data is loaded, preprocessed, saved, and returned.

    Returns
    -------
    pd.DataFrame
        Preprocessed weather data
    """
    weather_path = pathlib.Path("weather.csv")
    console.rule("[bold yellow]Load weather data", style=Style(color="dark_red"))
    if weather_path.exists():
        console.log(
            f"{weather_path} already exists in the current directory; skipping."
        )
        weather = util.read_data(weather_path)
    else:
        weather = preprocess.load_weather(weather_path)
        util.write_data(weather, weather_path)

    return weather


def compute_mean_t(weather: pd.DataFrame) -> pd.DataFrame:
    """Load the mean temperature data.

    If the data exists in the current directory, that data is loaded. Otherwise the
    mean temperature is calculated, saved, and returned.

    Parameters
    ----------
    weather : pd.DataFrame
        Weather data to be used to calculate the mean temperature

    Returns
    -------
    pd.DataFrame
        Mean temperature data
    """
    mean_t_path = pathlib.Path("mean_t.csv")
    console.rule(
        "[bold yellow]Compute the mean temperature", style=Style(color="dark_red")
    )
    if mean_t_path.exists():
        console.log(f"{mean_t_path} already exists in the current directory; skipping.")
        mean_t = util.read_data(mean_t_path)
    else:
        mean_t = analysis.calculate_mean_t(weather)
        util.write_data(mean_t, mean_t_path)

    return mean_t


def compute_correlation(
    cones: pd.DataFrame,
    mean_t: pd.DataFrame,
    groups: list[util.Group],
    output: str | pathlib.Path = "correlation.csv",
) -> pd.DataFrame:
    """Compute the correlation between the ΔT and cone crop data for groups of sites.

    If the data exists in the current directory, that data is loaded. Otherwise ΔT
    data is calculated, correlated with the cone crop, and the correlation coefficient
    is returned.

    Parameters
    ----------
    cones : pd.DataFrame
        Cone crop data
    mean_t : pd.DataFrame
        Mean temperature data
    groups: list[util.Group]
        List of site groups to use when correlating the cone crops to the ΔT data
    output: str | pathlib.Path
        Path to save the resulting correlation data

    Returns
    -------
    pd.DataFrame
        Correlation between ΔT and the number of cones for groups of sites
    """
    correlation_path = pathlib.Path(output)
    console.rule("[bold yellow]Compute the correlation", style=Style(color="dark_red"))
    if correlation_path.exists():
        console.log(
            f"{correlation_path} already exists in the current directory; skipping."
        )
        corr = util.read_data(correlation_path)
    else:
        corr = analysis.correlation(mean_t=mean_t, cones=cones, groups=groups)
        util.write_data(corr, correlation_path)

    return corr


def run_analysis(
    kind: util.CorrelationType = util.CorrelationType.DEFAULT,
    method: str = "pearson",
    cone_number_summand: float = 0,
):
    """Run the pycone analysis for each site separately.

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

    Parameters
    ----------
    kind : util.CorrelationType
        Correlation type to use
    method : str
        Method kwarg to pass pandas.DataFrame.corr
    cone_number_summand : float
        Number to be added to all cones. This prevents the infinities that arise when exp(ΔT/n) is
        calculated when n = 0.
    """
    cones = load_cones()
    weather = load_weather()
    mean_t = compute_mean_t(weather)

    groups = []
    # Build the data structure which controls how correlations are computed
    # Pines have a 3 year reproductive cycle, so treat separately
    for site in mean_t["site"].unique():
        groups.append(
            util.Group(
                name=util.code_to_site(site),
                sites=[site],
                crop_year_gap=util.get_crop_year_gap(site),
                delta_t_year_gap=1,
                kind=kind,
                cone_number_summand=cone_number_summand,
            )
        )

    correlation = compute_correlation(
        mean_t=mean_t,
        cones=cones,
        groups=groups,
        output=f"correlation_{method}_{kind.value}.csv",
    )

    output.plot_correlation_duration_grids(
        correlation,
        groups=groups,
        nrows=18,
        ncols=12,
        figsize=(40, 60),
        extent=(50, 280, 50, 280),
        filename="group_{}_correlations" + f"_{method}_{kind.value}.svg",
    )


def run_batch_analysis(
    kind: util.CorrelationType = util.CorrelationType.DEFAULT,
    method: str = "pearson",
):
    """Run the pycone analysis, grouping the sites with the same species together.

    Parameters
    ----------
    kind : util.CorrelationType
        Correlation type to use
    method : str
        Method kwarg to pass pandas.DataFrame.corr
    """
    cones = load_cones()
    weather = load_weather()
    mean_t = compute_mean_t(weather)

    groups: dict[str, util.Group] = {}
    for site, code in util.SITE_CODES.items():
        # Extract the 4-letter species name from the site code,
        # grouping all sites for the same species together
        species = util.get_species(site)
        if species in groups:
            groups[species].sites.append(code)
        else:
            groups[species] = util.Group(
                name=species,
                sites=[code],
                crop_year_gap=util.get_crop_year_gap(site),
                delta_t_year_gap=1,
                kind=kind,
                method=method,
            )

    correlation = compute_correlation(
        mean_t=mean_t,
        cones=cones,
        groups=list(groups.values()),
        output=f"correlation_{method}_grouped_{kind.value}.csv",
    )

    output.plot_correlation_duration_grids(
        correlation,
        groups=list(groups.values()),
        nrows=18,
        ncols=12,
        figsize=(40, 60),
        extent=(50, 280, 50, 280),
        filename="group_{}_correlations_grouped" + f"_{method}_{kind.value}.svg",
    )


def show_fft():
    """Show the fft of the cone crop data."""
    cones = load_cones()
    weather = load_weather()

    output.plot_fequency(weather, cones)


def run_all_correlation_kinds():
    """Run all types of correlations."""
    for kind in [
        util.CorrelationType.DEFAULT,
        util.CorrelationType.EXP_DT,
        util.CorrelationType.EXP_DT_OVER_N,
    ]:
        for method in ["pearson", "spearman"]:
            run_analysis(kind=kind, method=method)
            run_batch_analysis(kind=kind, method=method)


def run_exp_dt_over_n_correlation_with_offset():
    """Run both correlations, but add 0.5 to all cone counts before the correlation."""
    for method in ["pearson", "spearman"]:
        run_analysis(
            kind=util.CorrelationType.EXP_DT_OVER_N,
            method=method,
            cone_number_summand=0.5,
        )
