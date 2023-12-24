import pathlib

import pandas as pd
import rich.progress as rp


def load_data(
    cones_fname: pathlib.Path | str,
    weather_fname: pathlib.Path | str,
) -> pd.DataFrame:
    if isinstance(cones_fname, str):
        cones_fname = pathlib.Path(cones_fname)
    if isinstance(weather_fname, str):
        weather_fname = pathlib.Path(weather_fname)

    return load_cone_crop(cones_fname), load_weather(weather_fname)


def load_cone_crop(fname: pathlib.Path) -> pd.DataFrame:
    return pd.read_excel(
        fname,
        na_filter=False,
    )


def load_weather(fname: pathlib.Path) -> pd.DataFrame:
    """Load the daily weather data spreadsheet into a DataFrame.

    The file is expected to contain multiple sheets, with each sheet corresponding to a different
    year's data. Empty rows are dropped; dates are expected to be in ``mm/dd/yyyy`` format.

    Parameters
    ----------
    fname : pathlib.Path
        Path to the spreadsheed to be loaded.

    Returns
    -------
    pd.DataFrame
        Concatenated daily weather data.
    """
    file = pd.ExcelFile(fname)

    sheets = {}
    for sheet in rp.track(file.sheet_names, description="Loading weather data..."):
        sheets[sheet] = file.parse(
            sheet_name=sheet,
            parse_dates=[4],
            date_format="mm/dd/yyyy",
            skiprows=10,
        )

    return pd.concat(sheets.values(), ignore_index=True).dropna(
        axis=0, how="all", ignore_index=True
    )
