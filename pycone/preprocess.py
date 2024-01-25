import pathlib
import re
import warnings

import pandas as pd
import rich.progress as rp


def load_data(
    cones_fname: pathlib.Path | str | None = None,
    weather_fname: pathlib.Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the cone count and weather data, clean the data, and return the respective DataFrames.

    Parameters
    ----------
    cones_fname : pathlib.Path | str | None
        Name of the cone data to load.
    weather_fname : pathlib.Path | str | None
        Name of the weather data to load.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (cone crop data, weather data) as preprocessed DataFrames.
    """
    if isinstance(cones_fname, str):
        cones_fname = pathlib.Path(cones_fname)
    if isinstance(weather_fname, str):
        weather_fname = pathlib.Path(weather_fname)

    cones = load_cone_crop(cones_fname)
    weather = load_weather(weather_fname)

    return cones, weather


def extract_year_from_column_name(col: str) -> int | None:
    """Extract the year part of a cone crop column name, if possible.

    A warning is issued if the year extracted from a column fall outside the range 1980-2014.

    Parameters
    ----------
    col : str
        Column name; if it is of the form ``y1234``, the integer year is returned; otherwise None is
        returned.

    Returns
    -------
    int | None
        Year part of the column name, if applicable, otherwise None.
    """
    match = re.match(r"^y(?P<year>\d{4})$", col)
    if match:
        year = int(match.group("year"))
        if year < 1980 or year > 2014:
            warnings.warn(
                f"Unexpected year {year} extracted from column. Double check the data by hand.",
                stacklevel=2,
            )
        return year
    return None


def load_cone_crop(fname: pathlib.Path | None = None) -> pd.DataFrame:
    """Load the cone crop data into a DataFrame.

    Column names are stripped of leading and trailing whitespaces, then lowercased.

    Parameters
    ----------
    fname : pathlib.Path | None
        Path to the cone crop data spreadsheet. If unspecified, uses
        ``data/cone_crop.xlsx``.

    Returns
    -------
    pd.DataFrame
        Cleaned cone crop data. Each row is a separate cone count for a given species, site, and
        year.
    """
    if fname is None:
        fname = pathlib.Path(__file__).parent / "data" / "cone_crop.xlsx"

    df = pd.read_excel(
        fname,
        na_filter=False,
    )

    # Create a mapping from old column names to new ones. Keep a list of the year columns separately
    # from the other columns, so that we can pivot the table later on.
    year_columns, other_columns = [], []
    new_columns = {"code": "site"}
    for column in df.columns:
        col = column.lstrip().rstrip().lower()
        year = extract_year_from_column_name(col)
        if year is None:
            new_columns[column] = col
            other_columns.append(col)
        else:
            new_columns[column] = str(year)
            year_columns.append(str(year))

    # Pivot the dataframe so that each value in the year columns becomes a new row in the output
    # dataframe.
    return df.rename(columns=new_columns).melt(
        id_vars=other_columns,
        value_vars=year_columns,
        var_name="year",
        value_name="cones",
    )


def load_weather(fname: pathlib.Path | None = None) -> pd.DataFrame:
    """Load the daily weather data spreadsheet into a DataFrame.

    The file is expected to contain multiple sheets, with each sheet corresponding to a different
    year's data. Empty rows are dropped; dates are expected to be in ``mm/dd/yyyy`` format.

    Column names are stripped of leading and trailing whitespaces, then lowercased.

    The 'name' column is renamed to 'site' for clarity.

    Parameters
    ----------
    fname : pathlib.Path
        Path to the spreadsheet to be loaded. If unspecified, uses
        ``data/daily_weather_1981-2014.xlsx``.

    Returns
    -------
    pd.DataFrame
        Cleaned and concatenated daily weather data. Each row is a separate temperature
        measurement at a site on a given date.
    """
    if fname is None:
        fname = (
            pathlib.Path(__file__).parent.parent
            / "data"
            / "daily_weather_1981-2014.xlsx"
        )
    file = pd.ExcelFile(fname)

    sheets = {}
    for sheet in rp.track(file.sheet_names, description="Loading weather data..."):
        sheets[sheet] = file.parse(
            sheet_name=sheet,
            skiprows=10,
        )

    df = (
        pd.concat(sheets.values(), ignore_index=True)
        .dropna(axis=0, how="all", ignore_index=True)
        .rename(columns=lambda col: col.lstrip().rstrip().lower())
        .rename(columns={"name": "site"})
    )

    # Convert the date column from object dtype to a pandas datetime object;
    # Extract the year as a separate column.
    df["date"] = pd.to_datetime(
        df["date"],
        format="%m/%d/%Y",
    )
    df["year"] = df["date"].apply(lambda date: date.year)

    # Compute the day number of the given year
    df["day_of_year"] = df["date"].apply(lambda x: x.timetuple().tm_yday)
    return df
