import pathlib
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pycone import preprocess


@pytest.fixture
def raw_data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "data" / "raw_data"


@mock.patch("pandas.ExcelFile")
def test_load_weather(mock_load_excel_file):
    """Test that weather data can be loaded and parsed."""
    mock_file = mock.MagicMock()
    mock_file.sheet_names = ["2014"]
    mock_file.parse.return_value = pd.DataFrame(
        {
            "Name": ["4ABLA_OR", "4ABLA_OR", "4ABLA_OR", "4ABLA_OR", np.nan],
            "Longitude": [-121.9217, -121.9217, -121.9217, -121.9217, np.nan],
            "Latitude": [44.3833, 44.3833, 44.3833, 44.3833, np.nan],
            "Elevation (ft)": [4656, 4656, 4656, 4656, np.nan],
            "Date": ["03/01/2014", "03/02/2014", "03/03/2014", "03/04/2014", pd.NaT],
            "tmean (degrees F)": [34.5, 32.9, 37.3, 37.1, np.nan],
        }
    )

    mock_load_excel_file.return_value = mock_file
    result = preprocess.load_weather()

    # Check that no nan-valued rows made it into the final dataset
    assert result.loc[result["date"].isna()].empty


@mock.patch("pandas.read_excel")
def test_load_cone_crop(mock_read_excel):
    mock_read_excel.return_value = pd.DataFrame(
        {
            "Code ": ["1ABAM_OR", "2ABCO_OR", "3ABCO_OR", "4ABLA_OR"],
            "REF": [
                "SCHULZE & FRANKLIN 2018; FRANKLIN ET AL 1974",
                "SCHULZE & FRANKLIN 2019",
                "SCHULZE & FRANKLIN 2019",
                "SCHULZE & FRANKLIN 2019",
            ],
            "SPECIES": [
                "ABIES AMABILIS",
                "ABIES CONCOLOR",
                "ABIES CONCOLOR",
                "ABIES LASIOCARPA",
            ],
            "GENUS": ["ABIES", "ABIES", "ABIES", "ABIES"],
            "FAMILY": ["PINACEAE", "PINACEAE", "PINACEAE", "PINACEAE"],
            "NRCS_GENUS": ["ABIES", "ABIES", "ABIES", "ABIES"],
            "NRCS_SPECIES": ["ABAM", "ABCO", "ABCO", "ABLA"],
            "SITE": ["SANTIAM PASS", "ASHLAND RNA", "WICKIUP SPRINGS", "SAND MOUNTAIN"],
            "COUNTRY": ["USA", "USA", "USA", "USA"],
            "STATE_PROVINCE": ["OR", "OR", "OR", "OR"],
            "LATITUDE": [44.39085, 42.10711, 42.6058, 44.383333],
            "LONGITUDE": [-121.8494, -122.69266, -122.2966, -121.921667],
            "ELEVATION_m": [1446.1, 1400.0, 1479.0, 1421.5],
            "STRTYR": [1962, 1981, 1981, 1959],
            "ENDYR": [2002, 2017, 2017, 2019],
            "Nyears": [41, 34, 33, 54],
            "COMMENTS": [
                "UPDATED FROM FRANKLIN 1968; UPDATED FROM FRANKLIN 1974",
                "Andrews Forest LTER, initiated by Franklin in 64",
                "Andrews Forest LTER, initiated by Franklin in 68",
                "LARGER LTER DATASET REPLACES WOODWARD ET AL. 1994 DATA",
            ],
            "Y1980": [6, "NA", "NA", 0],
            "Y1981": [0, 35, 22, 1],
            "Y1982": [21, 3, 0, 26],
            "Y1983": [0, 0, 0, 0],
            "Y1984": [0, 12, 0, 0],
            "Y2013": ["NA", 0, 0, 0],
            "Y2014": ["NA", 60, 0, 6],
        }
    )

    df = preprocess.load_cone_crop()

    # The number of rows should be 4 sites * 7 years for the mock data above
    assert len(df) == 28


@pytest.mark.parametrize(
    "name,expected",
    (
        ["y2012", 2012],
        ["y10211", None],
        ["foo", None],
        [" y1921", None],
        ["y1921 ", None],
        ["yy1921", None],
    ),
)
def test_parse_year_column_name(name, expected):
    """Test that the year column name parser works as expected for various input."""
    parsed = preprocess.extract_year_from_column_name(name)
    if expected is None:
        assert parsed is None
    else:
        assert parsed == expected
