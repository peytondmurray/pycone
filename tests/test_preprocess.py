import pathlib

import pytest

from pycone import preprocess


@pytest.fixture
def raw_data_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent / "data" / "raw_data"


def test_load_weather(raw_data_path):
    preprocess.load_weather(raw_data_path / "daily_weather_1981-2014.xlsx")


def test_load_cone_crop(raw_data_path):
    df = preprocess.load_cone_crop(raw_data_path / "cone_crop.xlsx")
    breakpoint()
    print(df)


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
