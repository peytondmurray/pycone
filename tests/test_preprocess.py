import pathlib

from pycone import preprocess


def test_load_weather():
    data = (
        pathlib.Path(__file__).parent.parent
        / "data"
        / "raw_data"
        / "daily_weather_1981-2014.xlsx"
    )
    preprocess.load_weather(data)
