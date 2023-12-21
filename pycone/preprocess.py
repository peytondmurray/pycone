import pathlib

import pandas as pd


def load_data(
    cones_fname: pathlib.Path | str,
    weather_fname: pathlib.Path | str,
) -> pd.DataFrame:
    if isinstance(cones_fname, str):
        cones_fname = pathlib.Path(cones_fname)
    if isinstance(weather_fname, str):
        cones_fname = pathlib.Path(cones_fname)

    return load_cone_crop(cones_fname), load_weather(weather_fname)


def load_cone_crop(fname: pathlib.Path) -> pd.DataFrame:
    pass


def load_weather(fname: pathlib.Path) -> pd.DataFrame:
    pass
