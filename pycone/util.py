import pathlib

import numpy as np
import pandas as pd
import rich.progress as rp

# Set to be as small as possible to save memory and increase performance
DTYPES = {
    "mean_t": np.single,
    "start": np.short,
    "duration": np.short,
    "site": np.short,
    "year": np.short,
}

SITE_CODES = {
    "10ABAM_OR": 1,
    "11ABAM_OR": 2,
    "12ABGR_OR": 3,
    "13ABMAS_OR": 4,
    "14ABPR_OR": 5,
    "15ABPR_OR": 6,
    "16PIMO3_OR": 7,
    "17TSME_OR": 8,
    "18TSME_OR": 9,
    "19TSME_OR": 10,
    "1ABAM_OR": 11,
    "21PIEN_WA": 12,
    "22ABAM_WA": 13,
    "23ABLA_WA": 14,
    "25TSME_WA": 15,
    "26TSME_WA": 16,
    "27PIMO3_WA": 17,
    "28ABAM_WA": 18,
    "2ABCO_OR": 19,
    "30ABAM_WA": 20,
    "31ABAM_WA": 21,
    "33ABAM_WA": 22,
    "35ABAM_WA": 23,
    "36ABAM_WA": 24,
    "37ABAM_WA": 25,
    "39ABGR_WA": 26,
    "3ABCO_OR": 27,
    "40ABGR_WA": 28,
    "43ABMAS_WA": 29,
    "44ABPR_WA": 30,
    "45ABPR_WA": 31,
    "46ABPR_WA": 32,
    "48ABPR_WA": 33,
    "49ABPR_WA": 34,
    "4ABLA_OR": 35,
    "50ABPR_WA": 36,
    "54PIMO3_WA": 37,
    "55PIMO3_WA": 38,
    "57PIMO3_WA": 39,
    "58TSME_WA": 40,
    "59TSME_WA": 41,
    "5ABMA_OR": 42,
    "60TSME_WA": 43,
    "6ABMA_OR": 44,
    "7PILA_OR": 45,
    "8TSME_OR": 46,
    "9TSME_OR": 47,
}


def write_data(df: pd.DataFrame, path: str | pathlib.Path):
    """Write data to a csv file.

    If it isn't already, site data will be converted to numeric values
    to save space and to make io faster. This saves ~25% of the total
    size of the data just by converting this one column.

    Parameters
    ----------
    df : pd.DataFrame
        Data to be written
    path : str | pathlib.Path
        Path to write the data to
    """
    if df["site"].dtype == "object":
        df = df.replace({"site": SITE_CODES})

    with open(path, "w") as f:
        df.to_csv(f, index=False)


def read_data(path: str | pathlib.Path) -> pd.DataFrame:
    """Read data from a file.

    Importantly, this function sets the appropriate dtypes for the columns
    to ensure best performance.

    Parameters
    ----------
    path : str | pathlib.Path
        File to be read

    Returns
    -------
    pd.DataFrame
        Data read from the file
    """
    with rp.open(path, "rb") as f:
        return pd.read_csv(f, dtype=DTYPES)