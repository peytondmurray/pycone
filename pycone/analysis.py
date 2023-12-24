import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("QtAgg")


def correlate(
    data: pd.DataFrame,
    window: datetime.timedelta,
    offset: datetime.timedelta,
    onset: datetime.datetime,
) -> np.ndarray:
    pass
