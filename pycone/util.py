import datetime

import pandas as pd


class IntervalPair:
    """Single data structure representing two intervals to be compared.

    Parameters
    ----------
    start1 : datetime.datetime | str
        Start of the first interval; if this is a str, it is assumed to be in ISO format
    start2 : datetime.datetime | str
        Start of the second interval; if this is a str, it is assumed to be in ISO format
    length : datetime.timedelta | int
        Length of the intervals; if this is an int, it specifies the number of days in the interval
    """

    def __init__(
        self,
        start1: datetime.datetime | str,
        start2: datetime.datetime | str,
        length: datetime.timedelta | int,
    ):
        if isinstance(start1, str):
            start1 = datetime.datetime.fromisoformat(start1)
        if isinstance(start2, str):
            start2 = datetime.datetime.fromisoformat(start2)
        if isinstance(length, int):
            length = datetime.timedelta(days=length)

        self.start1 = start1
        self.start2 = start2
        self.length = length

    def __len__(self) -> datetime.timedelta:
        return self.length

    def __repr__(self) -> str:
        return f"""
        Interval 1: {self.start1.isoformat()} - {(self.start1 + self.length).isoformat()}
        Interval 2: {self.start2.isoformat()} - {(self.start2 + self.length).isoformat()}
        """

    def get_interval_masks(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Get the boolean masks containing dates that fall in the given intervals.

        Parameters
        ----------
        series : pd.Series
            Dates for which the intervals are to be applied to generate masks.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Tuple of boolean masks giving the dates in the first interval, and the dates in the
            second interval.
        """
        return (
            (self.start1 <= series) & (series <= (self.start1 + self.length)),
            (self.start2 <= series) & (series <= (self.start2 + self.length)),
        )

    def to_duration_offset_onset(
        self,
    ) -> tuple[datetime.timedelta, datetime.timedelta, datetime.timedelta]:
        """Calculate the duration, offset, and onset of the intervals.

            Duration: length of interval [days]
            Offset: Number of days between the start of the two intervals
            Onset: Number of days between March 10 and the start of the interval

        Not sure why onset is referenced to March 10; taken from R code.

        Returns
        -------
        tuple[datetime.timedelta, datetime.timedelta, datetime.timedelta]
            Duration, offset, and onset of the intervals
        """
        return (
            self.length.days,
            (self.start2 - self.start1).days % 365,
            (self.start1 - datetime.datetime(self.start1.year, 3, 10)).days,
        )
