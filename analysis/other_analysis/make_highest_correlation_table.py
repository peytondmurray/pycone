import pathlib
import sys

import pandas as pd


def get_max_correlation_dt_vs_n(path: str, min_duration: int | None = None) -> pd.DataFrame:
    """Generate a max correlation table for the dt vs n model.

    Edges are excluded due to artifacts arising from how data is
    collected on start days from leap years. The last starting
    day on each year is removed before calculating the max
    correlation

    Parameters
    ----------
    path : str
        Path to the correlation data csv
    min_duration : int | None
        Minimum duration to consider

    Returns
    -------
    pd.DataFrame
        Maximum correlation stats
    """
    result = {}
    variable = "dt_vs_n"
    kind = "pearson"

    fname = pathlib.Path(path) / f"correlation_{kind}_{variable}.csv"
    data = pd.read_csv(fname)

    if min_duration is not None:
        data = data.loc[data["duration"] >= min_duration]

    # The edge is _different_ for each duration, so you can't just
    # snip off 1 day from the max across _all_ durations...
    for (group, _duration), group_df in data.groupby(["group", "duration"]):
        correlation_col = group_df.loc[
            (group_df["start1"] < group_df["start1"].max() - 1)
            & (group_df["start2"] < group_df["start2"].max() - 1)
        ]["correlation"]

        if len(correlation_col) == 0 or correlation_col.isna().all():
            continue

        max_correlation = group_df.loc[correlation_col.idxmax()]

        if (group not in result) or (
            group in result and result[group]["correlation"] < max_correlation["correlation"]
        ):
            result[group] = max_correlation
            result[group]["kind"] = kind
            result[group]["variable"] = variable

    return pd.concat(result, axis=1).T.drop(columns=["group"])


def get_max_correlation(path: str) -> pd.DataFrame:
    result = {}
    for kind in ["pearson", "spearman"]:
        for variable in ["dt_vs_n", "exp_dt_vs_n", "exp_dt_over_n_vs_n"]:
            print(f"{kind} - {variable}")
            fname = pathlib.Path(path) / f"correlation_{kind}_{variable}.csv"
            data = pd.read_csv(fname)

            max_correlation = data.loc[data.groupby("group")["correlation"].idxmax()]

            for _, row in max_correlation.iterrows():
                group = row["group"]

                if (group not in result) or (
                    group in result and result[group]["correlation"] < row["correlation"]
                ):
                    result[group] = row
                    result[group]["kind"] = kind
                    result[group]["variable"] = variable

    return pd.concat(result, axis=1).T.drop(columns=["group"])


if __name__ == "__main__":
    result = get_max_correlation_dt_vs_n(sys.argv[1], int(sys.argv[2]))
    result.to_csv(f"max_correlation_by_site_dt_vs_n_duration={sys.argv[2]}_noedge.csv")

    # result = get_max_correlation(sys.argv[1])
    # result.to_csv("max_correlation_by_site.csv")
