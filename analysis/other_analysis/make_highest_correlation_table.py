import pathlib
import sys

import pandas as pd


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
    result = get_max_correlation(sys.argv[1])
    result.to_csv("max_correlation_by_site.csv")
