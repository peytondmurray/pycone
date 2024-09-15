import matplotlib.pyplot as plt
import pandas as pd


def get_data() -> dict[pd.DataFrame]:
    dfs = {}
    dfs["abam"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:J",
        header=0,
        index_col=0,
        nrows=9,
    )

    dfs["abgr"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=10,
        index_col=0,
        nrows=2,
    )

    dfs["abla"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=13,
        index_col=0,
        nrows=2,
    )

    dfs["abma"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=16,
        index_col=0,
        nrows=2,
    )

    dfs["abpr"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:H",
        header=19,
        index_col=0,
        nrows=6,
    )

    dfs["pimo"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=26,
        index_col=0,
        nrows=2,
    )

    dfs["tsme"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:I",
        header=29,
        index_col=0,
        nrows=7,
    )

    return dfs


def main():
    dfs = get_data()
    breakpoint()
    print(dfs)


if __name__ == "__main__":
    main()
