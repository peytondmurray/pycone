import matplotlib.pyplot as plt
import numpy as np
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


def plot_heatmap(dfs: dict[str, pd.DataFrame], key: str):
    df = dfs[key]

    fig, ax = plt.subplots(1, 1)
    cmap = plt.cm.Blues
    cmap.set_bad("gray", 0.2)
    im = ax.imshow(
        df.to_numpy(),
        origin="upper",
        aspect="equal",
        cmap=cmap,
    )

    correlation = df.to_numpy()

    for i, j in zip(*np.nonzero(~np.isnan(correlation)), strict=True):
        ax.text(
            j,
            i,
            f"{correlation[i, j]:0.2f}",
            ha="center",
            va="center",
            color="w",
        )

    fig.colorbar(im, ax=ax, label="Correlation", pad=0.02)

    ax.set_xlabel("Record ID")
    ax.set_ylabel("Record ID")
    ax.set_xticks(np.arange(len(df.columns)), df.columns)
    ax.set_yticks(np.arange(len(df.index)), df.index)
    ax.set_title(key.upper())


def main():
    dfs = get_data()
    plot_heatmap(dfs, "abam")
    plt.show()


if __name__ == "__main__":
    main()
