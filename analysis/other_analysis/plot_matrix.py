import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd


def get_data() -> dict[pd.DataFrame]:
    dfs_no_na = {}
    dfs_no_na["abam"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:J",
        header=0,
        index_col=0,
        nrows=9,
        sheet_name=0,
    )
    dfs_no_na["abgr"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=10,
        index_col=0,
        nrows=2,
        sheet_name=0,
    )
    dfs_no_na["abla"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=13,
        index_col=0,
        nrows=2,
        sheet_name=0,
    )
    dfs_no_na["abma"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=16,
        index_col=0,
        nrows=2,
        sheet_name=0,
    )
    dfs_no_na["abpr"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:H",
        header=19,
        index_col=0,
        nrows=6,
        sheet_name=0,
    )
    dfs_no_na["pimo"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=26,
        index_col=0,
        nrows=2,
        sheet_name=0,
    )
    dfs_no_na["tsme"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:I",
        header=29,
        index_col=0,
        nrows=7,
        sheet_name=0,
    )

    dfs_with_na = {}
    dfs_with_na["abam"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:M",
        header=0,
        index_col=0,
        nrows=10,
        sheet_name=1,
    )
    dfs_with_na["abco"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=12,
        index_col=0,
        nrows=1,
        sheet_name=1,
    )
    dfs_with_na["abgr"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=15,
        index_col=0,
        nrows=1,
        sheet_name=1,
    )
    dfs_with_na["abla"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:C",
        header=18,
        index_col=0,
        nrows=1,
        sheet_name=1,
    )
    dfs_with_na["abma"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:D",
        header=21,
        index_col=0,
        nrows=2,
        sheet_name=1,
    )
    dfs_with_na["abpr"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:H",
        header=25,
        index_col=0,
        nrows=6,
        sheet_name=1,
    )
    dfs_with_na["pimo"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:D",
        header=33,
        index_col=0,
        nrows=2,
        sheet_name=1,
    )
    dfs_with_na["tsme"] = pd.read_excel(
        "cone_crop_correlation_matrix.xlsx",
        usecols="B:J",
        header=37,
        index_col=0,
        nrows=8,
        sheet_name=1,
    )

    return dfs_no_na, dfs_with_na


def plot_heatmap(dfs: dict[str, pd.DataFrame], key: str, ax: plt.Axes | None = None):
    df = dfs[key]

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.figure

    cmap = plt.cm.Blues
    cmap.set_bad("gray", 0.2)
    im = ax.imshow(
        df.to_numpy(),
        origin="upper",
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    correlation = df.to_numpy()

    textcolors = ("black", "white")
    threshold = 0.5

    for i, j in zip(*np.nonzero(~np.isnan(correlation)), strict=True):
        ax.text(
            j,
            i,
            f"{correlation[i, j]:0.3f}",
            ha="center",
            va="center",
            color=textcolors[int(im.norm(correlation[i, j]) > threshold)],
        )

    ax.set_xlabel("Record ID")
    ax.set_ylabel("Record ID")
    ax.set_xticks(np.arange(len(df.columns)), df.columns)
    ax.set_yticks(np.arange(len(df.index)), df.index)
    ax.set_title(key.upper(), fontweight="bold", fontsize="large")


def plot_heatmaps_no_na(
    dfs: dict[str, pd.DataFrame], mosaic: str, keymap: dict[str, str], suptitle: str
):
    ax = plt.figure(layout="constrained", figsize=(14, 14)).subplot_mosaic(mosaic)

    for axkey, dfkey in keymap.items():
        plot_heatmap(dfs, dfkey, ax[axkey])

        if dfkey in ["tsme", "abpr"]:
            ax[axkey].set_xticks(
                ax[axkey].get_xticks(),
                ax[axkey].get_xticklabels(),
                rotation=45,
                ha="right",
            )

    ax[axkey].figure.suptitle(suptitle)
    divider = make_axes_locatable(ax["C"])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    ax[axkey].figure.colorbar(
        plt.cm.ScalarMappable(norm=None, cmap=plt.cm.Blues),
        cax=cax,
        ax=ax["C"],
        label="Correlation",
        pad=0.02,
    )


def plot_heatmaps_with_na(
    dfs: dict[str, pd.DataFrame], mosaic: str, keymap: dict[str, str], suptitle: str
):
    ax = plt.figure(layout="constrained", figsize=(14, 14)).subplot_mosaic(mosaic)

    for axkey, dfkey in keymap.items():
        plot_heatmap(dfs, dfkey, ax[axkey])

        if dfkey in ["tsme", "abpr"]:
            ax[axkey].set_xticks(
                ax[axkey].get_xticks(),
                ax[axkey].get_xticklabels(),
                rotation=45,
                ha="right",
            )

    ax[axkey].figure.suptitle(suptitle)
    divider = make_axes_locatable(ax["C"])
    cax = divider.append_axes("right", size="5%", pad=0.08)
    ax[axkey].figure.colorbar(
        plt.cm.ScalarMappable(norm=None, cmap=plt.cm.Blues),
        cax=cax,
        ax=ax["C"],
        label="Correlation",
        pad=0.02,
    )


def main():
    dfs_no_na, dfs_with_na = get_data()

    # no_na_mosaic = """
    # AADE
    # AAFG
    # BBCC
    # BBCC
    # """
    # no_na_keymap = {
    #     'A': 'abam',
    #     'B': 'abpr',
    #     'C': 'tsme',
    #     'D': 'abgr',
    #     'E': 'abla',
    #     'F': 'abma',
    #     'G': 'pimo',
    # }
    # plot_heatmaps_no_na(dfs_no_na, no_na_mosaic, no_na_keymap, "No NA")

    with_na_mosaic = """
    AADE.
    AAFGH
    BBCC.
    BBCC.
    """
    with_na_keymap = {
        "A": "abam",
        "B": "abpr",
        "C": "tsme",
        "D": "abgr",
        "E": "abla",
        "F": "abco",
        "G": "pimo",
        "H": "abma",
    }
    plot_heatmaps_with_na(dfs_with_na, with_na_mosaic, with_na_keymap, "With NA")
    plt.show()


if __name__ == "__main__":
    main()
