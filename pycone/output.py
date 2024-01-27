import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import util


def export_all_site_interval_colormaps(
    df: pd.DataFrame, data_col: str, extension: str = "pdf"
):
    """Export all site interval colormaps to pdfs.

    Parameters
    ----------
    df : pd.DataFrame
        Data for a single site
    data_col : str
        Data column to plot
    extension : str
        Filetype extension that the plots should be saved as
    """
    if len(df["site"].unique()) > 1:
        raise ValueError("Cannot plot site interval data with multiple sites.")

    site = df["site"][0]
    if isinstance(site, int):
        site = util.code_to_site(site)

    for duration in df["duration"].unique():
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_site_colormap(
            df.loc[df["duration"] == duration],
            ax=ax,
            xcol="start1",
            ycol="start2",
            data_col=data_col,
            extent=(0, 365, 0, 365),
        )
        fig.savefig(f"{data_col}_site_{site}_duration_{duration}.{extension}")


def plot_site_colormap(
    df: pd.DataFrame,
    ax: plt.Axes,
    xcol: str,
    ycol: str,
    data_col: str,
    extent: tuple[int, int, int, int],
    title: bool | str = False,
    xlabel: bool | str = False,
    ylabel: bool | str = False,
    colorbar: bool | str = False,
    cax: plt.Axes | None = None,
    cmap: str | Colormap = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
):
    """Plot data for a given site as a color map.

    X-axis is interval 1 start day of year,
    Y-axis is interval 2 start day of year,
    Z-axis is a colormap of the given `data_col`.

    Parameters
    ----------
    df : pd.DataFrame
        Data for the site to be plotted. Must contain only one interval duration
    ax : plt.Axes
        Axes on which the data is to be plotted
    xcol : str
        X column name
    ycol : str
        Y column name
    data_col : str
        Data column to plot as the colormap
    extent : tuple(int, int, int, int)
        Extent of the x and y axes
    title : bool | str
        If specified, add the title to the plot. If True a title is automatically generated,
        if False, no title is added.
    xlabel : bool | str
        If specified, add the title to the plot. If True a label is automatically generated,
        if False, no label is added.
    ylabel : bool | str
        If specified, add the title to the plot. If True a label is automatically generated,
        if False, no label is added.
    cax : plt.Axes | None
        Axes to use to plot the colorbar; if None, a new set of axes is added to the figure and
        used
    colorbar : bool | str
        If specified, add the title to the plot. If True a colorbar with label is automatically
        generated a title is automatically generated from the `data_col`; if False, no colorbar
        with label is added.
    cmap : str | Colormap
        Colormap to use to plot the data
    vmin : float | None
        Min value of the colormap; if None, the min value of the dataset is used
    vmax : float | None
        Max value of the colormap; if None, the max value of the dataset is used
    **kwargs
        Other arguments are passed to imshow

    Returns
    -------
    plt.AxesImage
        Image of the colormap data
    """
    zz = util.make_pixel_map(
        df[xcol],
        df[ycol],
        df[data_col],
        extent=extent,
    )

    im = ax.imshow(
        zz,
        interpolation="none",
        origin="lower",
        extent=extent,
        vmin=df[data_col].min() if vmin is None else vmin,
        vmax=df[data_col].max() if vmax is None else vmax,
        cmap=cmap,
        **kwargs,
    )

    site = df.iloc[0]["site"]
    if isinstance(site, int):
        site = util.code_to_site(site)

    if title:
        ax.set_title(
            title
            if isinstance(title, str)
            else f"Site {site}, duration {df.iloc[0]['duration']}"
        )
    if xlabel:
        ax.set_xlabel(xlabel if isinstance(xlabel, str) else xcol)
    if ylabel:
        ax.set_ylabel(ylabel if isinstance(ylabel, str) else ycol)
    if colorbar:
        if not cax:
            # Divide the current axes and create a new axis for the colorbar.
            # This ensures that the new colorbar is the same height as the existing axes.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

        ax.get_figure().colorbar(
            im,
            cax=cax,
            label=colorbar if isinstance(colorbar, str) else data_col,
        )
    return im
