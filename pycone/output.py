from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import util


def plot_correlation_duration_grids(
    data: pd.DataFrame,
    nrows: int = 18,
    ncols: int = 12,
    figsize: tuple[int, int] | None = None,
    extent: tuple[int, int, int, int] | None = None,
    filename: str | None = "site_{}_correlations.svg",
) -> list[plt.Figure] | None:
    """Generate a grid of correlation plots.

    Each plot is a 2D colormap of the correlation as a function of the start date of the interval in
    the first year $i$, and the start date of the interval in the second year $j$.

    Parameters
    ----------
    data : pd.DataFrame
        Correlation data; must have columns

            start1
            start2
            site
            duration
            correlation

    nrows : int
        Number of rows of 2D colormaps to plot on a single figure
    ncols : int
        Number of columns of 2D colormaps to plot on a single figure
    figsize : tuple[int, int] | None
        Size of each figure
    extent : tuple[int, int, int, int] | None
        Extent [xmin, xmax, ymin, ymax] of the data to show in each colormap
    filename : str | None
        Filename template to use for writing figures to disk. If None, a list of figures are
        returned. Otherwise, this must be a format string to save individual figures (one figure for
        each site)

    Returns
    -------
    list[plt.Figure | None]
        A list of figures (if `filename == None`), or None if figures are to be written to disk
    """
    gb = data.groupby(by="site")
    sites, site_dfs = tuple(zip(*tuple(gb), strict=True))

    with util.ParallelExecutor("Generating correlation plots...", processes=8) as pe:
        for site, site_df in zip(sites, site_dfs, strict=True):
            task_id = pe.add_task(f"Site: {site}", visible=False)
            pe.apply_async(
                plot_correlation_duration_grid,
                (site_df, site),
                {
                    "nrows": nrows,
                    "ncols": ncols,
                    "figsize": figsize,
                    "extent": extent,
                    "filename": filename.format(site) if filename is not None else None,
                    "task_id": task_id,
                    "worker_status": pe.worker_status,
                },
            )

        results = pe.wait_for_results()

    if filename is None:
        return results
    return None


def plot_correlation_duration_grid(
    site_df: pd.DataFrame,
    site: int,
    nrows: int = 18,
    ncols: int = 12,
    figsize: tuple[int, int] | None = None,
    extent: tuple[int, int, int, int] | None = None,
    filename: str | None = None,
    task_id: int | None = None,
    worker_status: dict[int, Any] | None = None,
) -> plt.Figure | None:
    """Generate a single correlation/duration plot for all durations in the given dataset.

    Parameters
    ----------
    site_df : pd.DataFrame
        Correlation data for a single site; must have columns

            start1
            start2
            site
            duration
            correlation

    site : int
        Site from which the given `site_df` data was computed
    nrows : int
        Number of rows of 2D colormaps to plot on a single figure
    ncols : int
        Number of columns of 2D colormaps to plot on a single figure
    figsize : tuple[int, int] | None
        Size of each figure
    extent : tuple[int, int, int, int] | None
        Extent [xmin, xmax, ymin, ymax] of the data to show in each colormap
    filename : str | None
        Filename template to use for writing figures to disk. If None, a list of figures are
        returned. Otherwise, this must be a format string to save individual figures (one figure for
        each site)
    task_id : int
        Task ID returned by ``rich.progress.Progress.add_task``, used for reporting progress to the
        main process
    worker_status : dict[int, Any]
        Dictionary where worker status information can be written. This is a multiprocessing-safe
        object shared across all workers.

    Returns
    -------
    plt.Figure | None
        Figure containing the colormaps (if `filename == None`), or None if figure is written to
        disk
    """
    if figsize is None:
        figsize = (40, 60)

    durations = np.sort(site_df["duration"].unique())

    # Largest duration is 213, but 216 has a pair of nice divisors. 3 empty plots left over :)
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        layout="constrained",
        sharex="all",
        sharey="all",
        gridspec_kw={"wspace": 0.02, "hspace": 0.02},
    )

    if extent is None:
        # Only days between 60 and 273 have data; adjust plot boundaries accordingly
        extent = (50, 280, 50, 280)

    is_subprocess = worker_status is not None and task_id is not None
    if is_subprocess:
        worker_status[task_id] = {"items_completed": 0, "total": len(durations)}  # type: ignore

    for i, duration in enumerate(durations):
        ax_row = i // ncols
        ax_col = i % ncols

        axis = ax[ax_row, ax_col]

        im = plot_site_colormap(
            site_df.loc[site_df["duration"] == duration],
            ax=axis,
            xcol="start1",
            ycol="start2",
            data_col="correlation",
            extent=extent,
            vmin=-1,
            vmax=1,
            cmap="BrBG",
            aspect="equal",
        )
        axis.text(
            0.75, 0.05, f"$d$ = {duration}", transform=ax[ax_row, ax_col].transAxes
        )

        if ax_row == nrows - 1 and ax_col == ncols // 2:
            axis.set_xlabel("Start of interval for year $i$", fontsize="x-large")

        if ax_col == 0 and ax_row == nrows // 2:
            axis.set_ylabel("Start of interval for year $j$", fontsize="x-large")

        if is_subprocess:
            worker_status[task_id] = {"items_completed": i + 1, "total": len(durations)}  # type: ignore

    # Turn off any of the extra axes that don't have data plotted in them
    for j in range(i + 1, nrows * ncols):
        ax[j // ncols, j % ncols].set_axis_off()

    # Add an axis for colorbars in the unused space in the bottom right of the figure
    cax = fig.add_axes([0.85, 0.03, 0.1, 0.005])
    fig.colorbar(
        im,
        cax=cax,
        label="Correlation " + r"$\rho_{\Delta T_{ij} N_k}$",
        orientation="horizontal",
    )

    fig.suptitle(
        r"$\rho_{\Delta T_{ij} N_k}$"
        + f" Site: {util.code_to_site(site)}, $k = j+1 = i+2$",
        fontsize="xx-large",
    )

    if filename:
        fig.savefig(filename)
        return None
    return fig


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
