import sys
from collections import defaultdict

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import analysis, output, util

# Must import PySide6 above matplotlib (at least for qt_compat, not sure about others)
from PySide6 import QtWidgets  # isort:skip


class ApplicationWindow(QtWidgets.QMainWindow):
    """Explorer application."""

    def __init__(self):
        """Instantiate the application."""
        super().__init__()
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.layout = QtWidgets.QHBoxLayout(self._main)

        self.load_controls()
        self.get_base_data()
        self.get_correlation_data()
        self.populate_controls()

        self.canvases = {
            "weather": FigureCanvas(Figure(figsize=(5, 3))),
            "cones": FigureCanvas(Figure(figsize=(5, 3))),
            "correlation": FigureCanvas(Figure(figsize=(5, 3))),
            "scatter": FigureCanvas(Figure(figsize=(5, 3))),
            "scatter_dt_vs_n": FigureCanvas(Figure(figsize=(5, 3))),
        }
        self.canvases["correlation"].mpl_connect("button_press_event", self.correlation_clicked)
        self.ax = {}

        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        plots_grid = QtWidgets.QGridLayout()
        self.layout.addLayout(plots_grid)
        for i, (name, canvas) in enumerate(self.canvases.items()):
            plot_layout = QtWidgets.QVBoxLayout()
            plot_layout.addWidget(NavigationToolbar(canvas, self))
            plot_layout.addWidget(canvas)
            plots_grid.addLayout(
                plot_layout,
                i // 3,
                i % 3,
            )

            self.ax[name] = canvas.figure.subplots()

        self.group_changed(list(self.correlation_groups)[0])
        self.set_signals_slots()

    def load_controls(self):
        """Load the control panel."""
        controls_widget = QtWidgets.QWidget(self)
        l_input = QtWidgets.QGridLayout()
        self.w_duration = QtWidgets.QComboBox(self._main)
        l_input.addWidget(QtWidgets.QLabel("Duration", self._main), 0, 0)
        l_input.addWidget(self.w_duration, 0, 1)
        self.w_t1 = QtWidgets.QSpinBox(self._main)
        l_input.addWidget(QtWidgets.QLabel("T1", self._main), 1, 0)
        l_input.addWidget(self.w_t1, 1, 1)
        self.w_t2 = QtWidgets.QSpinBox(self._main)
        l_input.addWidget(QtWidgets.QLabel("T2", self._main), 2, 0)
        l_input.addWidget(self.w_t2, 2, 1)
        self.w_group = QtWidgets.QComboBox(self._main)
        l_input.addWidget(QtWidgets.QLabel("Group", self._main), 3, 0)
        l_input.addWidget(self.w_group, 3, 1)
        self.w_method = QtWidgets.QComboBox(self._main)
        l_input.addWidget(QtWidgets.QLabel("Method", self._main), 4, 0)
        l_input.addWidget(self.w_method, 4, 1)
        self.w_kind = QtWidgets.QComboBox(self._main)
        l_input.addWidget(QtWidgets.QLabel("Kind", self._main), 5, 0)
        l_input.addWidget(self.w_kind, 5, 1)
        controls_widget.setLayout(l_input)
        controls_widget.setMaximumWidth(300)
        self.layout.addWidget(controls_widget)

    def set_signals_slots(self):
        """Set up the signals and slots for the control panel."""
        self.w_duration.currentIndexChanged.connect(self.duration_changed)
        self.w_t1.valueChanged.connect(self.t1t2_changed)
        self.w_t2.valueChanged.connect(self.t1t2_changed)
        self.w_group.currentTextChanged.connect(self.group_changed)
        self.w_method.currentTextChanged.connect(self.method_changed)

    def t1t2_changed(self):
        """Slot called when t1 or t2 is changed."""
        self.refresh_weather_mean_t_plot()
        self.refresh_cones_plot()
        self.update_t1t2_lines()
        self.refresh_scatter_plot()

    def populate_controls(self):
        """Populate the QComboBox widgets with values from the data."""
        self.w_group.addItems(self.correlation_groups.keys())
        self.w_method.addItems(["pearson", "spearman"])
        self.w_kind.addItems(["dt_vs_n", "exp_dt_vs_n", "exp_dt_over_n_vs_n"])

    def duration_changed(self, index: int):
        """Slot called when the duration is changed.

        Parameters
        ----------
        index : int
            Index of the newly selected duration
        """
        self.update_t1t2_limits(
            self.w_group.currentText(),
            self.w_duration.itemData(index),
        )
        self.refresh_weather_mean_t_plot()
        self.refresh_cones_plot()
        self.refresh_correlation_plot()
        self.refresh_scatter_plot()
        self.update_t1t2_lines()

    def group_changed(self, group: str):
        """Slot called when the group is changed.

        Parameters
        ----------
        group : str
            The newly selected group (this is the name of the site)
        """
        self.update_duration_items(group)
        self.update_t1t2_limits(group, self.w_duration.currentData())
        self.refresh_weather_mean_t_plot()
        self.refresh_cones_plot()
        self.refresh_correlation_plot()
        self.refresh_scatter_plot()

    def update_t1t2_limits(self, group: str, duration: int):
        """Update the limits of the t1 and t2 QSpinBox widgets.

        Parameters
        ----------
        group : str
            Currently selected group
        duration : int
            Currently selected duration
        """
        current_correlation = self.correlation_gb.get_group((group, duration))

        self.w_t1.setMinimum(current_correlation["start2"].min())
        self.w_t1.setMaximum(current_correlation["start2"].max())
        self.w_t2.setMinimum(current_correlation["start1"].min())
        self.w_t2.setMaximum(current_correlation["start1"].max())

    def update_duration_items(self, group: str):
        """Update the items in the duration QComboBox.

        Parameters
        ----------
        group : str
            Currently selected group
        """
        self.w_duration.disconnect(self.w_duration, None, None, None)
        self.w_duration.clear()
        for duration in self.correlation_groups[group]:
            self.w_duration.addItem(str(duration), duration)
        self.w_duration.setCurrentIndex(0)
        self.w_duration.currentIndexChanged.connect(self.duration_changed)
        self.duration_changed(0)

    def method_changed(self, method: str):
        """Update the plots when the method changes.

        Parameters
        ----------
        method : str
            Currently selected method
        """
        self.get_correlation_data(method)
        self.w_group.setCurrentIndex(0)
        self.group_changed(self.w_group.currentData())

    def refresh_correlation_plot(self):
        """Refresh the heatmap of the correlation data."""
        ax = self.ax["correlation"]
        ax.cla()

        method = self.w_method.currentText()
        group = self.w_group.currentText()
        duration = self.w_duration.currentData()
        if method and group and duration:
            im = output.plot_site_colormap(
                self.correlation_gb.get_group((group, duration)),
                ax=ax,
                xcol="start1",
                ycol="start2",
                data_col="correlation",
                extent=(50, 280, 50, 280),
                vmin=-1,
                vmax=1,
                cmap="BrBG",
                aspect="equal",
            )
            ax.set_xlim(50, 300)
            ax.set_ylim(50, 300)
            ax.set_xticks(
                [60, 91, 121, 152, 182, 213, 244, 274],
                labels=["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
            )
            ax.set_yticks(
                [60, 91, 121, 152, 182, 213, 244, 274],
                labels=["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"],
            )
            ax.set_title(self.w_kind.currentText())
            ax.set_xlabel("T2")
            ax.set_ylabel("T1")

            divider = make_axes_locatable(ax)

            if not hasattr(ax, "cax"):
                cax = divider.append_axes("right", size="5%", pad=0.05)
                ax.figure.colorbar(im, cax=cax, label="Correlation")
                ax.figure.tight_layout()

            self.update_t1t2_lines()

        ax.figure.canvas.draw()

    def update_t1t2_lines(self):
        """Update the crosshair lines on the correlation plot."""
        ax = self.ax["correlation"]
        if len(ax.lines) > 0 and hasattr(ax, "t1_line"):
            ax.t2_line.set_xdata([self.w_t2.value()])
            ax.t1_line.set_ydata([self.w_t1.value()])
        else:
            ax.t2_line = ax.axvline(x=self.w_t2.value(), color="k", alpha=0.5)
            ax.t1_line = ax.axhline(y=self.w_t1.value(), color="k", alpha=0.5)
        ax.figure.canvas.draw()

    def refresh_cones_plot(self):
        """Refresh the cone number vs year plot."""
        ax = self.ax["cones"]
        group = self.w_group.currentText()
        if group:
            site_code = util.site_to_code(group)
            cones = self.cones.loc[(self.cones["site"] == site_code)]

            if ax.has_data():
                ax.line_cones.set_data(cones["year"], cones["cones"])
            else:
                (ax.line_cones,) = ax.plot(cones["year"], cones["cones"], "-b", label="cones")
                ax.set_autoscale_on(True)
                ax.set_xlabel("Year")
                ax.set_ylabel("Cones")
                ax.figure.tight_layout()
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.figure.canvas.draw()

    def refresh_weather_mean_t_plot(self):
        """Refresh the year vs T/mean T plot."""
        ax = self.ax["weather"]
        method = self.w_method.currentText()
        group = self.w_group.currentText()
        duration = self.w_duration.currentData()
        t1 = self.w_t1.value()
        t2 = self.w_t2.value()

        if method and group and duration:
            duration = int(duration)
            site_code = util.site_to_code(group)
            weather = self.weather.loc[self.weather["site"] == site_code]
            mean_t_t1 = self.mean_t.loc[
                (self.mean_t["site"] == site_code)
                & (self.mean_t["duration"] == duration)
                & (self.mean_t["start"] == t1)
            ]
            mean_t_t2 = self.mean_t.loc[
                (self.mean_t["site"] == site_code)
                & (self.mean_t["duration"] == duration)
                & (self.mean_t["start"] == t2)
            ]

            if ax.has_data():
                ax.line_mean_t_t2.set_data(
                    mean_t_t2["days_since_start"],
                    mean_t_t2["mean_t"],
                )
                ax.line_mean_t_t1.set_data(
                    mean_t_t1["days_since_start"],
                    mean_t_t1["mean_t"],
                )
                ax.line_t.set_data(
                    weather["days_since_start"],
                    weather["tmean (degrees f)"],
                )
            else:
                (ax.line_t,) = ax.plot(
                    weather["days_since_start"],
                    weather["tmean (degrees f)"],
                    "-k",
                    label="daily mean temperature",
                )
                (ax.line_mean_t_t2,) = ax.plot(
                    mean_t_t2["days_since_start"],
                    mean_t_t2["mean_t"],
                    "-r",
                    label="mean temperature for interval t2",
                )
                (ax.line_mean_t_t1,) = ax.plot(
                    mean_t_t1["days_since_start"],
                    mean_t_t1["mean_t"],
                    "-b",
                    label="mean temperature for interval t1",
                )
                ax.set_xlabel("Days since start of data")
                ax.set_ylabel("Temperature [°F]")
                ax.figure.tight_layout()
        ax.relim()
        ax.autoscale_view(True, True, True)
        ax.figure.canvas.draw()

    def get_base_data(self):
        """Get the weather, cone, and mean_t data."""
        self.cones = util.read_data("cones.csv")

        weather = util.read_data("weather.csv")
        weather = util.add_days_since_start(weather, doy_col="day_of_year")
        self.weather = weather.sort_values(by=["site", "days_since_start"])

        mean_t = util.read_data("mean_t.csv")
        mean_t = util.add_days_since_start(mean_t, doy_col="start")
        self.mean_t = mean_t.sort_values(by=["site", "duration", "days_since_start"])

    def get_correlation_data(self, method: str = ""):
        """Get the correlation data.

        Parameters
        ----------
        method : str
            Load the correlation data computed with this method
        """
        kind = self.w_kind.currentText()
        if not kind:
            kind = "dt_vs_n"
            self.w_kind.setCurrentText(kind)

        if not method:
            method = "pearson"
            self.w_method.setCurrentText(method)

        correlation = util.read_data(f"correlation_{method}_{kind}.csv")
        self.correlation_gb = correlation.groupby(by=["group", "duration"])
        self.correlation_groups = defaultdict(list)
        for (group, duration), _ in self.correlation_gb:
            self.correlation_groups[group].append(duration)

    def correlation_clicked(self, event: MouseEvent):
        """Handle click events on the correlation plot.

        This will move the selected T1/T2 values to the click location, and update
        the plots.

        Parameters
        ----------
        event : MouseEvent
            Click event
        """
        if event.xdata or event.ydata:
            self.w_t1.disconnect(self.w_t1, None, None, None)
            self.w_t2.disconnect(self.w_t2, None, None, None)
            self.w_t1.setValue(int(event.ydata))
            self.w_t2.setValue(int(event.xdata))
            self.w_t1.valueChanged.connect(self.t1t2_changed)
            self.w_t2.valueChanged.connect(self.t1t2_changed)
            self.t1t2_changed()

    def refresh_scatter_plot(self):
        """Refresh the scatter plots."""
        t1 = self.w_t1.value()
        t2 = self.w_t2.value()
        duration = self.w_duration.currentData()
        group = self.w_group.currentText()

        site = util.site_to_code(group)
        mean_t = self.mean_t.loc[
            (self.mean_t["site"] == site) & (self.mean_t["duration"] == duration)
        ]

        crop_year_gap = util.get_crop_year_gap(site)
        delta_t_year_gap = 1
        years = np.sort(mean_t["year"].unique())[:-delta_t_year_gap]
        dt = analysis.calculate_delta_t_site_duration_fast(
            mean_t,
            years,
            year_gap=delta_t_year_gap,
        )
        dt["crop_year"] = dt["year2"] + crop_year_gap

        cones = self.cones.loc[self.cones["site"] == site]
        dt_cone_df = cones[["year", "cones"]].merge(
            dt,
            how="inner",
            left_on=["year"],
            right_on=["crop_year"],
        )

        df = dt_cone_df.loc[(dt_cone_df["start1"] == t2) & (dt_cone_df["start2"] == t1)]

        ax = self.ax["scatter"]
        ax.cla()
        exp_dt_over_n_plus_half = np.exp(df["delta_t"] / (df["cones"] + 0.5))
        n_plus_half = df["cones"] + 0.5
        ax.line_data = ax.scatter(exp_dt_over_n_plus_half, n_plus_half, c=exp_dt_over_n_plus_half)
        ax.set_xlabel(r"$\mathrm{exp}(\frac{\Delta T}{N + 0.5})$")
        ax.set_ylabel(r"$N + 0.5$")
        ax.set_xlim(exp_dt_over_n_plus_half.min(), exp_dt_over_n_plus_half.max())
        ax.set_ylim(n_plus_half.min(), n_plus_half.max())
        ax.figure.tight_layout()
        ax.figure.canvas.draw()

        ax = self.ax["scatter_dt_vs_n"]
        ax.cla()
        ax.line_data = ax.scatter(df["delta_t"], df["cones"], c=exp_dt_over_n_plus_half)
        ax.set_xlabel(r"$\Delta T$")
        ax.set_ylabel("$N$")
        ax.set_xlim(dt_cone_df["delta_t"].min(), dt_cone_df["delta_t"].max())
        ax.set_ylim(0, dt_cone_df["cones"].max() * 1.05)
        ax.figure.tight_layout()
        ax.figure.canvas.draw()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    app = ApplicationWindow()
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
