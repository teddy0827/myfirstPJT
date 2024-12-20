import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from plotting.plot_overlay import plot_overlay
import numpy as np

class EnlargedPlotWindow(QMainWindow):
    def __init__(self, x, y, u, v, title, v_lines, h_lines):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 800)

        fig, ax = plt.subplots(figsize=(8, 8))
        plot_overlay(ax, x, y, u, v, v_lines, h_lines, title=title)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)


class PlotWindow(QMainWindow):
    def __init__(self, unique_id, df, vertical_lines, horizontal_lines):
        super().__init__()
        self.setWindowTitle(f"Plot for {unique_id}")
        self.setGeometry(100, 100, 1200, 800)
        self.df = df
        self.unique_id = unique_id
        self.vertical_lines = vertical_lines
        self.horizontal_lines = horizontal_lines
        self.enlarged_plot_windows = []
        self.cid = None
        self.init_ui()

    def init_ui(self):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        region_label = QLabel("Select Region:")
        self.region_combo = QComboBox()
        self.region_combo.addItems(['All', 'Center', 'Middle', 'Edge'])
        self.region_combo.setCurrentIndex(0)
        self.region_combo.currentIndexChanged.connect(self.update_plot)

        region_layout = QHBoxLayout()
        region_layout.addWidget(region_label)
        region_layout.addWidget(self.region_combo)

        self.fig, self.axes = plt.subplots(4, 3, figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        main_layout.addLayout(region_layout)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.update_plot()

    def update_plot(self):
        df_lot = self.df[self.df['UNIQUE_ID'] == self.unique_id].copy()

        # 반지름 및 영역 계산
        df_lot['radius'] = np.sqrt(df_lot['wf_x']**2 + df_lot['wf_y']**2)
        conditions = [
            (df_lot['radius'] <= 50000),
            (df_lot['radius'] > 50000) & (df_lot['radius'] <= 100000),
            (df_lot['radius'] > 100000) & (df_lot['radius'] <= 150000)
        ]
        choices = ['Center', 'Middle', 'Edge']
        df_lot['region'] = np.select(conditions, choices, default='Outside')

        selected_region = self.region_combo.currentText()
        data = df_lot if selected_region == 'All' else df_lot[df_lot['region'] == selected_region]

        wf_x = data['wf_x'].tolist()
        wf_y = data['wf_y'].tolist()

        variables = ['X_reg', 'Y_reg', 'pred_x', 'pred_y', 'residual_x', 'residual_y',
                     'psm_fit_x', 'psm_fit_y', 'residual_x_depsm', 'residual_y_depsm',
                     'cpe19p_pred_x', 'cpe19p_pred_y', 'cpe19p_resi_x', 'cpe19p_resi_y',
                     'ideal_psm_x', 'ideal_psm_y', 'delta_psm_x', 'delta_psm_y']
        data_vars = {var: data[var].tolist() for var in variables}

        self.fig.clear()
        axes = self.fig.subplots(4, 3)
        self.fig.suptitle(f'Visualizations for Lot {self.unique_id} - {selected_region}', fontsize=16)

        plot_configs = [
            {'ax': axes[0, 0], 'dx': data_vars['X_reg'], 'dy': data_vars['Y_reg'], 'title': 'Raw(X_reg,Y_reg)'},
            {'ax': axes[0, 1], 'dx': data_vars['pred_x'], 'dy': data_vars['pred_y'], 'title': 'OSR_Fitting(WK,RK)'},
            {'ax': axes[0, 2], 'dx': data_vars['residual_x'], 'dy': data_vars['residual_y'], 'title': 'Residual'},
            {'ax': axes[1, 0], 'dx': data_vars['psm_fit_x'], 'dy': data_vars['psm_fit_y'], 'title': 'PSM Input'},
            {'ax': axes[1, 1], 'dx': data_vars['residual_x_depsm'], 'dy': data_vars['residual_y_depsm'], 'title': 'Residual(Remove_PSM)'},
            {'ax': axes[1, 2], 'dx': data_vars['cpe19p_pred_x'], 'dy': data_vars['cpe19p_pred_y'], 'title': 'CPE 19para Fitting'},
            {'ax': axes[2, 0], 'dx': data_vars['cpe19p_resi_x'], 'dy': data_vars['cpe19p_resi_y'], 'title': 'CPE 19para Residual'},
            {'ax': axes[2, 1], 'dx': data_vars['ideal_psm_x'], 'dy': data_vars['ideal_psm_y'], 'title': 'Ideal PSM'},
            {'ax': axes[2, 2], 'dx': data_vars['delta_psm_x'], 'dy': data_vars['delta_psm_y'], 'title': 'Delta PSM'},
            {'ax': axes[3, 0], 'dx': data_vars['delta_psm_x'], 'dy': [0]*len(data_vars['delta_psm_y']), 'title': 'Delta PSM X'},
            {'ax': axes[3, 1], 'dx': [0]*len(data_vars['delta_psm_x']), 'dy': data_vars['delta_psm_y'], 'title': 'Delta PSM Y'},
        ]

        self.quivers = []
        for config in plot_configs:
            quiver = plot_overlay(config['ax'], wf_x, wf_y, config['dx'], config['dy'],
                                  self.vertical_lines, self.horizontal_lines, title=config['title'])
            self.quivers.append((quiver, config['title']))

        axes[3, 2].axis('off')

        self.canvas.draw()

        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        for quiver, title in self.quivers:
            if event.inaxes == quiver.axes and quiver.contains(event)[0]:
                selected_region = self.region_combo.currentText()
                df_lot = self.df[self.df['UNIQUE_ID'] == self.unique_id]
                data = df_lot if selected_region == 'All' else df_lot[df_lot['region'] == selected_region]

                wf_x = data['wf_x'].tolist()
                wf_y = data['wf_y'].tolist()
                dx = quiver.U
                dy = quiver.V

                enlarged_window = EnlargedPlotWindow(
                    wf_x, wf_y, dx, dy,
                    title=title + f' - {selected_region}',
                    v_lines=self.vertical_lines,
                    h_lines=self.horizontal_lines
                )
                self.enlarged_plot_windows.append(enlarged_window)
                enlarged_window.show()
