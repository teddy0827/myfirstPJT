import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np

from config.config import Config
from utils.calculations import calculate_statistics

def plot_overlay(ax, x, y, dx, dy, v_lines, h_lines,
                 wafer_radius=Config.WAFER_RADIUS,
                 title='Wafer Vector Map', scale_factor=Config.SCALE_FACTOR,
                 show_statistics=True):
    magnitudes = np.sqrt(np.array(dx)**2 + np.array(dy)**2)
    magnitudes_nm = magnitudes * 1e3

    norm = plt.Normalize(vmin=magnitudes_nm.min(), vmax=magnitudes_nm.max())
    cmap = plt.cm.jet

    quiver = ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=scale_factor,
                       color=cmap(norm(magnitudes_nm)), width=0.0015, headwidth=3, headlength=3)

    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Overlay Error Magnitude (nm)')

    ax.axvline(0, color='red', linewidth=1.0, label='Central X')
    ax.axhline(0, color='red', linewidth=1.0, label='Central Y')

    for vline in v_lines:
        ax.axvline(vline, color='black', linestyle='--', linewidth=0.8)
    for hline in h_lines:
        ax.axhline(hline, color='black', linestyle='--', linewidth=0.8)

    wafer_circle = Circle((0, 0), wafer_radius, color='green', fill=False,
                          linestyle='-', linewidth=2, label='Wafer Boundary')
    ax.add_patch(wafer_circle)

    scale_bar_label = f'{Config.SCALE_BAR_LENGTH * Config.SCALE_FACTOR * 1e3:.1f}nm'
    fontprops = fm.FontProperties(size=10)
    scalebar = AnchoredSizeBar(ax.transData, Config.SCALE_BAR_LENGTH_PIXELS, scale_bar_label,
                               Config.SCALE_BAR_POSITION, pad=0.1, color='black',
                               frameon=False, size_vertical=Config.SCALE_BAR_SIZE_VERTICAL,
                               fontproperties=fontprops)
    ax.add_artist(scalebar)

    if show_statistics:
        mean_plus_3sigma_x = calculate_statistics(dx)
        mean_plus_3sigma_y = calculate_statistics(dy)
        ax.text(0, Config.TEXT_POSITION_Y,
                f'|m|+3s X: {mean_plus_3sigma_x:.2f} nm\n|m|+3s Y: {mean_plus_3sigma_y:.2f} nm',
                fontsize=10, color='red', ha='center')

    ax.set_xlabel('Wafer X Coordinate (wf_x)')
    ax.set_ylabel('Wafer Y Coordinate (wf_y)')
    ax.set_title(title)
    ax.axis('equal')
    ax.grid(False)

    return quiver
