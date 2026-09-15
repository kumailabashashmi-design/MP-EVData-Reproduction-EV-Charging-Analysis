import time
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

gmt_format = '%a, %d %b %Y %H:%M:%S GMT'

def gmt_to_datetime(gmt):
    time_struct = time.strptime(gmt, gmt_format)
    output_format = '%Y-%m-%d %H:%M:%S'
    return time.strftime(output_format, time_struct)

def gmt_to_timestamp(gmt):
    if type(gmt) is str:
        time_step = time.mktime(time.strptime(gmt, gmt_format))
        return time_step
    else:
        time_steps = [time.mktime(time.strptime(t, gmt_format)) for t in gmt]
        return time_steps

def interpolate_signal(timestamp, signal):
    interp_func = interp1d(timestamp, signal, kind="nearest")
    xs = np.arange(np.min(timestamp), np.max(timestamp)+1, 1)
    ys = interp_func(xs)
    return xs.tolist(), ys.tolist()

def down_sample(signal, scale):
    sample_index = np.arange(0, len(signal), scale).tolist()
    signal_down = np.array(signal)[sample_index].tolist()
    if len(signal) % scale != 0:
        signal_down = signal_down + [signal[-1]]
    return signal_down

def plot_session(pilot, current, title, path):
    plt.subplot(2, 1, 1)
    plt.plot(pilot, label="pilotSignal")
    plt.plot(current, label="chargingCurrent")
    plt.legend()
    plt.title(title)
    plt.subplot(2, 1, 2)
    plt.legend()
    plt.savefig(path)
    plt.clf()

def plot_training_loss(*args, model_name, labels):
    # Import required modules
    from matplotlib.ticker import FixedLocator, AutoMinorLocator

    # Configure global plot parameters
    plt.rcParams.update({
        'font.size': 18,  # Global font size
        'axes.linewidth': 1.5,  # Axis line width
        'lines.linewidth': 3,  # Default line width
        'lines.markersize': 4,  # Marker size
        'xtick.direction': 'in',  # X-axis tick direction
        'ytick.direction': 'in',  # Y-axis tick direction
        'xtick.major.size': 5,  # X-axis major tick length
        'ytick.major.size': 5,  # Y-axis major tick length
        'xtick.minor.size': 3,  # X-axis minor tick length
        'ytick.minor.size': 3  # Y-axis minor tick length
    })

    sub_num = len(labels)
    fig = plt.figure(figsize=(8, 6 if sub_num == 1 else 4 * sub_num))

    # Predefined line style combinations (color, linestyle, marker)
    style_presets = [
        ('#1f77b4', '-', None),  # Blue solid line
        ('#ff7f0e', '--', None),  # Orange dashed line
        ('#2ca02c', '-.', None),  # Green dash-dot line
        ('#d62728', (0, (3, 1, 1, 1)), None)  # Red custom dashed line
    ]

    for i in range(sub_num):
        ax = fig.add_subplot(sub_num, 1, i + 1)

        # Set axes
        ax.set_xlabel('Epoch', fontsize=18, labelpad=8)
        ax.set_ylabel('Loss', fontsize=18, labelpad=8)

        # Set fixed tick ranges
        ax.set_xlim(0, 500)  # Fixed x-axis range 0-500
        ax.set_ylim(0, 0.8)  # Fixed y-axis range 0-0.8

        # Configure tick parameters - use fixed ticks
        ax.xaxis.set_major_locator(FixedLocator([0, 100, 200, 300, 400, 500]))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8]))
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        # Draw curves (cycle through predefined styles)
        line_style = style_presets[i % len(style_presets)]
        ax.plot(args[i],
                color=line_style[0],
                linestyle=line_style[1],
                marker=line_style[2])

        # Decorate plot
        ax.grid(False)  # Turn off grid
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend (auto-select best position)
        ax.legend([labels[i]], loc='upper right', frameon=False,
                  bbox_to_anchor=(1.0, 1.0), borderaxespad=0.5)

        # Add tick marks
        ax.tick_params(axis='both', which='both',
                       bottom=True, top=False,
                       left=True, right=False,
                       labelbottom=True)

    # Adjust layout and save
    plt.tight_layout(pad=2.0)
    plt.savefig(f"{model_name}_loss.png",
                dpi=600,
                bbox_inches='tight')
    plt.close(fig)

def plot_driver_generation(x1, x2, path):
    plt.subplot(2, 1, 1)
    plt.plot(x1, color="green", label="with zero padding")
    plt.plot(x2, color="orange", label="without zero padding")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(x2, label="Generated curve")
    img_path = f"{path}.png"
    plt.savefig(img_path)
    plt.clf()
    pkl_path = f"{path}.pkl"
    with open(f"{pkl_path}", "wb") as f:
        pkl.dump(x2, f)
    print(f"{path} generated done!")

def plot_station_generation(x, path):
    plt.plot(x, label="Generated charging station load")
    plt.savefig(f"{path}.png")
    plt.clf()
    with open(f"{path}.pkl", "wb") as f:
        pkl.dump(x, f)
    print(f"{path} generated done!")