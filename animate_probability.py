"""
This script generates an animation for STA/LTA algorithm.
You can change animation and other parameters (eg. slice time span, file name) in
dedicated section "Parameters".

Usage:
"python animate_sta_lta.py"

If no ffmpeg found during animation saving, run:
"conda install -c conda-forge ffmpeg"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from obspy import read, UTCDateTime
from obspy.signal.trigger import classic_sta_lta


# Parameters
save_animation = True

save_fps = 60
animation_dt = 20

stream_path = '0_wave.npy'
prediction_path = '0_scores.npy'
save_name = 'probability_60sec.gif'

window_length = 20
max_plot_length = 800
lta_length = 8
sta_length = 2

highpass_frequency = 2

normalize_stream = True

sta_lta_threshold = 2.

events = ['2021-04-01T12:35:55.5', '2021-04-01T12:36:05']

plot_step = 100

ratio_gradient_start = '#3d3685'
ratio_gradient_end = '#f54263'
ratio_start = 0
ratio_end = 3

# Internally used parameters
pause = False


# Functions
def prepare_plot(figure_size=(13, 4), dpi=90, frequency=100, w_length=1000, step=50):
    figure = plt.figure(figsize=figure_size, dpi=dpi)

    axes = figure.subplots(2, 1, sharex=True)
    axes = {
        'wave': axes[0],
        'scores': axes[1]
    }

    n_steps = w_length // step
    if w_length % step:
        n_steps += 1

    plots = {
        'wave': axes['wave'].plot([], [], lw=1., color='#000')[0],
        'p-scores': [axes['scores'].plot([], [], lw=1., color='#000')[0] for _ in range(n_steps)],
        's-scores': [axes['scores'].plot([], [], '--', lw=1., color='#000')[0] for _ in range(n_steps)]
    }

    figure.tight_layout()

    return figure, axes, plots


def hex_color(color):
    """
    Convert hex color string to four int values: rgba
    :param color:
    :return: (int, int, int, int)
    """
    if color[0] == '#':
        color = color[1:]

    alpha = 255
    if len(color) in [3, 4]:
        red = int(color[0]*2, 16)
        green = int(color[1]*2, 16)
        blue = int(color[2]*2, 16)
        if len(color) == 4:
            alpha = int(color[3]*2, 16)
    elif len(color) in [6, 8]:
        red = int(color[:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
        if len(color) == 8:
            alpha = int(color[6:8], 16)
    else:
        raise AttributeError('Failed to parse hex color! Please, make sure that color is in one of these formats: '
                             '"#rgb" "#rgba" "#rrggbb" "#rrggbbaa"')

    return red, green, blue, alpha


def gradient(value, color_1, color_2, value_1, value_2):
    """
    Returns color linearly based on value's position between value_1 or value_2 as left and right borders.
    """
    if value_1 >= value_2:
        raise AttributeError('value_1 should be lower than value_2!')
    if value <= value_1:
        return color_1
    if value >= value_2:
        return color_2

    value_diff = value_2 - value_1
    value_x = (value - value_1)/value_diff  # value relative offset

    red_diff = color_2[0] - color_1[0]
    green_diff = color_2[1] - color_1[1]
    blue_diff = color_2[2] - color_1[2]
    alpha_diff = color_2[3] - color_1[3]

    red = int(red_diff * value_x + color_1[0])
    green = int(green_diff * value_x + color_1[1])
    blue = int(blue_diff * value_x + color_1[2])
    alpha = int(alpha_diff * value_x + color_1[3])

    return red, green, blue, alpha


def _chs(channel):
    """
    Returns string representation of a single color channel in hex (type str)
    """
    if channel < 15:
        return '0' + hex(channel)[2:]
    return hex(channel)[2:]


def color_hex_string(color):
    """
    Returns hex string representation of a int color tuple: (r,g,b,a)
    """
    return f'#{_chs(color[0])}{_chs(color[1])}{_chs(color[2])}{_chs(color[3])}'


def plot_gradient(plots, x_data, y_data, min_value, max_value, min_color, max_color, step = 10):

    # Split data, initialize split vectors for x_data, y_data and colors
    data_length = x_data.shape[0]
    n_steps = len(plots)
    last_step = data_length % step
    if last_step:
        n_steps -= 1
        last_plot = plots[-1]
        plots = plots[:-1]

    if last_step:
        x_steps = np.resize(x_data[:-last_step], (n_steps, step))
        y_steps = np.resize(y_data[:-last_step], (n_steps, step))
    else:
        x_steps = np.resize(x_data, (n_steps, step))
        y_steps = np.resize(y_data, (n_steps, step))

    # Split colors into three (four) hex values
    min_color = hex_color(min_color)
    max_color = hex_color(max_color)

    # Plot fragments
    for x, y, plot in zip(x_steps, y_steps, plots):
        # Calculate colors for every step
        color = color_hex_string(gradient(int(y.mean()), min_color, max_color, min_value, max_value))
        plot.set_color(color)
        plot.set_data(x, y)

    if last_step:
        x = x_data[-last_step:]
        y = y_data[-last_step:]
        color = color_hex_string(gradient(int(y.mean()), min_color, max_color, min_value, max_value))
        last_plot.set_color(color)
        last_plot.set_data(x, y)


def plot_stream(i, figure, axes, plots, wave, scores, w_length, events):

    x_data = np.arange(wave.shape[0])

    for _, ax in axes.items():
        ax.set_xlim(i, i + w_length)

    plots['wave'].set_data(x_data[i:i + w_length], wave[i:i + w_length, 1])

    # Plot calculated data
    global max_plot_length

    m_length = w_length
    if max_plot_length and m_length > max_plot_length:
        m_length = max_plot_length

    # Plot sta/lta ratio
    global ratio_gradient_start
    global ratio_gradient_end
    global ratio_start
    global ratio_end
    global plot_step

    plot_gradient(plots['p-scores'], x_data[i:i + m_length], scores[i:i + m_length, 0],
                  ratio_start, ratio_end, ratio_gradient_start, ratio_gradient_end, step=plot_step)
    plot_gradient(plots['s-scores'], x_data[i:i + m_length], scores[i:i + m_length, 1],
                  ratio_start, ratio_end, '#34db1a', ratio_gradient_end, step=plot_step)


if __name__ == '__main__':

    # Events to UTCDateTime
    events = [2650, 3450]

    # Read data
    wave = np.load(stream_path)
    scores = np.load(prediction_path)

    w_max = np.max(np.abs(wave))
    wave /= w_max

    stream_frequency = 100
    window_samples_length = window_length * stream_frequency

    # Setup plots
    figure, axes, plots = prepare_plot(w_length=window_samples_length, frequency=stream_frequency, step=plot_step)

    max_prediction = np.max(scores) * 1.1
    extra_min_max = 0.1 * (np.max(wave) - np.min(wave))

    axes['wave'].set_ylim(np.min(wave) - extra_min_max, np.max(wave) + extra_min_max)
    axes['scores'].set_ylim(0, max_prediction)
    axes['scores'].axhline(0.95, color='#ff4a4a66', lw=2)

    for x in events:
        axes['wave'].axvline(x, color='#ff4a4a66', lw=2)

    # Start animation
    animation = FuncAnimation(figure, plot_stream,
                              range(0, wave.shape[0] - window_samples_length, 10),
                              fargs=[figure, axes, plots, wave, scores, window_samples_length, events],
                              blit=False, interval=animation_dt, repeat=True, save_count=600)

    plt.show()

    if save_animation:
        print(f'Saving animation to a file: "{save_name}"')
        animation.save(save_name, fps=save_fps)
