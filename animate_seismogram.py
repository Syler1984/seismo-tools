"""
This script plots a slice of seismic trace animated from left to right, can also plot multiple traces with
shared X-axis values.
You can change animation, filtering and other parameters (eg. slice time span, file name) in
dedicated section "Parameters".

Usage:
"python animate_seismogram.py"

If no ffmpeg found during animation saving, run:
"conda install -c conda-forge ffmpeg"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from obspy import read, UTCDateTime
from os import listdir
from os.path import isdir, isfile, join

# Parameters
save_fps = 60
animation_dt = 30

animation_step = 5

streams_paths = ['C:/data/seismic_streams/']
max_plots = 30
save_animation = True
save_name = 'stream_60sec_many_short.gif'

slice_start = 45330
slice_end = 120

highpass_frequency = 2.
normalize_stream = True

hide_y_labels = True
hide_x_labels = True

window_length = 60
plot_length = 59

data_sampling_rate = 100


def prepare_plot(v_num=1, figure_size=(13, 5), dpi=90):
    figure = plt.figure(figsize=figure_size, dpi=dpi)
    axes = figure.subplots(v_num, 1, sharex=True)

    try:
        plots = [ax.plot([], [], lw=1., color='#000')[0] for ax in axes]
    except TypeError:
        axes = [axes]
        plots = [ax.plot([], [], lw=1., color='#000')[0] for ax in axes]

    return figure, axes, plots


def prepare_stream(stream, start=None, end=None):

    global normalize_stream
    global highpass_frequency

    if not start:
        start = stream[0].stats.starttime
    elif type(start) in [float, int]:
        start = stream[0].stats.starttime + start
    else:
        start = UTCDateTime(start)

    if not end:
        end = stream[-1].stats.endtime
    elif type(end) in [float, int]:
        end = start + end
    else:
        end = UTCDateTime(end)

    stream = stream.slice(start, end)

    if len(stream) > 1:
        raise AttributeError('Stream has more than one trace on sliced data! '
                             'Please, chose slice on continuous data span.')

    if normalize_stream:
        stream.normalize()
    if highpass_frequency:
        stream.filter(type='highpass', freq=highpass_frequency)

    if len(stream) == 1:
        return stream[0].data
    else:
        return None


def plot_stream(i, figure, axes, plots, streams, x_data):

    global window_length
    global plot_length
    global x_length

    for ax in axes:
        ax.set_xlim(i, i + window_length)

    for plot, stream in zip(plots, streams):
        plot.set_data(x_data[i:i + plot_length], stream[i:i + plot_length])


if __name__ == '__main__':

    # Parse input
    if len(streams_paths) == 1 and isdir(streams_paths[0]):
        streams_dir = streams_paths[0]
        streams_paths = [join(streams_dir, f) for f in listdir(streams_dir) if isfile(join(streams_dir, f))]

        if len(streams_paths) > max_plots:
            streams_paths = streams_paths[:max_plots]

    # From seconds to samples
    window_length = int(data_sampling_rate * window_length)
    plot_length = int(data_sampling_rate * plot_length)

    # Read data
    streams = [read(x) for x in streams_paths]
    print(f'Read seismic stream(s), total stream read: {len(streams)}')

    t_streams = []
    for x in streams:
        stream = prepare_stream(x, slice_start, slice_end)
        if stream is not None:
            t_streams.append(stream)
    streams = t_streams

    # Setup plots
    figure, axes, plots = prepare_plot(len(streams))

    for ax, stream in zip(axes, streams):

        if normalize_stream:
            y_max = 1.1
        else:
            y_max = np.max(stream) * 1.1

        ax.set_ylim(-y_max, y_max)

    if hide_y_labels:
        for ax in axes:
            y_axis = ax.get_yaxis()
            y_axis.set_visible(False)

    if hide_x_labels:
        for ax in axes[:-1]:
            x_axis = ax.get_xaxis()
            x_axis.set_visible(False)

    x_length = min([stream.shape[0] for stream in streams])
    for i in range(len(streams)):
        if streams[i].shape[0] > x_length:
            streams[i] = streams[i][:x_length]

    figure.tight_layout()
    x_data = np.arange(x_length)

    # Start animation
    animation = FuncAnimation(figure, plot_stream, range(1),
                              fargs=[figure, axes, plots, streams, x_data],
                              blit=False, interval=animation_dt, repeat=True, save_count=1)

    plt.show()

    if save_animation:
        print(f'Saving animation to a file: "{save_name}"')
        animation.save(save_name, fps=save_fps)
