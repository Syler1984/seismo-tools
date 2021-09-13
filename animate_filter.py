"""
This script plots a slice of seismic trace animation.
You can change animation, filtering and other parameters (eg. slice time span, file name) either in
dedicated section "Parameters" or by using command line arguments.
Use "-h" option for command line arguments description.

Usage:
"python animate_filter.py"
OR
"python animate_filter.py [OPTION VALUE]"
OR
"python animate_filter.py -h"

If no ffmpeg found during animation saving, run:
"conda install -c conda-forge ffmpeg"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from obspy import read, UTCDateTime

# Parameters
pause = False

save_fps = 60
animation_dt = 10

stream_path = 'C:/data/seismic_streams/TMSK.D0.00.SHZ.2015.347'
save_name = 'filter.gif'

slice_start = 120
slice_end = 180

highpass_frequencies = [0.0, (0.000001, 0.001, 100)]


# Functions
def prepare_plot():
    figure = plt.figure(figsize = (8, 4), dpi = 90)
    figure.suptitle('no filter')
    axes = figure.add_subplot(111)
    plot, = axes.plot([], [], lw=0.5, color='#000')
    figure.tight_layout()

    return figure, axes, plot


def plot_filtered(figure, axes, plot, stream, freq, do_filter):

    if len(stream) > 1:
        raise AttributeError('Stream has more than one trace on sliced data! '
                             'Please, chose slice on continuous data span.')

    freq = freq
    stream = stream.copy()

    if do_filter:
        stream.filter('highpass', freq=freq)
        figure.suptitle(f'highpass = {freq}')
    else:
        figure.suptitle('no filter')

    axes.set_xlim(0, stream[0].data.shape[0])
    axes.set_ylim(np.min(stream[0].data), np.max(stream[0].data))
    plot.set_data(np.array(range(stream[0].data.shape[0])), stream[0].data)


def plot_stream(frequency, figure, axes, plot, stream, start=None, end=None):
    if not start:
        start = stream[0].stats.starttime
    elif type(start) in [float, int]:
        start = stream[0].stats.starttime + start*60.
    else:
        start = UTCDateTime(start)
    if not end:
        end = stream[-1].stats.endtime
    elif type(end) in [float, int]:
        end = start + end*60.
    else:
        end = UTCDateTime(end)

    stream = stream.slice(start, end)

    if frequency == 0.:
        plot_filtered(figure, axes, plot, stream, frequency, False)
    else:
        plot_filtered(figure, axes, plot, stream, frequency, True)

    return plot


def frequencies() -> object:
    global highpass_frequencies
    for x in highpass_frequencies:
        if type(x) is tuple:
            f_range = np.linspace(x[0], x[1], x[2])
            for y in f_range:
                yield y
        else:
            yield x


if __name__ == '__main__':

    # Read data
    st = read(stream_path)
    s_start = st[0].stats.starttime.strftime("%d.%m.%YT%H:%M:%S")
    s_end = st[-1].stats.endtime.strftime("%d.%m.%YT%H:%M:%S")
    print(f'Read seismic stream with {len(st)} traces and time span: {s_start} - {s_end}')

    max_time = 10
    plot_dt = 1

    figure, axes, plot = prepare_plot()

    animation = FuncAnimation(figure, plot_stream, frequencies,
                              fargs=[figure, axes, plot, st, slice_start, slice_end],
                              blit=False, interval=animation_dt, repeat=True)

    # Display the animation
    plt.show()
    # animation.save(save_name, fps=save_fps)
