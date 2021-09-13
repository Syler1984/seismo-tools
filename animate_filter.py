"""
This script plots a slice of seismic trace butterworth highpass filtering animation.
You can change animation, filtering and other parameters (eg. slice time span, file name) in
dedicated section "Parameters".

Usage:
"python animate_filter.py"

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

stream_path = 'C:/data/seismic_streams/NYSH.IM.00.EHZ.2021.091'
save_animation = True
save_name = 'filter_60sec.gif'

slice_start = '2021-04-01T12:35:30'
slice_end = 60

highpass_frequencies = [(0., 0., 20), (0.0001, 0.001, 100),
                        (0.001, 0.1, 300), (0.1, 0.5, 100), (0.5, 1., 50), (1., 2., 100), (2., 2., 100)]

normalize_stream = True


# Functions
def prepare_plot():
    figure = plt.figure(figsize = (8, 4), dpi = 90)
    figure.suptitle('no filter')
    axes = figure.add_subplot(111)
    plot, = axes.plot([], [], lw=0.5, color='#000')
    figure.tight_layout()

    return figure, axes, plot


def plot_filtered(figure, axes, plot, stream, freq, frame, do_filter):

    if len(stream) > 1:
        raise AttributeError('Stream has more than one trace on sliced data! '
                             'Please, chose slice on continuous data span.')

    freq = freq
    stream = stream.copy()

    # Apply filter (or not) and update figure title
    if do_filter:
        stream.filter('highpass', freq=freq)
        if freq >= 0.:
            figure.suptitle(f'highpass = {freq:.2f} Hz')
        elif freq >= 0.1:
            figure.suptitle(f'highpass = {freq:.4f} Hz')
        else:
            figure.suptitle(f'highpass = {freq:.5e}')
    else:
        figure.suptitle(f'no filter')

    # Update Ox and Oy limits
    axes.set_xlim(0, stream[0].data.shape[0])
    axes.set_ylim(np.min(stream[0].data), np.max(stream[0].data))

    # Update plot
    plot.set_data(np.array(range(stream[0].data.shape[0])), stream[0].data)


def plot_stream(data, figure, axes, plot, stream):
    frame, frequency = data

    if frequency == 0.:
        plot_filtered(figure, axes, plot, stream, frequency, frame, False)
    else:
        plot_filtered(figure, axes, plot, stream, frequency, frame, True)


def frequencies() -> object:
    global highpass_frequencies
    i = 0
    for x in highpass_frequencies:
        if type(x) is tuple:
            f_range = np.linspace(x[0], x[1], x[2])
            for y in f_range:
                yield i, y
                i += 1
        else:
            yield i, x
            i += 1


def prepare_stream(stream, start=None, end=None, normalize=False):
    if not start:
        start = stream[0].stats.starttime
    elif type(start) in [float, int]:
        start = stream[0].stats.starttime + start*60.
    else:
        start = UTCDateTime(start)
    if not end:
        end = stream[-1].stats.endtime
    elif type(end) in [float, int]:
        end = start + end
    else:
        end = UTCDateTime(end)

    stream = stream.slice(start, end)
    if normalize:
        stream.normalize()
    return stream


if __name__ == '__main__':

    # Read data
    st = read(stream_path)
    s_start = st[0].stats.starttime.strftime("%d.%m.%YT%H:%M:%S")
    s_end = st[-1].stats.endtime.strftime("%d.%m.%YT%H:%M:%S")
    print(f'Read seismic stream with {len(st)} traces and time span: {s_start} - {s_end}')

    st = prepare_stream(st, slice_start, slice_end, normalize_stream)
    figure, axes, plot = prepare_plot()

    animation = FuncAnimation(figure, plot_stream, frequencies,
                              fargs=[figure, axes, plot, st],
                              blit=False, interval=animation_dt, repeat=True, save_count=600)

    # Display the animation
    plt.show()

    if save_animation:
        print(f'Saving animation to a file: "{save_name}"')
        animation.save(save_name, fps=save_fps)
