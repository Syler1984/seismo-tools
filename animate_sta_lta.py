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
pause = False

save_fps = 60
animation_dt = 10

stream_path = 'C:/data/seismic_streams/NYSH.IM.00.EHZ.2021.091'
save_animation = False
save_name = 'sta_lta_60sec.gif'

slice_start = '2021-04-01T12:35:30'
slice_end = 60
window_length = 20
lta_length = 12
sta_length = 3

actual_slice_end = slice_end + 20

highpass_frequency = 2

normalize_stream = True

sta_lta_threshold = 2.

events = ['2021-04-01T12:35:55.5', '2021-04-01T12:36:05']


# Functions
def prepare_plot(figure_size=(14, 8), dpi=90):
    figure = plt.figure(figsize=figure_size, dpi=dpi)
    figure.suptitle('STA/LTA')

    axes = figure.subplots(3, 1, sharex=True)
    axes = {
        'wave': axes[0],
        'sta_lta': axes[1],
        'ratio': axes[2]
    }

    plots = {
        'wave': axes['wave'].plot([], [], lw=1., color='#000')[0],
        'sta': axes['sta_lta'].plot([], [], '--', lw=1., color='#4290f5')[0],
        'lta': axes['sta_lta'].plot([], [], lw=1., color='#f54263')[0],
        'ratio': axes['ratio'].plot([], [], lw=1., color='#000')[0]
    }

    figure.tight_layout()

    return figure, axes, plots


def plot_stream(i, figure, axes, plots, wave, sta, lta, sta_lta_ratio, w_length, events):
    x_data = np.arange(wave.shape[0])

    for _, ax in axes.items():
        ax.set_xlim(i, i + w_length)

    plots['wave'].set_data(x_data[i:i + w_length], wave[i:i + w_length])
    plots['sta'].set_data(x_data[i:i + w_length], sta[i:i + w_length])
    plots['lta'].set_data(x_data[i:i + w_length], lta[i:i + w_length])
    plots['ratio'].set_data(x_data[i:i + w_length], sta_lta_ratio[i:i + w_length])

    # axes['ratio'].vlines(sta_lta_threshold, 0, 2., color='#ff4a4a66', lw=2)


def prepare_stream(stream, start=None, end=None, w_length=0, normalize=False, frequency=0., events=None):

    if not events:
        events = []

    if not start:
        start = stream[0].stats.starttime
    elif type(start) in [float, int]:
        start = stream[0].stats.starttime + start * 60.
    else:
        start = UTCDateTime(start)
    display_end = None
    if not end:
        end = stream[-1].stats.endtime
        display_end = end - w_length
    elif type(end) in [float, int]:
        end = start + end
        display_end = end
        end += w_length
    else:
        end = UTCDateTime(end)
        display_end = end
        end += w_length

    display_length = display_end - start
    if display_length < window_length:
        raise AttributeError('Available length of a stream to plot is shorter than window_length! '
                             'Please, adjust slice_start, slice_end or window_length. '
                             f'\nCurrent window_length: {w_length} s, '
                             f'current available stream length: {display_length} s')

    stream = stream.slice(start, end)

    if len(stream) > 1:
        raise AttributeError('Stream has more than one trace on sliced data! '
                             'Please, chose slice on continuous data span.')

    if normalize:
        stream.normalize()
    if frequency:
        stream.filter(type='highpass', freq=frequency)

    # Convert events to sample positions
    stream_frequency = stream[0].stats.sampling_rate
    events = [int((x - start)*stream_frequency) for x in events]

    return stream, start, end, events


def window_average(data, w_length, default_length, default_value=0.):
    result = np.full(data.shape[0], default_value)
    for i in range(default_length, data.shape[0]):
        result[i] = np.mean(np.abs(data[i - w_length:i]))

    return result


def build_data(stream, sta_length, lta_length, w_length):
    stream_data = stream[0].data
    frequency = int(stream[0].stats.sampling_rate)

    # Translate from seconds to samples
    l_samples_sta = sta_length * frequency
    l_samples_lta = lta_length * frequency
    l_samples_w_length = w_length * frequency

    sta = window_average(stream_data, l_samples_sta, l_samples_lta, 0.)
    lta = window_average(stream_data, l_samples_lta, l_samples_lta, 0.001)

    sta_lta = sta / lta

    return stream_data, sta, lta, sta_lta, l_samples_lta, l_samples_w_length


if __name__ == '__main__':

    # Events to UTCDateTime
    events = [UTCDateTime(x) for x in events]

    # Read data
    st = read(stream_path)
    s_start = st[0].stats.starttime.strftime("%d.%m.%YT%H:%M:%S")
    s_end = st[-1].stats.endtime.strftime("%d.%m.%YT%H:%M:%S")
    print(f'Read seismic stream with {len(st)} traces and time span: {s_start} - {s_end}')

    st, start, end, events = prepare_stream(st, slice_start, slice_end, window_length,
                                            normalize_stream, highpass_frequency, events)

    wave, sta, lta, sta_lta_ratio, \
        lta_samples_length, window_samples_length = build_data(st, sta_length, lta_length, window_length)

    # Setup plots
    figure, axes, plots = prepare_plot()

    axes['wave'].set_ylabel('Wave')
    axes['sta_lta'].set_ylabel('STA & LTA')
    axes['ratio'].set_ylabel('STA/LTA')

    max_ratio = np.max(sta_lta_ratio)
    max_ratio *= 1.1
    max_sta_lta = max(np.max(sta), np.max(lta))
    extra_min_max = 0.1 * (np.max(wave) - np.min(wave))

    axes['wave'].set_ylim(np.min(wave) - extra_min_max, np.max(wave) + extra_min_max)
    axes['sta_lta'].set_ylim(0, max_sta_lta)
    axes['ratio'].set_ylim(0, max_ratio)

    axes['ratio'].axhline(sta_lta_threshold, color='#ff4a4a66', lw=2)

    for x in events:
        axes['wave'].axvline(x, color='#ff4a4a66', lw=2)

    # Start animation
    animation = FuncAnimation(figure, plot_stream,
                              range(lta_samples_length, wave.shape[0] - window_samples_length, 5),
                              fargs=[figure, axes, plots, wave, sta, lta, sta_lta_ratio, window_samples_length, events],
                              blit=False, interval=animation_dt, repeat=True, save_count=600)

    plt.show()

    if save_animation:
        print(f'Saving animation to a file: "{save_name}"')
        animation.save(save_name, fps=save_fps)
