import numpy as np
import h5py as h5
import obspy.core as oc
import sys
import argparse
import os
import multiprocessing as mp

from utils.h5_tools import write_batch


def convert_data(dataset, idx, lock):
    """
    Converts data from meier format to requiered:
        1. Detrend.
        2. 2Hz highpass filtering.
        3. Slice.
        4. Local absolute max normalization.
    """
    # Get data
    lock.acquire()
    try:
        data = dataset[:, idx, :]
    finally:
        lock.release()

    channels = [data[i, :] for i in range(data.shape[0])]

    d_length = data.shape[1]
    r_length = 400
    ch_num = len(channels)

    X = np.zeros((d_length, ch_num))

    # Process
    for i, chan in enumerate(channels):

        trace = oc.Trace(data = chan)
        trace.stats.sampling_rate = 100

        trace.detrend(type = 'linear')
        trace.filter(type = 'highpass', freq = 2.)

        X[:, i] = trace.data

    # Slice
    X = X[:r_length, :]

    # Normalize
    global_norm = True

    if global_norm:

        loc_max = np.max(np.abs(X[:, :]))
        X[:, :] = X[:, :] / loc_max

    else:

        for i in range(ch_num):

            loc_max = np.max(np.abs(X[:, i]))
            X[:, i] = X[:, i] / loc_max

    return X.reshape((1, *X.shape))


def process(read_lock, write_lock, path, names_stack, span, save_path, label, id):

    batch_size = span[1] - span[0]
    X = np.zeros((batch_size, 400, 3))
    Y = np.full(batch_size, label, dtype = int)
    Z = np.full(batch_size, id, dtype = object)

    with h5.File(path, 'r') as meier_set:

        for s_name in names_stack:
            meier_set = meier_set[s_name]

        b = 0
        for i in range(span[0], span[1]):

            X[b] = convert_data(meier_set, i, read_lock)
            b += 1

        write_lock.acquire()
        try:
            write_batch(save_path, 'X', X)
            write_batch(save_path, 'Y', Y)
            write_batch(save_path, 'Z', Z, string = True)
        finally:
            write_lock.release()


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('meier_path', help = 'Path to Meier dataset file')
    parser.add_argument('--save_path', help = 'Save file path, default: "meier_converted.h5"',
                        default = 'meier_converted.h5')
    parser.add_argument('--start', '-s', help = 'Start index for data conversion, default: 0', default = 0)
    parser.add_argument('--end', '-e', help = 'Last index for data conversion, default: EOF', default = None)
    parser.add_argument('--procs', help = 'Number of parallel processes, set 0 to run '
                                          'os.cpu_count() number; default: 1', default = 1)
    parser.add_argument('--batch_size', '-b', help = 'Batch size in records (one record is: waveform + label)'
                                                     ', default: 10000', default = 1000)
    parser.add_argument('--inspect', help = 'Use this flag to print info about Meier dataset without'
                                            ' performing data conversion', action = 'store_true')

    args = parser.parse_args()

    # Initialize parameters
    meier_path = args.meier_path
    save_path = args.save_path

    meier_set_names_stack = ['X']
    # meier_set_names_stack = ['noise', 'wforms']

    batch_size = int(args.batch_size)

    label = 2
    _id = 'meier_noise'

    if int(args.procs):
        procs = int(args.procs)
    else:
        procs = os.cpu_count()

    # Read data
    with h5.File(meier_path, 'r') as meier_set:

        for s_name in meier_set_names_stack:
            meier_set = meier_set[s_name]

        meier_inspect_line = str(meier_set)
        meier_set_length = meier_set.shape[1]

    # Print set info?
    if args.inspect:

        print('Dataset keys stack: ', meier_set_names_stack)
        print('Dataset info: ', meier_inspect_line)
        sys.exit(0)

    # Convert data
    start = int(args.start)
    end = meier_set_length

    if args.end:
        end = int(args.end)

    data_span = end - start
    last_batch = data_span % batch_size
    batch_num = data_span // batch_size
    if last_batch:
        batch_num += 1

    b = 0
    print(f'Converting data from {start} to {end}')
    for b in range(batch_num):

        # Get current batch size
        c_batch_size = batch_size
        if b == batch_num - 1 and last_batch:
            c_batch_size = last_batch

        batch_start_pos = b * batch_size

        # Split batch between processes
        c_proc_batch_spans = []
        c_proc_batch_size = c_batch_size // procs
        c_start_pos = batch_start_pos
        for i in range(procs - 1):
            c_proc_batch_spans += [(c_start_pos, c_start_pos + c_proc_batch_size)]
            c_start_pos += c_proc_batch_size

        c_proc_batch_spans += [(c_start_pos, c_start_pos + c_proc_batch_size + c_batch_size % procs)]

        print(f'Batch {b} out of {batch_num} '
              f'(from {c_proc_batch_spans[0][0]} to {c_proc_batch_spans[-1][1]})..', end = '', flush = True)

        # Preparing sub-processes
        read_lock = mp.Lock()
        write_lock = mp.Lock()
        processes = []
        for i in range(procs):
            processes += [mp.Process(target = process, args = (read_lock, write_lock,
                                                               meier_path, meier_set_names_stack,
                                                               c_proc_batch_spans[i],
                                                               save_path,
                                                               label, _id))]

        # Process batch
        for i in range(procs):
            processes[i].start()

        # Join processes
        for i in range(procs):
            processes[i].join()

        print('\t\t..saved!')
