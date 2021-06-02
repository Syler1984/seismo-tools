import numpy as np
import h5py as h5
import obspy.core as oc
import sys
import argparse
import os
import multiprocessing as mp

from utils.h5_tools import write_batch


def convert_data(data, no_filter, no_detrend):
    """
    Converts data from meier format to requiered:
        1. Detrend.
        2. 2Hz highpass filtering.
        3. Slice.
        4. Local absolute max normalization.
    """
    X = np.zeros((data.shape[1], data.shape[2], 3))

    for i in range(data.shape[1]):

        c_data = data[:, i, :]

        channels = [c_data[j, :] for j in range(c_data.shape[0])]

        d_length = c_data.shape[1]
        r_length = 400
        ch_num = len(channels)

        X = np.zeros((d_length, ch_num))

        # Process
        for j, chan in enumerate(channels):

            if no_detrend and no_filter:
                X[i, :, j] = chan
            else:
                trace = oc.Trace(data = chan)
                trace.stats.sampling_rate = 100

                if not no_detrend:
                    trace.detrend(type = 'linear')
                if not no_filter:
                    trace.filter(type = 'highpass', freq = 2.)

                X[i, :, j] = trace.data

    # Slice
    X = X[:, :r_length, :]

    # Normalize
    global_norm = True

    if global_norm:

        for i in range(X.shape[0]):

            loc_max = np.max(np.abs(X[i, :, :]))
            X[i, :, :] = X[i, :, :] / loc_max

    else:

        for i in range(X.shape[0]):
            for j in range(ch_num):

                loc_max = np.max(np.abs(X[i, :, j]))
                X[i, :, j] = X[i, :, j] / loc_max

    return X


def process(path, names_stack, span, save_path, label,
            id, read_lock = None, write_lock = None,
            no_filter = False, no_detrend = False):
    """
    Represents single data conversion process
    :param path: Path to original .h5 file
    :param names_stack: Name stack to dataset within original .h5 file
    :param span: (start_idx, end_id) tuple for current batch
    :param save_path: Converted .h5 dataset path
    :param label: Y dataset value
    :param id: ID dataset value
    :param read_lock: multiprocessing lock for original .h5 file
    :param write_lock: multiprocessing lock for converted .h5 file
    :param no_filter: disable filter
    :param no_detrend: disable detrend
    :return:
    """
    batch_size = span[1] - span[0]
    Y = np.full(batch_size, label, dtype = int)
    Z = np.full(batch_size, id, dtype = object)

    # Read data
    with h5.File(path, 'r') as meier_set:

        for s_name in names_stack:
            meier_set = meier_set[s_name]

        # Get data
        if read_lock:
            read_lock.acquire()
            try:
                data = meier_set[:, span[0]:span[1], :]
            finally:
                read_lock.release()
        else:
            data = meier_set[:, span[0]:span[1], :]

    # Convert data
    X = convert_data(data, no_filter, no_detrend)

    # Save data
    if write_lock:
        write_lock.acquire()
        try:
            write_batch(save_path, 'X', X)
            write_batch(save_path, 'Y', Y)
            write_batch(save_path, 'Z', Z, string = True)
        finally:
            write_lock.release()
    else:
        write_batch(save_path, 'X', X)
        write_batch(save_path, 'Y', Y)
        write_batch(save_path, 'Z', Z, string = True)


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
    parser.add_argument('--no-filter', help = 'Disable filter', action = 'store_true')
    parser.add_argument('--no-detrend', help = 'Disable detrend', action = 'store_true')

    args = parser.parse_args()

    # Initialize parameters
    meier_path = args.meier_path
    save_path = args.save_path

    meier_set_names_stack = ['noise', 'wforms']

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
    import time
    print(f'Converting data from {start} to {end} with {procs} processes:')
    for b in range(batch_num):

        # Track batch process time
        start_time = time.time()

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
            c_proc_batch_spans += [(start + c_start_pos, start + c_start_pos + c_proc_batch_size)]
            c_start_pos += c_proc_batch_size

        c_proc_batch_spans += [(start + c_start_pos, start + c_start_pos + c_proc_batch_size + c_batch_size % procs)]

        print_message = f'Batch {b} out of {batch_num} ' \
                        f'(from {c_proc_batch_spans[0][0]} to {c_proc_batch_spans[-1][1]})..' + ' ' * 40
        print(f'{print_message[:70]}', end = '', flush = True)

        if procs == 1:
            process(meier_path, meier_set_names_stack,
                    c_proc_batch_spans[0],
                    save_path,
                    label, _id,
                    no_filter = args.no_filter, no_detrend = args.no_detrend)
        else:
            # Preparing sub-processes
            read_lock = mp.Lock()
            write_lock = mp.Lock()
            processes = []
            for i in range(procs):
                processes += [mp.Process(target = process, args = (meier_path, meier_set_names_stack,
                                                                   c_proc_batch_spans[i],
                                                                   save_path,
                                                                   label, _id,
                                                                   read_lock, write_lock,
                                                                   args.no_filter, args.no_detrend))]

            # Process batch
            for i in range(procs):
                processes[i].start()

            # Join processes
            for i in range(procs):
                processes[i].join()

        time_span = time.time() - start_time
        print(f'..saved! {time_span:.4f} seconds!')
