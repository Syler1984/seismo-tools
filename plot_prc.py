import pandas as pd
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import argparse


def load_keras_json_model(json_path, weights):
    """
    Load Keras model from model file and weights file
    :param json_path:
    :param weights:
    :return:
    """
    import tensorflow as tf
    from keras.models import model_from_json

    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json, custom_objects = {"tf": tf})

    model.load_weights(weights)

    return model


# PRC Utility
def get_prc_info(model, X, Y):
    """
    Predicts data on model and return percision-recall
    curve info (each prediction info: predicted label, true label, probability) as pandas.DataFrame
    """

    scores = model.predict(X, verbose = 1)

    Y_true = np.array(Y, dtype = 'int')
    Y_pred = scores.argmax(axis = 1)
    Y_score = scores.max(axis = 1)

    return pd.DataFrame({'Y_true': Y_true, 'Y_pred': Y_pred, 'Y_score': Y_score})


def get_data(data):
    """
    Returns data for later PRC building.
    Parameters:
    data - string to .csv file or pandas DataFrame object
    """
    from pandas.core.frame import DataFrame

    if type(data) == str:
        data = pd.read_csv(data)

    if type(data) == DataFrame:
        return data.to_numpy()

    return None


def predict_data(model, path, ratio = 1.):
    """
    Predicts data on model and return percision-recall
    curve info (each prediction info: predicted label, true label, probability) as pandas.DataFrame
    """

    with h5.File(path, 'r') as data:

        length = data['X'].shape[0]
        length = int(ratio * length)

        return get_data(get_prc_info(model, data['X'][:length], data['Y'][:length]))


def get_prc_data(data, true_label, threshold):
    """
    Return number of true positives, false positives and false negatives.
    Data shape: <row, column>. columns: 0 - true label, 1 - predicted label, 2 - probability.
    """
    tp = fp = fn = 0
    for i in range(data.shape[0]):

        if data[i][2] >= threshold:
            y_pred = data[i][1]
        else:
            y_pred = -1

        if data[i][0] == y_pred == true_label:
            tp += 1
        elif y_pred == true_label:
            fp += 1
        elif data[i][0] == true_label:
            fn += 1

    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = np.Inf
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = np.Inf

    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall}


def plot_prc(data, label, thresholds):
    """
    Gather and plot percision/recall curve.
    """
    # Gather data
    precision = []
    recall = []

    for t in thresholds:

        prc_data = get_prc_data(data, label, t)

        precision.append(prc_data['precision'])
        recall.append(prc_data['recall'])

    # Plot PRC
    plt.plot(recall, precision)


def append_prc(data, label, threshold, precision, recall):

    prc_data = get_prc_data(data, label, threshold)
    precision.append(prc_data['precision'])
    recall.append(prc_data['recall'])


def precision_recall_single(data, thresholds):

    precision_P = []
    precision_S = []
    precision_N = []

    recall_P = []
    recall_S = []
    recall_N = []

    for t in thresholds:

        append_prc(data, 0, t, precision_P, recall_P)
        append_prc(data, 1, t, precision_S, recall_S)
        append_prc(data, 2, t, precision_N, recall_N)

    return {'precision': {'p': precision_P, 's': precision_S, 'n': precision_N},
            'recall': {'p': recall_P, 's': recall_S, 'n': recall_N}}


def precision_recall(data_1, data_2, thresholds):

    precision_1_P = []
    recall_1_P = []
    precision_2_P = []
    recall_2_P = []

    precision_1_S = []
    recall_1_S = []
    precision_2_S = []
    recall_2_S = []

    for t in thresholds:

        append_prc(data_1, 0, t, precision_1_P, recall_1_P)
        append_prc(data_1, 1, t, precision_1_S, recall_1_S)

        append_prc(data_2, 0, t, precision_2_P, recall_2_P)
        append_prc(data_2, 1, t, precision_2_S, recall_2_S)

    return [precision_1_P, precision_1_S, precision_2_P, precision_2_S], [recall_1_P, recall_1_S, recall_2_P, recall_2_S]


def plot_prc_final(precision_recall, colors, labels, save_path = None, shared_axis = False, middle_point = None):
    """
    Gather and plot percision/recall curve.
    """
    from matplotlib.ticker import FormatStrFormatter

    # Set shorter alias
    pr = precision_recall

    # Plot PRC
    if shared_axis:
        fig = plt.figure(figsize = (3, 3), dpi = 300)
        axes = [fig.subplots(1, 1, sharey = True)]
        p_ax = 0
        s_ax = 0
    else:
        fig = plt.figure(figsize = (7, 3), dpi = 300)
        axes = fig.subplots(1, 2, sharey = True)
        p_ax = 0
        s_ax = 1

    mark_size = 4.
    star_size = 8.

    axes[p_ax].set_ylabel('Precision')

    if shared_axis:
        axes[p_ax].set_xlabel('Recall')
    else:
        fig.text(0.512, 0.00, 'Recall', va = 'center')

    star_color = '#d00000'

    # P

    for key, data in precision_recall.items():

        axes[p_ax].plot(data['recall']['p'], data['precision']['p'],
                        '-^', linewidth = 1., markersize = mark_size,
                        color = colors[key], label = labels[key])

        if not middle_point:
            continue

        axes[p_ax].plot(data['recall']['p'][middle_point], data['precision']['p'][middle_point],
                        '*', color = star_color, markersize = star_size)

    # S

    for key, data in precision_recall.items():

        axes[s_ax].plot(data['recall']['s'], data['precision']['s'],
                        '-o', linewidth = 1., markersize = mark_size,
                        color = colors[key])

        if not middle_point:
            continue

        axes[s_ax].plot(data['recall']['s'][middle_point], data['precision']['s'][middle_point],
                        '*', color = star_color, markersize = star_size)

    ticks = np.array([0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975])

    axes[p_ax].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[s_ax].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    axes[p_ax].legend()

    if shared_axis:
        axes[p_ax].set_title('P/S Waves')
    else:
        axes[p_ax].set_title('P-Wave')
        axes[s_ax].set_title('S-Wave')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)


if __name__ == '__main__':

    # Command line arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help = 'Path to dataset')
    parser.add_argument('--test-ratio', help = 'Amount of data to use for PRC building', type = float, default = 1.0)
    parser.add_argument('--points', help = 'How many threshold points to use in PRC plotting', default = 100)
    parser.add_argument('--favor', '-w', help = 'Use standard fast-attention model', action = 'store_true')
    parser.add_argument('--cnn', help = 'Use simple CNN model on top of spectrogram', action = 'store_true')
    parser.add_argument('--keras', help = 'Load keras model .json model, follow this option with path to model file')
    parser.add_argument('--weights', help = 'Path to model weights file')

    args = parser.parse_args()  # parse arguments

    # Load model
    if args.keras:
        model = load_keras_json_model(args.keras, args.weights)
    elif args.favor:

        import sys
        sys.path.append('../seismo-transformer/')
        sys.path.append('../seismo-transformer/utils')
        import seismo_load

        model = seismo_load.load_performer('C:/dev/seismo-transformer/WEIGHTS/w_model_performer_with_spec.hd5')

    elif args.cnn:

        import sys
        sys.path.append('../seismo-transformer/')
        sys.path.append('../seismo-transformer/utils')
        import seismo_load

        model = seismo_load.load_cnn('C:/dev/seismo-transformer/WEIGHTS/weights_model_cnn_spec.hd5')

    else:
        raise AttributeError('No model selected! Use --favor, --cnn or --keras options, or -h for help.')

    thresholds = np.linspace(0.01, 0.99, args.points)

    data_gpd = predict_data(model, 'C:/data/datasets/scsn_ps_2000_2017_shuf.hdf5', args.test_ratio)

    precision_recall = {
        'gpd': precision_recall_single(data_gpd, thresholds)
    }

    colors = {
        'gpd': '#2d38ad'
    }

    labels = {
        'gpd': 'GPD'
    }

    plot_prc_final(precision_recall, colors, labels, 'C:/dev/prc_saved.jpg')
