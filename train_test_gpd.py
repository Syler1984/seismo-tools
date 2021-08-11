import argparse
from time import time

from h5_generator import train_test_split


def gpd_fixed(n_samples = 400, n_channels = 3, n_classes = 3, flatten = True):

    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(n_samples, n_channels))

    # CNN
    n_filters = [32, 64, 128, 256]
    s_kernels = [21, 15, 11, 9]

    x = inputs
    for n_filter, s_kernel in zip(n_filters, s_kernels):

        x = layers.Conv1D(filters = n_filter, kernel_size = s_kernel, padding = 'same', activation = None)(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

    # Concatenate to a single vector
    if flatten:
        x = layers.Flatten()(x)
    else:
        x = layers.GlobalAveragePooling1D()(x)

    # FCNN
    for _ in range(2):
        x = layers.Dense(200, activation = None)(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)

    outputs = layers.Dense(n_classes, activation = 'softmax')(x)

    return keras.Model(inputs, outputs)


def gpd_ross(model_path):

    import tensorflow as tf
    from keras.models import model_from_json

    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json, custom_objects = {"tf": tf})

    return model


def train_test_model(loader, i, train_set, test_sets, callbacks, keras, result_flag,
                     epochs = 300, lr = 0.0001, test_batch_size = 32):

    model = loader()

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = lr),
                  loss = keras.losses.SparseCategoricalCrossentropy(),
                  metrics = [keras.metrics.SparseCategoricalAccuracy()])

    # Load train data
    X_train, X_test = train_test_split(train_set['path'],
                                       batch_size = train_set['batch-size'],
                                       shuffle = False,
                                       test_size = train_set['test-size'])

    # Train
    start = time()
    history = model.fit(X_train,
                        validation_data = (X_test),
                        epochs = epochs,
                        callbacks = callbacks)
    t_delta = time() - start

    model.save_weights(f'gpd_{result_flag}_{train_set["name"]}.h5')

    # Print results
    string = f'{i} ({t_delta} s): '
    for metric in model.metrics_names:
        metric = 'val_' + metric
        string += f'{metric} = {history.history[metric][-1]} '
    string += '\n'

    with open(f'gpd_{result_flag}_{train_set["name"]}_train_test.txt', 'a') as f:
        f.write(string)

    # Test
    for name, path in test_sets.items():

        _, X_test = train_test_split(path,
                                     batch_size = test_batch_size,
                                     shuffle = False,
                                     test_size = 1.)

        start = time()
        results = model.evaluate(X_test)
        t_delta = time() - start

        string = f'{i} ({t_delta} s): '
        for metric, value in zip(model.metrics_names, results):
            metric = 'val_' + metric
            string += f'{metric} = {value} '

        string += '\n'

        with open(f'gpd_{result_flag}_{name}_test.txt', 'a') as f:
            f.write(string)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpd', help = 'Use original GPD model', action = 'store_true')
    parser.add_argument('--gpd-fixed', help = 'Use GPD-Fixed model', action = 'store_true')
    parser.add_argument('--train-cali', help = 'Train on South-California data', action = 'store_true')
    parser.add_argument('--train-sak', help = 'Train on Sakhalin data', action = 'store_true')
    parser.add_argument('--train-dag', help = 'Train on Dagestan data', action = 'store_true')
    args = parser.parse_args()

    # Load model
    default_gpd_model = 'generalized-phase-detection/model_pol.json'

    if args.gpd:

        import keras
        flag = 'ross_'
        result_flag = 'ross'

        def load_model():
            return gpd_ross(default_gpd_model)

    elif args.gpd_fixed:

        from tensorflow import keras
        flag = 'fixed_'
        result_flag = 'fixed'

        def load_model():
            return gpd_fixed()

    else:
        raise AttributeError('No model specified: use --gpd or --gpd-fixed flag')

    # Prepare datasets info
    datasets = {
        'cali': '/path/to/cali',
        'sak': '/path/to/sak',
        'dag': '/path/to/dag'
    }

    if args.train_cali:
        train_set = {
            'name': 'cali',
            'batch-size': 128,
            'test-size': 0.2
        }
    elif args.train_sak:
        train_set = {
            'name': 'sak',
            'batch-size': 128,
            'test-size': 0.2
        }
    elif args.train_dag:
        train_set = {
            'name': 'dag',
            'batch-size': 128,
            'test-size': 0.2
        }
    else:
        raise AttributeError('No training set specified: use --train-cali, --train-sak or --train-dag flag')

    train_set['path'] = datasets[train_set['name']]

    flag += train_set['name']

    test_sets = {}
    for name, path in datasets.items():
        if name != train_set['name']:
            test_sets[name] = path

    # Compile the model
    model_checkpoint_name = 'w_gpd_' + flag + '_{epoch}.h5'
    callbacks = [
        keras.callbacks.EarlyStopping(monitor = 'val_loss',
                                      patience = 3, min_delta = 0.001,
                                      restore_best_weights = True, verbose = 1),
        keras.callbacks.ModelCheckpoint(model_checkpoint_name, save_weights_only=True),
    ]

    # Training and testing
    reps = 10
    for i in range(reps):
        train_test_model(load_model, i, train_set, test_sets, callbacks, keras, result_flag)
