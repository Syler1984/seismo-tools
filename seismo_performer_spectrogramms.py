import utils.fast_attention as models


if __name__ == '__main__':

    cnn = False

    weights = {
        'favor': 'utils/fast_attention/weights/w_model_cnn_spec.hd5',
        'cnn': 'utils/fast_attention/weights/w_model_performer_with_spec.hd5',
    }

    if cnn:
        model = models.load_cnn(weights['cnn'])
    else:
        model = models.load_performer(weights['favor'])
