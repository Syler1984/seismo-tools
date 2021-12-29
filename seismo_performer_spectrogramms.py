import argparse
import utils.fast_attention as models


if __name__ == '__main__':

    print('Parsing arguments..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn', help='Use cnn model', action='store_true')
    args = parser.parse_args()

    weights = {
        'favor': 'utils/fast_attention/weights/w_model_performer_with_spec.hd5',
        'cnn': 'utils/fast_attention/weights/w_model_cnn_spec.hd5',
    }

    if args.cnn:
        print('Loading CNN model..')
        model = models.load_cnn(weights['cnn'])
    else:
        print('Loading Seismo-Performer..')
        model = models.load_performer(weights['favor'])

    model.summary()
