"""
Benchmark of CIFAR10 for myia and tensorflow
by LittleCats, 2019
"""

import sys
import argparse
import random as rd
import numpy as np
import cifar10 as cf
from models.tf import MiniVGG as tf_vgg
from models.myia import MiniVGG as myia_vgg

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('bigcat')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

handler = logging.FileHandler("bigcat.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

params_tf_vgg = {'lr':0.003, 'dropout':0.1, 'name':'tf_vgg'}
params_myia_vgg = {'lr':0.003, 'dropout':0.2, 'name':'myia_vgg'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='tf_vgg',
            help='the model to run')
    parser.add_argument('-t', '--train', action='store_true',
            help='train to model? [default: True]')
    parser.add_argument('--num_epochs', type=int, default=20,
            help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
            help='batch size')
    parser.add_argument('--log_interval', type=int, default=20,
            help='log interval')
    parser.add_argument('--data_dir', default='./data/',
            help='directory of data')
    parser.add_argument('--model_dir', default='./saved_models/',
            help='directory of saved models')
    parser.add_argument('-v', '--verbose', action='count', default='store_true',
            help='verbosity')
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNNING)

    if args.model == 'tf_vgg':
        model = tf_vgg(**params_tf_vgg)
    elif args.model == 'myia_vgg':
        model = myia_vgg(**params_myia_vgg)
    else:
        raise Exception('not supported model')

    # load and preprocess the cifar data
    data_train, data_test = cf.load_data(args.data_dir)
    logger.info("Run with the parameters: %s" %(args))

    # train the model
    if args.train:
        x_train, y_train = data_train
        x_train = x_train.astype('float32')
        y_train = y_train.astype('int64')
        x_train /= 255.
        num_samples = len(x_train)
        batch_size = args.batch_size
        num_batches = num_samples // batch_size
        if num_samples % batch_size != 0:
            num_batches += 1
        step = 0
        idx = np.arange(num_samples)
        for epoch in range(args.num_epochs):
            rd.shuffle(idx)
            for bt in range(num_batches):
                step += 1
                batch_idx = idx[bt*batch_size : min((bt+1)*batch_size, num_samples)]
                batch_x = x_train[batch_idx]
                batch_y = y_train[batch_idx]
                l = model.update(batch_x, batch_y)
                if step % args.log_interval == 0:
                    logger.info('step %d, training loss: %f' %(step, l))
            logger.info('epoch %d, training loss: %f' %(epoch, l))
    else:
        model.load(args.model_dir)

    # evaluate the model
    x_test, y_test = data_test
    x_test = x_test.astype('float32')
    y_test = y_test.astype('int64')
    x_test /= 255.
    loss, acc = model.evaluate(x_test, y_test)
    logger.info('test accuracy: %f, test loss: %f', acc, loss)

    # save the model
    if args.train:
        model.save(args.model_dir)

if __name__ == '__main__':
    """Usage: python run_main.py -d './data/'
    """
    sys.exit(main())

