#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print('Train data: {}'.format(mnist.train.images.shape))
    print('Train labels: {}'.format(mnist.train.labels.shape))
    print('Test data: {}'.format(mnist.test.images.shape))
    print('Test labels: {}'.format(mnist.test.labels.shape))


if __name__ == '__main__':
    main()
