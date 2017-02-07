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

    def MLP(x, weights, biases):
        # hidden layer1 with sigmoid function
        with tf.name_scope('hidden_layer1') as scope:
            hidden1 = tf.add(tf.mul(x, weights['W1']), biases['b1'])
            hidden1 = tf.sigmoid(hidden1)
        # hidden layer2 with sigmoid function
        with tf.name_scope('hidden_layer2') as scope:
            hidden2 = tf.add(tf.mul(hidden1, weights['W2']), biases['b2'])
            hidden2 = tf.sigmoid(hidden2)
        # output layer with softmax function
        with tf.name_scope('output_layer') as scope:
            out = tf.add(tf.mul(hidden2, weights['W3']), biases['b3'])
            out = tf.nn.softmax(out)
        return out


if __name__ == '__main__':
    main()
