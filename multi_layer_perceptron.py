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

    # network parameters
    n_input = mnist.train.images.shape[1]
    n_classes = mnist.train.labels.shape[1]
    n_hidden1 = (n_input + n_classes) // 3
    n_hidden2 = n_hidden1 // 2

    # input
    x = tf.placeholder(tf.float32, shape=(None, n_input))
    y = tf.placeholder(tf.float32, shape=(None, n_classes))

    # weights and biases
    weights = {
        'W1':
        tf.Variable(
            tf.random_normal(shape=(n_input, n_hidden1), stddev=0.1),
            name='weights'),
        'W2':
        tf.Variable(
            tf.random_normal(shape=(n_hidden1, n_hidden2), stddev=0.1),
            name='weights'),
        'W3':
        tf.Variable(
            tf.random_normal(shape=(n_hidden2, n_classes), stddev=0.1),
            name='weights')
    }
    biases = {
        'b1':
        tf.Variable(
            tf.random_normal(shape=[n_hidden1], stddev=0.1), name='biases'),
        'b2':
        tf.Variable(
            tf.random_normal(shape=[n_hidden2], stddev=0.1), name='biases'),
        'b3':
        tf.Variable(
            tf.random_normal(shape=[n_classes], stddev=0.1), name='biases')
    }

    # construct a model
    pred = MLP(x, weights, biases)

    # define loss funtion and optimizer
    learning_rate = 0.001
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred, y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # initailize variables
    init = tf.global_variables_initializer()


if __name__ == '__main__':
    main()
