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
            hidden1 = tf.add(tf.matmul(x, weights['W1']), biases['b1'])
            hidden1 = tf.sigmoid(hidden1)
        # hidden layer2 with sigmoid function
        with tf.name_scope('hidden_layer2') as scope:
            hidden2 = tf.add(tf.matmul(hidden1, weights['W2']), biases['b2'])
            hidden2 = tf.sigmoid(hidden2)
        # output layer with softmax function
        with tf.name_scope('output_layer') as scope:
            out = tf.add(tf.matmul(hidden2, weights['W3']), biases['b3'])
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
            tf.random_normal(shape=[n_input, n_hidden1], stddev=0.1),
            name='weights'),
        'W2':
        tf.Variable(
            tf.random_normal(shape=[n_hidden1, n_hidden2], stddev=0.1),
            name='weights'),
        'W3':
        tf.Variable(
            tf.random_normal(shape=[n_hidden2, n_classes], stddev=0.1),
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
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # initailize variables
    init = tf.global_variables_initializer()

    # launch the graph
    training_epochs = 100
    batch_size = 100
    display_step = 10
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            '/tmp/MultiLayerPerceptron', graph=sess.graph)
        sess.run(init)
        summary_op = tf.summary.merge_all()

        for epoch in range(training_epochs):
            avg_cost = 0
            total_batch = mnist.train.images.shape[0] // batch_size
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                feed_dict = {x: batch_xs, y: batch_ys}
                _, c = sess.run(
                    [train_step, cross_entropy], feed_dict=feed_dict)
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch %04d" % (epoch + 1),
                      "cost = {:.9f}".format(avg_cost))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch)
        print("Training finished.")

        # eval for the test set
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy: ',
              accuracy.eval({
                  x: mnist.test.images,
                  y: mnist.test.labels
              }))


if __name__ == '__main__':
    main()
