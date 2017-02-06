#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def one_hot_encoding(target):
    n_classes = np.max(target) + 1
    one_hot = [
        list((np.arange(n_classes) == t).astype(np.int)) for t in target
    ]
    return np.array(one_hot)


def main():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, one_hot_encoding(digits.target))
    print('Train data: {}'.format(X_train.shape))
    print('Train labels: {}'.format(y_train.shape))
    print('Test data: {}'.format(X_test.shape))
    print('Test labels: {}'.format(y_test.shape))


if __name__ == '__main__':
    main()
