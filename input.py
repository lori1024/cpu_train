#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import multiprocessing

import tensorflow as tf
from tensorflow import data

import task

import scipy.io as sio



# **************************************************************************
# YOU NEED NOT TO CHANGE THIS FUNCTION TO READ DATA FILES
# **************************************************************************


def generate_input_fn(file_names_pattern,
                     file_encoding='.mat',
                     mode=tf.estimator.ModeKeys.EVAL,
                     skip_header_lines=0,
                     num_epochs=1,
                     batch_size=200,
                     multi_threading=True):
    """Generates an input function for reading training and evaluation data file(s).
    This uses the tf.data APIs.
    Args:
        file_names_pattern: [str] - file name or file name patterns from which to read the data.
        mode: tf.estimator.ModeKeys - either TRAIN or EVAL.
            Used to determine whether or not to randomize the order of data.
        file_encoding: type of the text files. Can be 'csv' or 'tfrecords'
        skip_header_lines: int set to non-zero in order to skip header lines in CSV files.
        num_epochs: int - how many times through to read the data.
          If None will loop through data indefinitely
        batch_size: int - first dimension size of the Tensors returned by input_fn
        multi_threading: boolean - indicator to use multi-threading or not
    Returns:
        A function () -> (features, indices) where features is a dictionary of
          Tensors, and indices is a single Tensor of label indices.
    """
    def _input_fn():

        shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

        data_size = task.HYPER_PARAMS.train_size if mode == tf.estimator.ModeKeys.TRAIN else None

        num_threads = multiprocessing.cpu_count() if multi_threading else 1

        buffer_size = 2 * batch_size + 1

        print("")
        print("* data input_fn:")
        print("================")
        print("Mode: {}".format(mode))
        print("Input file(s): {}".format(file_names_pattern))
        print("Files encoding: {}".format(file_encoding))
        print("Data size: {}".format(data_size))
        print("Batch size: {}".format(batch_size))
        print("Epoch count: {}".format(num_epochs))
        print("Thread count: {}".format(num_threads))
        print("Shuffle: {}".format(shuffle))
        print("================")
        print("")

        # file_names = tf.matching_files(file_names_pattern)
        Data = sio.loadmat(open(file_names_pattern[0], "rb"))

        # dataset = data.TFRecordDataset(filenames=file_names_pattern[0])

        data = Data['trainInput']
        label = Data['trainLabel']
        feature = {'cpu-usage': data}
        dataset = (tf.data.Dataset)
        feature_columns = [
            tf.feature_column.numeric_column(key="cpu-usage", shape=[2016])]


        # dataset = dataset.map(lambda feature_columns, target: (feature, label),
        #                       num_parallel_calls=num_threads)
        dataset = tf.data.Dataset.from_tensor_slices((feature, label))
        # if shuffle:
        #     dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()
        features, target = iterator.get_next()

        return features, target

    return _input_fn







