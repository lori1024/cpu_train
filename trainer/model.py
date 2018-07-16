
# !/usr/bin/env python

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


import tensorflow as tf
import task





# ***************************************************************************************
# YOU NEED TO MODIFY THIS FUNCTIONS IF YOU WANT TO IMPLEMENT A CUSTOM ESTIMATOR
# ***************************************************************************************


def create_estimator(config):
    """ Create a custom estimator based on _model_fn
    Args:
        config - used for model directory
    Returns:
        Estimator
    """

    feature_columns = [
        tf.feature_column.numeric_column(key="cpu-usage", shape=[2016])]


    linear_optimizer = tf.train.FtrlOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)
    dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=task.HYPER_PARAMS.learning_rate)

    estimator = tf.estimator.DNNRegressor(
        hidden_units=[200, 20],
        feature_columns=feature_columns,
        label_dimension=2,
        activation_fn=tf.nn.relu,
        dropout=task.HYPER_PARAMS.dropout_prob,
        config=config
    )

    #


    print("creating a dnn regressor model: {}".format(estimator))

    # estimator = tf.contrib.estimator.add_metrics(estimator, metric_fn)

    return estimator


# ***************************************************************************************
# YOU NEED NOT TO CHANGE THESE HELPER FUNCTIONS USED FOR CONSTRUCTING THE MODELS
# ***************************************************************************************


def construct_hidden_units():
    """ Create the number of hidden units in each layer
    if the HYPER_PARAMS.layer_sizes_scale_factor > 0 then it will use a "decay" mechanism
    to define the number of units in each layer. Otherwise, task.HYPER_PARAMS.hidden_units
    will be used as-is.
    Returns:
        list of int
    """
    hidden_units = list(map(int, task.HYPER_PARAMS.hidden_units.split(',')))

    if task.HYPER_PARAMS.layer_sizes_scale_factor > 0:
        first_layer_size = hidden_units[0]
        scale_factor = task.HYPER_PARAMS.layer_sizes_scale_factor
        num_layers = task.HYPER_PARAMS.num_layers

        hidden_units = [
            max(2, int(first_layer_size * scale_factor ** i))
            for i in range(num_layers)
        ]

    print("Hidden units structure: {}".format(hidden_units))

    return hidden_units


def update_learning_rate():
    """ Updates learning rate using an exponential decay method
    Returns:
       float - updated (decayed) learning rate
    """
    initial_learning_rate = task.HYPER_PARAMS.learning_rate
    decay_steps = task.HYPER_PARAMS.train_steps  # decay after each training step
    decay_factor = task.HYPER_PARAMS.learning_rate_decay_factor  # if set to 1, then no decay.

    global_step = tf.train.get_global_step()

    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_factor)

    return learning_rate



