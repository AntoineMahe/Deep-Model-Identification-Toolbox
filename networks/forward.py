#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

class Forward:
    def __init__(self, model_path, data_path, trajectory_length, history_length):
        self.model = tf.keras.models.load_model(model_path)
		self.train_mean = np.load(model_path/"means.npy")
		self.train_std = np.load(model_path/"std.npy")
        self.data = pd.read_hdf(data_path, key="train_X", mode='r').to_numpy()
        self.trajectory_length = trajectory_length  # 12
        self.history_length = history_length  # 20
        random.seed()

    def forward_pass(self, input_data):
        # Preprocess the input data if needed
        preprocessed_data = preprocess(input_data)

        # Perform the forward pass through the model
        output = self.model.predict(preprocessed_data)

        # Postprocess the output if needed
        postprocessed_output = postprocess(output)

        return postprocessed_output

    def plot_result(self):
        # Perform the forward pass
        input_data, expected_output = self.select_data()

        tmp_input_data = input_data[:self.history_length, :]
        output = input_data

        for i in range(trajectory_length):
            tmp_computed_output = self.forward_pass(tmp_input_data)
            tmp_input_data = np.append(tmp_input_data, tmp_computed_output, axis=0)

        # Plot the results
        fig, axs = plt.subplots(tmp_input_data.shape[1])
        for i in tmp_input_data.shape[1]:
            line1, = axs[i].plot(expected_output[:,i], label = 'expected output')
            line2, = axs[i].plot(tmp_input_data[:,i], label = 'result')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Forward Pass Result for axis {i}')
        plt.savefig(f"result_vs_data.pdf")
        # TODO mpf candle plot of both


    def remove_ts(self, data):
        """Remove the timestamp from data."""
        timestamp_idx = 0
        return data[:,[x for x in range(data.shape[1]) if x != timestamp_idx]]

	def preprocess(self, input_data):
        # data with timestamp (!).
        #for now our nn model use index not the tmestamp that need to be removed
        input_data = self.remove_ts(input_data)
        return (input_data - self.train_mean) / self.train_std

	def postprocess(self, output_data):
        return output_data * self.train_std + self.train_mean

    # FIXME pre and post process, quid of constant feature (std ~= 0)
    # for i in len(std):
    #   if std(i) < epsilon:
    #       # constant feature
    #       preprosseced = input_data - self.train_mean
    #   else # normal stuff

    def select_data(self):
        # select random starting point
        # return trajectory_length as expected ouptu and  history_length as input_data
        starting_point = random.randrange(self.data.shape[0] - self.trajectory_length - self.history_length)

        seleted_input = self.data[starting_point:starting_point + self.history_length, :]
        expected_output = self.data[:starting_point + self.trajectory_length + self.history_length, :]

        return seleted_input, expected_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                             prog='Test forward pass',
                             description='Compute forward path of a model and plot the result.',
                             epilog='We do one plot per dim.'
    )

    parser.add_argument('-m', '--model_path', help="path to the model folder", type=str, default='/model')
    parser.add_argument('-d', '--data', help="path to the data file", type=str, default='/data/validation.h5')
    parser.add_argument('-t', '--trajectory_length', help="length of the predicted trajectory", type=int, default=12)
    parser.add_argument('-l', '--history_length', help="length of the input sequence", type=int, default=20)  # TODO this should be read from the model.

    args = parser.parse_args()

    f = Forward(model_path=args.model_path, data_path=args.data, trajectory_length=args.trajectory_length, history_length=args.history_length)

    f.plot_result()

