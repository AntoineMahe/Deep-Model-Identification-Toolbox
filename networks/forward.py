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

        # Create a TensorFlow session
        self.sess = tf.Session()

        model_graph = model_path + "model_ckpt/final_NN"
        # Load the model from the checkpoint
        self.saver = tf.train.import_meta_graph(model_graph + '.meta')
        self.saver.restore(self.sess, model_graph)

        # Get the input and output tensors of the model
        self.graph = tf.get_default_graph()
        self.input_tensor = self.graph.get_tensor_by_name('inputs:0')  # Replace 'input_tensor_name' with the actual name of your input tensor
        # self.output_tensor = self.graph.get_tensor_by_name('target:0')  # Replace 'output_tensor_name' with the actual name of your output tensor
        self.output_tensor = self.graph.get_tensor_by_name('time_distributed_7/Reshape_1:0')  # Replace 'output_tensor_name' with the actual name of your output tensor


        # self.model = tf.keras.models.load_model(model_path)

        self.train_mean = np.load(model_path + "means.npy")
        self.train_std = np.load(model_path + "std.npy")
        self.data = pd.read_hdf(data_path, key="train_X", mode='r').to_numpy()

        self.trajectory_length = trajectory_length  # 12
        self.history_length = history_length  # 20
        random.seed()

    def forward_pass(self, input_data):
        # Preprocess the input data if needed
        preprocessed_data = np.empty((1,20,5))
        preprocessed_data[0] = self.preprocess(input_data)

        print(f"input : {preprocessed_data}")
        print(f"input.shape : {preprocessed_data.shape}")
        # Perform the forward pass through the model
        # output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: preprocessed_data, "keep_prob:0":1.0, "self.M.weights:0":np.ones(preprocessed_data.shape[0]), "self.istraining":False,"self.step":0 })
        output = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: preprocessed_data, "keep_prob:0":1.0, "is_training:0":False, "step:0":0})

        # output = self.model.predict(preprocessed_data)

        # Postprocess the output if needed
        print(f" output : {output}")
        print(f" output.shape : {output.shape}")
        postprocessed_output = self.postprocess(output[:,-1,:])
        print(f" postprocess output.shape : {postprocessed_output.shape}")

        return postprocessed_output

    def plot_result(self):
        # Perform the forward pass
        input_data, expected_output = self.select_data()

        tmp_input_data = input_data
        output = input_data

        for i in range(self.trajectory_length):
            tmp_computed_output = self.forward_pass(tmp_input_data[-self.history_length:, :])
            tmp_input_data = np.append(tmp_input_data, tmp_computed_output, axis=0)

        # Plot the results
        fig, axs = plt.subplots(tmp_input_data.shape[1])
        for i in range(tmp_input_data.shape[1]):
            line1, = axs[i].plot(expected_output[:self.history_length+self.trajectory_length,i], label = 'expected output')
            line2, = axs[i].plot(tmp_input_data[:,i], label = 'result')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'Forward Pass Result for axis {i}')
        plt.savefig(f"/result/result_vs_data.pdf")
        # TODO mpf candle plot of both


    def remove_ts(self, data):
        """Remove the timestamp from data."""
        timestamp_idx = 0
        return data[:,[x for x in range(data.shape[1]) if x != timestamp_idx]]

    def preprocess(self, input_data):
        # data with timestamp (!).
        #for now our nn model use index not the tmestamp that need to be removed
        # input_data = self.remove_ts(input_data)
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
        expected_output = self.data[starting_point:starting_point + self.trajectory_length + self.history_length, :]

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

