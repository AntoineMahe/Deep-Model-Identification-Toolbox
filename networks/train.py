import sys
import math
import os
import datetime
import tensorflow as tf
import numpy as np
import samplers
import readers
import models
import settings
import network_generator

#TODO fix depecrated for full binding with TensorFlow 1.14.0

class Training_Uniform:
    """
    Training container in its most basic form. Used to train and evaluate the
    neural networks performances.
    """
    def __init__(self, Settings):
        # Setting object
        self.sts = Settings
        # Dataset object
        self.DS = None
        self.load_dataset()
        # Samplers
        self.SR = None
        self.load_sampler()
        # Model
        self.M = None
        self.load_model()
        # Train
        self.init_records()
        self.train()

    def load_dataset(self):
        """
        Instantiate the dataset-reader object.
        """
        self.DS = readers.H5Reader(self.sts)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.UniformSampler(self.DS, self.sts)

    def load_model(self):
        """
        Calls the network generator to load/generate the requested model.
        Please note that the generation of models is still experimental
        and may change frequently. For more information have look at the
        network_generator.
        """
        self.M = network_generator.get_graph(self.sts)

    def init_records(self):
        """
        Creates empty list to save the output of the network
        """
        self.train_logs = []
        self.test_logs = []
        self.test_logs_multi_step = []
        self.best_1s = np.inf
        self.best_ms = np.inf
        self.lw_ms = np.inf
        self.start_time = datetime.datetime.now()
        self.forward_time = []
        self.backward_time = []

    def saver_init_and_restore(self):
        """
        Creates a saver object to save the model as we train. Allows to
        load pre-trained models for fine tuning, and instantiate the
        networks variables.
        """
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(os.path.join(self.sts.tb_log_name,'train'), self.sess.graph)
        self.test_writer = tf.summary.FileWriter(os.path.join(self.sts.tb_log_name,'test'))
        if self.sts.restore:
            self.saver.restore(self.sess, self.sts.path_weight)

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        Input:
            i : the current step (int)
        """
        prct, batch_xs, batch_ys = self.SR.sample_train_batch()
        ts = datetime.datetime.now()
        _ = self.sess.run(self.M.train_step, feed_dict = {self.M.x: batch_xs,
                                                          self.M.y: batch_ys,
                                                          self.M.weights: np.ones(self.sts.batch_size),
                                                          self.M.drop_rate: self.sts.dropout,
                                                          self.M.step: i,
                                                          self.M.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

    def eval_on_train(self, i):
        """
        Evaluation Step: Samples a new batch and perform forward pass. The sampler here
        is independant from the training one. Also logs information about training
        performance in a list.
        Input:
            i : the current step (int)
        """
        prct, batch_xs, batch_ys = self.SR.sample_eval_train_batch()
        # Computes accuracy and loss + acquires summaries
        ts = datetime.datetime.now()
        acc, loss, summaries = self.sess.run([self.M.acc_op, self.M.s_loss, self.M.merged],
                                        feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.drop_rate: self.sts.dropout,
                                                     self.M.step: i,
                                                     self.M.is_training: False})
        self.forward_time.append(datetime.datetime.now() - ts)
        # Update hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.train_logs.append([i] + [datetime.datetime.now()] + list(acc))
        # Write tensorboard logs
        self.train_writer.add_summary(summaries, i)

    def eval_on_validation_single_step(self, i):
        """
        Evaluation Step: Samples a new batch out of the validation set and performs
        a forward pass. Also logs the performance about the evaluation set. Saves
        the model if it performed better than ever before.
        Input:
            i : the current step (int)
        """
        prct, batch_xs , batch_ys = self.SR.sample_val_batch()
        # Computes accuracy and loss + acquires summaries
        acc, loss, summaries = self.sess.run([self.M.acc_op, self.M.s_loss, self.M.merged],
                                        feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.drop_rate: self.sts.dropout,
                                                     self.M.step: i,
                                                     self.M.is_training: False})
        # Update Single-Step hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.test_logs.append([i] + [datetime.datetime.now()]+list(acc))
        # Update inner variables and saves best model weights
        avg = np.mean(acc)
        if  avg < self.best_1s:
            self.best_1s = avg
            NN_save_name = os.path.join(self.sts.model_ckpt,'Best_1S')
            self.saver.save(self.sess, NN_save_name)
        # Write tensorboard logs
        self.test_writer.add_summary(summaries, i)
        # Return accuracy for console display
        return acc

    def run_test_single_step(self):
        """
        Test Step: runs the model on the whole of the test set and performs
        a forward pass.
        """
        try:
            prct, batch_xs , batch_ys = self.SR.sample_test_batch()
        except:
            batch_xs , batch_ys = self.SR.sample_test_batch()
        # Computes accuracy and loss + acquires summaries
        y_ = self.sess.run([self.M.y_], feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.drop_rate: self.sts.dropout,
                                                     self.M.step: 0,
                                                     self.M.is_training: False})
        # Compute RMSE on test set
        self.test_ss_RMSE = np.sqrt(np.mean((np.squeeze(batch_ys) - y_[0])**2,axis=0))

    def run_test_on_multi_step(self): # TODO replace i by train_step
        """
        Test on Trajectories: Samples a new batch of trajectories to
        evaluate the network on. Performs a forward pass and logs the
        performance of the model on multistep predictions.
        """
        predictions = []
        # Sample trajectory batch out of the test set
        batch_x, batch_y = self.SR.sample_test_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            pred = self.get_predictions(full, 0)
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            if self.sts.cmd_dim > 0:
                cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
                new = np.concatenate((pred, cmd), axis=1)
            else:
                new = pred
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((old,new), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        # Compute error
        self.test_ms_RMSE = np.sqrt(np.mean((predictions[:,:,:] - batch_y[:,:-1,:])**2,axis=(0,1)))
        per_traj_RMSE = np.sqrt(np.mean((predictions[:,:,:] - batch_y[:,:-1,:])**2,axis=1))
        self.test_ms_STD = np.std(per_traj_RMSE,axis=0)

    def get_predictions(self, full, i):
        """
        Get the predictions of the network.
        Input:
            full: a batch of inputs in the form [Batch_size x Sequence_size x Input_size]
        Output:
            pred: the predictions associated with those batches in the
                  form [Batch_size x Output_dim]
        """
        pred = self.sess.run(self.M.y_, feed_dict = {self.M.x: full,
                                                     self.M.drop_rate:self.sts.dropout,
                                                     self.M.weights: np.ones(full.shape[0]),
                                                     self.M.is_training: False,
                                                     self.M.step: i})
        return pred

    def eval_multistep(self, predictions, batch_y, i):
        """
        Input:
            predictions: a batch of predictions in the form [Batch_size x Trajectory_size - Sequence_size x Output_size]
            batch_y: a batch of groundtruth in the form [Batch_size x Trajectory_size - Sequence_size x Output_size]
        Output:

        """
        # Compute error
        error_x = (predictions[:,:,:] - batch_y[:,:-1,:])**2
        std_x = np.std(np.mean(error_x,axis=0), axis=1)
        error_x = np.sqrt(np.mean(error_x, axis=(0,1)))
        worse = np.max(error_x)
        avg = np.mean(error_x)
        # Update multistep hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.test_logs_multi_step.append([i] + [datetime.datetime.now()] + list(error_x) + [worse])
        # Update inner variable
        if avg < self.best_ms:
            self.best_ms = avg
            NN_save_name = os.path.join(self.sts.model_ckpt,'Best_MS')
            self.saver.save(self.sess, NN_save_name)
        if worse < self.lw_ms:
            self.lw_ms = worse
            NN_save_name = os.path.join(self.sts.model_ckpt,'Least_Worse_MS')
            self.saver.save(self.sess, NN_save_name)
        return error_x, worse

    def eval_on_validation_multi_step(self, i): # TODO replace i by train_step
        """
        Evaluation Step on Trajectories: Samples a new batch of trajectories to
        evaluate the network on. Performs a forward pass and logs the
        performance of the model on multistep predictions. If the models performed
        better than ever before then save model.
        Input:
            i : the current step (int)
        """
        predictions = []
        # Sample trajectory batch out of the validation set
        batch_x, batch_y = self.SR.sample_val_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            pred = self.get_predictions(full, i)
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            if self.sts.cmd_dim > 0:
                cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
                new = np.concatenate((pred, cmd), axis=1)
            else:
                new = pred
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((old,new), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        # Compute per variable error
        return self.eval_multistep(predictions, batch_y, i)

    def display(self, i, acc, worse, ms_acc):
        """
        Prints the current training status in the terminal
        Input:
            i: the current step (int)
            acc: the accuracy in the form [Output_size]
            worse: the worse accuracy (float)
            ms_acc: the multistep accuracy [Output_size]
        """
        print('Step: ', str(i), ', 1s acc:', str(acc), ', ', str(self.sts.trajectory_length),
                       's worse acc: ', str(worse), ', ',
                       str(self.sts.trajectory_length), 's avg acc: ', str(ms_acc))

    def get_model_parameters(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def dump_logs(self):
        """
        Saves the logs of the training in npy format
        """
        # Save model weights at the end of training
        NN_save_name = os.path.join(self.sts.model_ckpt,'final_NN')
        self.saver.save(self.sess, NN_save_name)
        # Display training statistics
        print('#### TRAINING DONE ! ####')
        print('Best single-step-Accuracy reach for: ' + str(self.best_1s))
        print('Best multi-steps-Accuracy reach for: ' + str(self.best_ms))
        print('Least Worse Accuracy reach for: ' + str(self.lw_ms))

        # Write hard-logs as numpy arrays
        np.save(self.sts.output_dir + "/train_loss_log.npy", np.array(self.train_logs))
        np.save(self.sts.output_dir + "/test_single_step_loss_log.npy", np.array(self.test_logs))
        np.save(self.sts.output_dir + "/test_multi_step_loss_log.npy", np.array(self.test_logs_multi_step))
        np.save(self.sts.output_dir + "/means.npy", self.DS.mean)
        np.save(self.sts.output_dir + "/std.npy", self.DS.std)

        # Write networks statistics
        with open(os.path.join(self.sts.output_dir,'statistics.txt'), 'w') as stats:
            stats.write('network parameters: ' + str(self.get_model_parameters()) + os.linesep)
            stats.write('network forward time: ' + str(np.mean(self.forward_time)) + ' batch_size (' + str(self.sts.val_batch_size) + ')' + os.linesep)
            stats.write('network backward time: ' + str(np.mean(self.backward_time)) + ' batch_size (' + str(self.sts.batch_size) + ')' + os.linesep)
            stats.write('VAL best single-step-Accuracy reach for: ' + str(self.best_1s) + os.linesep)
            stats.write('VAL best multi-steps-Accuracy reach for: ' + str(self.best_ms) + os.linesep)
            stats.write('VAL least Worse Accuracy reach for: ' + str(self.lw_ms) + os.linesep)
            stats.write('TEST single step RMSE: '+str(self.test_ss_RMSE) + os.linesep)
            stats.write('TEST multi step RMSE: '+str(self.test_ms_RMSE) + os.linesep)
            stats.write('TEST multi step STD: '+str(self.test_ms_STD) + os.linesep)

    def train(self):
        """
        The training loop
        """
        with tf.Session() as self.sess:
            self.saver_init_and_restore()
            for i in range(self.sts.max_iterations):
                self.train_step(i)
                if i%10 == 0:
                    self.eval_on_train(i)
                if i%self.sts.log_frequency == 0:
                    acc = self.eval_on_validation_single_step(i)
                    acc_t, worse = self.eval_on_validation_multi_step(i)
                if i%250 == 0:
                    self.display(i, acc, worse, acc_t)
            self.run_test_single_step()
            self.run_test_on_multi_step()
            self.dump_logs()

class Training_RNN_Seq2Seq(Training_Uniform):
    """
    Training container with support seq2seq processing.
    Used to train and evaluate the neural networks performances.
    """
    def __init__(self, Settings):
        super(Training_RNN_Seq2Seq, self).__init__(Settings)

    def load_dataset(self):
        """
        Instantiate the dataset-reader object based on user inputs.
        See the settings object for more information.
        """
        self.DS = readers.H5Reader_Seq2Seq(self.sts)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.UniformSampler(self.DS, self.sts)

    def load_model(self):
        """
        Calls the network generator to load/generate the requested model.
        Please note that the generation of models is still experimental
        and may change frequently. For more information have look at the
        network_generator.
        """
        self.M = network_generator.get_graph(self.sts)
        self.train_hs = self.M.get_hidden_state(self.sts.batch_size)
        self.train_val_hs = self.M.get_hidden_state(self.sts.val_batch_size)
        self.val_hs = self.M.get_hidden_state(self.sts.val_batch_size)
        self.val_traj_hs = self.M.get_hidden_state(self.sts.val_traj_batch_size)
        if self.sts.test_batch_size is None:
            self.test_hs = self.M.get_hidden_state(self.DS.test_x.shape[0])
            self.test_traj_hs = self.M.get_hidden_state(self.DS.test_traj_x.shape[0])
        else:
            self.test_hs = self.M.get_hidden_state(self.sts.test_batch_size)
            self.test_traj_hs = self.M.get_hidden_state(self.sts.test_traj_batch_size)

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        Input:
            i : the current step (int)
        """
        ts = datetime.datetime.now()
        prct, batch_x, batch_y = self.SR.sample_train_batch()
        _ = self.sess.run([self.M.train_step],
                                          feed_dict = {self.M.x: batch_x,
                                                       self.M.y: batch_y,
                                                       self.M.hs: self.train_hs,
                                                       self.M.weights: np.ones(self.sts.batch_size),
                                                       self.M.drop_rate: self.sts.dropout,
                                                       self.M.step: i,
                                                       self.M.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

    def eval_on_train(self, i):
        """
        Evaluation Step: Samples a new batch and perform forward pass. The sampler here
        is independant from the training one. Also logs information about training
        performance in a list.
        Input:
            i : the current step (int)
        """
        prct, batch_xs, batch_ys = self.SR.sample_eval_train_batch()
        # Computes accuracy and loss + acquires summaries
        ts = datetime.datetime.now()
        acc, loss, summaries = self.sess.run([self.M.acc_op, self.M.s_loss,
                                              self.M.merged],
                                              feed_dict = {self.M.x: batch_xs,
                                                           self.M.y: batch_ys,
                                                           self.M.hs: self.train_val_hs,
                                                           self.M.weights: np.ones(batch_xs.shape[0]),
                                                           self.M.drop_rate: self.sts.dropout,
                                                           self.M.step: i,
                                                           self.M.is_training: False})
        self.forward_time.append(datetime.datetime.now() - ts)
        # Update hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.train_logs.append([i] + [datetime.datetime.now()] + list(acc))
        # Write tensorboard logs
        self.train_writer.add_summary(summaries, i)

    def eval_on_validation_single_step(self, i):
        """
        Evaluation Step: Samples a new batch out of the validation set and performs
        a forward pass. Also logs the performance about the evaluation set. Saves
        the model if it performed better than ever before.
        Input:
            i : the current step (int)
        """
        prct, batch_xs , batch_ys = self.SR.sample_val_batch()
        # Computes accuracy and loss + acquires summaries
        acc, loss, summaries = self.sess.run([self.M.acc_op, self.M.s_loss,
                                              self.M.merged],
                                              feed_dict = {self.M.x: batch_xs,
                                                           self.M.y: batch_ys,
                                                           self.M.hs: self.val_hs,
                                                           self.M.weights: np.ones(batch_xs.shape[0]),
                                                           self.M.drop_rate: self.sts.dropout,
                                                           self.M.step: i,
                                                           self.M.is_training: False})
        # Update Single-Step hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.test_logs.append([i] + [datetime.datetime.now()]+list(acc))
        # Update inner variables and saves best model weights
        avg = np.mean(acc)
        if  avg < self.best_1s:
            self.best_1s = avg
            NN_save_name = os.path.join(self.sts.model_ckpt,'Best_1S')
            self.saver.save(self.sess, NN_save_name)
        # Write tensorboard logs
        self.test_writer.add_summary(summaries, i)
        # Return accuracy for console display
        return acc


    def get_predictions(self, full, hs, i):
        """
        Get the predictions of the network.
        Input:
            full: a batch of inputs in the form [Batch_size x Sequence_size x Input_size]
        Output:
            pred: the predictions associated with those batches in the
                  form [Batch_size x Output_dim]
        """
        pred = self.sess.run(self.M.y_,
                                 feed_dict = {self.M.x: full,
                                              self.M.hs: hs,
                                              self.M.drop_rate:self.sts.dropout,
                                              self.M.weights: np.ones(full.shape[0]),
                                              self.M.is_training: False,
                                              self.M.step: i})
        return pred

    def eval_on_validation_multi_step(self, i): # TODO replace i by train_step
        """
        Evaluation Step on Trajectories: Samples a new batch of trajectories to
        evaluate the network on. Performs a forward pass and logs the
        performance of the model on multistep predictions. If the models performed
        better than ever before then save model.
        Input:
            i : the current step (int)
        """
        predictions = []
        # Sample trajectory batch out of the evaluation set
        batch_x, batch_y = self.SR.sample_val_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        hs = self.val_traj_hs
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            pred = self.get_predictions(full, hs, i)
            pred = pred[:,-1,:]
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            if self.sts.cmd_dim > 0:
                cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
                new = np.concatenate((pred, cmd), axis=1)
            else:
                new = pred
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((old,new), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        return self.eval_multistep(predictions, batch_y, i)

    def run_test_on_multi_step(self): # TODO replace i by train_step
        """
        Test on Trajectories: Samples a new batch of trajectories to
        evaluate the network on. Performs a forward pass and logs the
        performance of the model on multistep predictions.
        """
        predictions = []
        # Sample trajectory batch out of the test set
        batch_x, batch_y = self.SR.sample_test_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        hs = self.test_traj_hs
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            pred = self.get_predictions(full, hs, 0)
            pred = pred[:,-1,:]
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            if self.sts.cmd_dim > 0:
                cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
                new = np.concatenate((pred, cmd), axis=1)
            else:
                new = pred
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((old,new), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        # Compute error
        self.test_ms_RMSE = np.sqrt(np.mean((predictions[:,:,:] - batch_y[:,:-1,:])**2,axis=(0,1)))
        per_traj_RMSE = np.sqrt(np.mean((predictions[:,:,:] - batch_y[:,:-1,:])**2,axis=1))
        self.test_ms_STD = np.std(per_traj_RMSE,axis=0)

    def run_test_single_step(self):
        """
        Test Step: runs the model on the whole of the test set and performs
        a forward pass.
        """
        try:
            prct, batch_xs , batch_ys = self.SR.sample_test_batch()
        except:
            batch_xs , batch_ys = self.SR.sample_test_batch()
        # Computes accuracy and loss + acquires summaries
        y_ = self.sess.run([self.M.y_], feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.hs: self.test_hs,
                                                     self.M.drop_rate: self.sts.dropout,
                                                     self.M.step: 0,
                                                     self.M.is_training: False})
        # Compute RMSE on test set
        batch_ys = batch_ys[:,-1,:]
        self.test_ss_RMSE = np.sqrt(np.mean((batch_ys - y_[0][:,-1,:])**2,axis=0))


class Training_RNN_Continuous_Seq2Seq(Training_RNN_Seq2Seq):
    """
    Training container with support for continuous time seq2seq processing.
    Used to train and evaluate the neural networks performances.
    """
    def __init__(self, Settings):
        super(Training_RNN_Continuous_Seq2Seq, self).__init__(Settings)

    def load_dataset(self):
        """
        Instantiate the dataset-reader object based on user inputs.
        See the settings object for more information.
        """
        self.DS = readers.H5Reader_Seq2Seq_RNN(self.sts)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.RNNSampler(self.DS, self.sts)

    def load_model(self):
        """
        Calls the network generator to load/generate the requested model.
        Please note that the generation of models is still experimental
        and may change frequently. For more information have look at the
        network_generator.
        """
        self.M = network_generator.get_graph(self.sts)
        self.train_hs = self.M.get_hidden_state(self.sts.batch_size)
        self.train_val_hs = self.M.get_hidden_state(self.sts.val_batch_size)
        self.val_hs = self.M.get_hidden_state(self.sts.val_batch_size)
        self.val_traj_hs = self.M.get_hidden_state(self.sts.val_traj_batch_size)
        if self.sts.test_batch_size is None:
            self.test_hs = self.M.get_hidden_state(self.DS.test_x.shape[0])
            self.test_traj_hs = self.M.get_hidden_state(self.DS.test_traj_x.shape[0])
        else:
            self.test_hs = self.M.get_hidden_state(self.sts.test_batch_size)
            self.test_traj_hs = self.M.get_hidden_state(self.sts.test_traj_batch_size)

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        Input:
            i : the current step (int)
        """
        prct, batch_x, batch_y, continuity = self.SR.sample_train_batch()
        ts = datetime.datetime.now()
        self.train_hs = np.swapaxes(np.swapaxes(self.train_hs,-2,-1)*continuity,-2,-1)
        _, self.train_hs = self.sess.run([self.M.train_step, self.M.current_state],
                                          feed_dict = {self.M.x: batch_x,
                                                       self.M.y: batch_y,
                                                       self.M.hs: self.train_hs,
                                                       self.M.weights: np.ones(self.sts.batch_size),
                                                       self.M.drop_rate: self.sts.dropout,
                                                       self.M.step: i,
                                                       self.M.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

    def eval_on_train(self, i):
        """
        Evaluation Step: Samples a new batch and perform forward pass. The sampler here
        is independant from the training one. Also logs information about training
        performance in a list.
        Input:
            i : the current step (int)
        """
        prct, batch_xs, batch_ys, continuity = self.SR.sample_eval_train_batch()
        self.train_val_hs = np.swapaxes(np.swapaxes(self.train_val_hs,-2,-1)*continuity,-2,-1)
        # Computes accuracy and loss + acquires summaries
        ts = datetime.datetime.now()
        acc, loss, summaries, self.train_val_hs = self.sess.run([self.M.acc_op, self.M.s_loss,
                                                                 self.M.merged, self.M.current_state],
                                                                 feed_dict = {self.M.x: batch_xs,
                                                                              self.M.y: batch_ys,
                                                                              self.M.hs: self.train_val_hs,
                                                                              self.M.weights: np.ones(batch_xs.shape[0]),
                                                                              self.M.drop_rate: self.sts.dropout,
                                                                              self.M.step: i,
                                                                              self.M.is_training: False})
        self.forward_time.append(datetime.datetime.now() - ts)
        # Update hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.train_logs.append([i] + [datetime.datetime.now()] + list(acc))
        # Write tensorboard logs
        self.train_writer.add_summary(summaries, i)

    def eval_on_validation_single_step(self, i):
        """
        Evaluation Step: Samples a new batch out of the validation set and performs
        a forward pass. Also logs the performance about the evaluation set. Saves
        the model if it performed better than ever before.
        Input:
            i : the current step (int)
        """
        prct, batch_xs , batch_ys, continuity = self.SR.sample_val_batch()
        self.val_hs = np.swapaxes(np.swapaxes(self.val_hs,-2,-1)*continuity,-2,-1)
        # Computes accuracy and loss + acquires summaries
        acc, loss, summaries, self.val_hs = self.sess.run([self.M.acc_op, self.M.s_loss,
                                                           self.M.merged, self.M.current_state],
                                                          feed_dict = {self.M.x: batch_xs,
                                                                       self.M.y: batch_ys,
                                                                       self.M.hs: self.val_hs,
                                                                       self.M.weights: np.ones(batch_xs.shape[0]),
                                                                       self.M.drop_rate: self.sts.dropout,
                                                                       self.M.step: i,
                                                                       self.M.is_training: False})
        # Update Single-Step hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.test_logs.append([i] + [datetime.datetime.now()]+list(acc))
        # Update inner variables and saves best model weights
        avg = np.mean(acc)
        if  avg < self.best_1s:
            self.best_1s = avg
            NN_save_name = os.path.join(self.sts.model_ckpt,'Best_1S')
            self.saver.save(self.sess, NN_save_name)
        # Write tensorboard logs
        self.test_writer.add_summary(summaries, i)
        # Return accuracy for console display
        return acc

    def get_predictions(self, full, hs, i):
        """
        Get the predictions of the network.
        Input:
            full: a batch of inputs in the form [Batch_size x Sequence_size x Input_size]
        Output:
            pred: the predictions associated with those batches in the
                  form [Batch_size x Output_dim]
        """
        pred, hs = self.sess.run([self.M.y_, self.M.mid_state],
                                 feed_dict = {self.M.x: full,
                                              self.M.hs: hs,
                                              self.M.drop_rate:self.sts.dropout,
                                              self.M.weights: np.ones(full.shape[0]),
                                              self.M.is_training: False,
                                              self.M.step: i})
        return pred, hs

    def eval_on_validation_multi_step(self, i): # TODO replace i by train_step
        """
        Evaluation Step on Trajectories: Samples a new batch of trajectories to
        evaluate the network on. Performs a forward pass and logs the
        performance of the model on multistep predictions. If the models performed
        better than ever before then save model.
        Input:
            i : the current step (int)
        """
        predictions = []
        # Sample trajectory batch out of the evaluation set
        batch_x, batch_y = self.SR.sample_val_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        hs = self.val_traj_hs
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            pred, hs = self.get_predictions(full, hs, i)
            pred = pred[:,-1,:]
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            if self.sts.cmd_dim > 0:
                cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
                new = np.concatenate((pred, cmd), axis=1)
            else:
                new = pred
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((old,new), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        return self.eval_multistep(predictions, batch_y, i)

    def run_test_on_multi_step(self): # TODO replace i by train_step
        """
        Test on Trajectories: Samples a new batch of trajectories to
        evaluate the network on. Performs a forward pass and logs the
        performance of the model on multistep predictions.
        """
        predictions = []
        # Sample trajectory batch out of the test set
        batch_x, batch_y = self.SR.sample_test_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        hs = self.test_traj_hs
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            pred, hs = self.get_predictions(full, hs, 0)
            pred = pred[:,-1,:]
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            if self.sts.cmd_dim > 0:
                cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
                new = np.concatenate((pred, cmd), axis=1)
            else:
                new = pred
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((old,new), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        # Compute error
        self.test_ms_RMSE = np.sqrt(np.mean((predictions[:,:,:] - batch_y[:,:-1,:])**2,axis=(0,1)))
        per_traj_RMSE = np.sqrt(np.mean((predictions[:,:,:] - batch_y[:,:-1,:])**2,axis=1))
        self.test_ms_STD = np.std(per_traj_RMSE,axis=0)

class Training_Seq2Seq(Training_Uniform):
    def __init__(self, Settings):
        super(Training_Seq2Seq, self).__init__(Settings)

    def load_dataset(self):
        """
        Instantiate the dataset-reader object.
        """
        self.DS = readers.H5Reader_Seq2Seq(self.sts)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.UniformSampler(self.DS, self.sts)

    def get_predictions(self, full, i):
        """
        Get the predictions of the network.
        Input:
            full: a batch of inputs in the form [Batch_size x Sequence_size x Input_size]
        Output:
            pred: the predictions associated with those batches in the
                  form [Batch_size x Output_dim]
        """
        pred = self.sess.run(self.M.y_, feed_dict = {self.M.x: full,
                                                     self.M.drop_rate:self.sts.dropout,
                                                     self.M.weights: np.ones(full.shape[0]),
                                                     self.M.is_training: False,
                                                     self.M.step: i})
        return pred

    def run_test_single_step(self):
        """
        Test Step: runs the model on the whole of the test set and performs
        a forward pass.
        """
        try:
            prct, batch_xs , batch_ys = self.SR.sample_test_batch()
        except:
            batch_xs , batch_ys = self.SR.sample_test_batch()
        # Computes accuracy and loss + acquires summaries
        y_ = self.sess.run([self.M.y_], feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.drop_rate: self.sts.dropout,
                                                     self.M.step: 0,
                                                     self.M.is_training: False})
        batch_ys = batch_ys[:,-1,:]
        # Compute RMSE on test set
        self.test_ss_RMSE = np.sqrt(np.mean((np.squeeze(batch_ys) - y_[0])**2,axis=0))
        print(self.test_ss_RMSE)

class Training_PER(Training_Uniform):
    """
    Training container with the Priorized Experience Replay scheme. Used to train
    and evaluate the neural networks performances.
    """
    def __init__(self, Settings):
        super(Training_PER, self).__init__(Settings)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.PERSampler(self.DS, self.sts)

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        Input:
            i : the current step (int)
        """
        # Update weights
        ts = datetime.datetime.now()
        if ((i%self.sts.per_refresh_rate == 0) and (i!=0)):
            self.SR.reset_for_update()
            loss = []
            try:
                while True:
                    prct, batch_x, batch_y = self.SR.sample_for_update()
                    batch_loss = self.sess.run(self.M.s_loss,
                                           feed_dict = {self.M.x: batch_x,
                                                        self.M.y: batch_y,
                                                        self.M.weights: np.ones(self.sts.update_batchsize),
                                                        self.M.drop_rate: self.sts.dropout,
                                                        self.M.step: i,
                                                        self.M.is_training: True})
                    loss.append(batch_loss)
            except:
                loss = np.hstack(loss)
                self.SR.update_weights(loss)
        # Train
        batch_x, batch_y, weights = self.SR.sample_train_batch(self.sts.batch_size)
        _, s_loss, s_weights= self.sess.run([self.M.train_step, self.M.s_loss, self.M.weights], feed_dict = {self.M.x: batch_x,
                                                          self.M.y: batch_y,
                                                          self.M.weights: weights,
                                                          self.M.drop_rate: self.sts.dropout,
                                                          self.M.step: i,
                                                          self.M.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

class Training_Seq2Seq_PER(Training_Seq2Seq):
    """
    Training container with the Priorized Experience Replay scheme. Used to train
    and evaluate the neural networks performances.
    """
    def __init__(self, Settings):
        super(Training_Seq2Seq_PER, self).__init__(Settings)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.PERSampler(self.DS, self.sts)

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        Input:
            i : the current step (int)
        """
        # Update weights
        ts = datetime.datetime.now()
        if ((i%self.sts.per_refresh_rate == 0) and (i!=0)):
            self.SR.reset_for_update()
            loss = []
            try:
                while True:
                    prct, batch_x, batch_y = self.SR.sample_for_update()
                    batch_loss = self.sess.run(self.M.s_loss,
                                           feed_dict = {self.M.x: batch_x,
                                                        self.M.y: batch_y,
                                                        self.M.weights: np.ones(self.sts.update_batchsize),
                                                        self.M.drop_rate: self.sts.dropout,
                                                        self.M.step: i,
                                                        self.M.is_training: True})
                    loss.append(batch_loss)
            except:
                loss = np.hstack(loss)
                self.SR.update_weights(loss)
        # Train
        batch_x, batch_y, weights = self.SR.sample_train_batch(self.sts.batch_size)
        _, s_loss, s_weights= self.sess.run([self.M.train_step, self.M.s_loss, self.M.weights], feed_dict = {self.M.x: batch_x,
                                                          self.M.y: batch_y,
                                                          self.M.weights: weights,
                                                          self.M.drop_rate: self.sts.dropout,
                                                          self.M.step: i,
                                                          self.M.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

class Training_GRAD(Training_Uniform):
    """
    Training container with a Gradient Upperbound priorization scheme. Used to
    train and evaluate the neural networks performances.
    """
    def __init__(self, Settings):
        super(Training_GRAD, self).__init__(Settings)

    def load_sampler(self):
        """
        Instantiate the sampler object
        """
        self.SR = samplers.GRADSampler(self.DS, self.sts)

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        This training step leverages the Gradient-UpperBound priorization
        scheme: A superbatch is first sampled and the norm of the gradient of
        the network is used to evaluate the importance of each samples.
        Then we resample from that superbatch a batch using the value of the
        gradient of each sample as the sampling distribution. This introduces
        a bias in the learning process which canceled by weighting the loss.
        Input:
            i : the current step (int)
        """
        ts = datetime.datetime.now()
        prct, superbatch_x, superbatch_y = self.SR.sample_superbatch()
        G = self.sess.run(self.M.grad, feed_dict = {self.M.x: superbatch_x,
                                                    self.M.y: superbatch_y,
                                                    self.M.weights: np.ones(self.sts.superbatch_size),
                                                    self.M.drop_rate: self.sts.dropout,
                                                    self.M.step: i,
                                                    self.M.is_training: True})
        batch_x, batch_y, weights = self.SR.sample_train_batch(superbatch_x, superbatch_y, G[0])
        _ = self.sess.run(self.M.train_step, feed_dict = {self.M.x: batch_x,
                                                          self.M.y: batch_y,
                                                          self.M.weights: weights,
                                                          self.M.drop_rate: self.sts.dropout,
                                                          self.M.step: i,
                                                          self.M.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

class Training_CoTeaching(Training_Uniform):
    """
    Co-Teaching Training container. Used to train and evaluate the
    neural networks performances on noisy datasets.
    """
    def __init__(self, Settings):
        # Setting object
        self.sts = Settings
        # Dataset object
        self.DS = None
        self.load_dataset()
        # Samplers
        self.SR = None
        self.load_sampler()
        # Model
        self.M = None
        self.M_2 = None
        self.load_model()
        # Train
        self.init_records()
        self.train()

    def load_model(self):
        """
        Calls the network generator to load/generate the requested model.
        Please note that the generation of models is still experimental
        and may change frequently. For more information have look at the
        network_generator.
        """
        self.M = network_generator.get_graph(self.sts, name='net1')
        self.M_2 = network_generator.get_graph(self.sts, name='net2')

    def train_step(self, i):
        """
        Training step: Samples a new batch, perform forward and backward pass.
        Please note that the networks take a large amount of placeholders as
        input. They are not necessarily used they are here to maximize
        compatibility between the differemt models, and priorization schemes.
        Input:
            i : the current step (int)
        """
        # Run forward pass
        prct, batch_xs, batch_ys = self.SR.sample_train_batch()
        if prct == 0:
            print('STARTING NEW EPOCH: !')
            self.threshold = 1 - np.min([i*self.sts.tau/self.sts.k_iter, self.sts.tau])
            print('Updating Co-Teaching threshold: ', self.threshold)

        ts = datetime.datetime.now()
        L1 = self.sess.run(self.M.s_loss, feed_dict = {self.M.x: batch_xs,
                                                         self.M.y: batch_ys,
                                                         self.M.weights: np.ones(self.sts.batch_size),
                                                         self.M.drop_rate: self.sts.dropout,
                                                         self.M.step: i,
                                                         self.M.is_training: True})
        L2 = self.sess.run(self.M_2.s_loss, feed_dict = {self.M_2.x: batch_xs,
                                                         self.M_2.y: batch_ys,
                                                         self.M_2.weights: np.ones(self.sts.batch_size),
                                                         self.M_2.drop_rate: self.sts.dropout,
                                                         self.M_2.step: i,
                                                         self.M_2.is_training: True})
        # Compute threshold
        filtered_from_L1 = np.argsort(L1)[:int(self.sts.batch_size*self.threshold)]
        filtered_from_L2 = np.argsort(L2)[:int(self.sts.batch_size*self.threshold)]
        # Apply backward pass
        _ = self.sess.run(self.M.train_step, feed_dict = {self.M.x: batch_xs[filtered_from_L2],
                                                            self.M.y: batch_ys[filtered_from_L2],
                                                            self.M.weights: np.ones(int(self.sts.batch_size*self.threshold)),
                                                            self.M.drop_rate: self.sts.dropout,
                                                            self.M.step: i,
                                                            self.M.is_training: True})
        _ = self.sess.run(self.M_2.train_step, feed_dict = {self.M_2.x: batch_xs[filtered_from_L1],
                                                            self.M_2.y: batch_ys[filtered_from_L1],
                                                            self.M_2.weights: np.ones(int(self.sts.batch_size*self.threshold)),
                                                            self.M_2.drop_rate: self.sts.dropout,
                                                            self.M_2.step: i,
                                                            self.M_2.is_training: True})
        self.backward_time.append(datetime.datetime.now() - ts)

    def eval_on_train(self, i):
        """
        Evaluation Step: Samples a new batch and perform forward pass. The sampler here
        is independant from the training one. Also logs information about training
        performance in a list.
        Input:
            i : the current step (int)
        """
        prct, batch_xs, batch_ys = self.SR.sample_eval_train_batch()
        # Computes accuracy and loss + acquires summaries
        acc_1, loss_1, summaries = self.sess.run([self.M.acc_op, self.M.s_loss, self.M.merged],
                                        feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.drop_rate: self.sts.dropout,
                                                     self.M.step: i,
                                                     self.M.is_training: False})
        ts = datetime.datetime.now()
        acc_2, loss_2 = self.sess.run([self.M_2.acc_op, self.M_2.s_loss],
                                        feed_dict = {self.M_2.x: batch_xs,
                                                     self.M_2.y: batch_ys,
                                                     self.M_2.weights: np.ones(batch_xs.shape[0]),
                                                     self.M_2.drop_rate: self.sts.dropout,
                                                     self.M_2.step: i,
                                                     self.M_2.is_training: False})
        acc = (acc_1 + acc_2)/2
        #print(np.mean(loss_1-loss_2))
        #print(np.max(loss_1-loss_2))
        self.forward_time.append(datetime.datetime.now() - ts)
        # Update hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.train_logs.append([i] + [datetime.datetime.now()] + list(acc))
        # Write tensorboard logs
        self.train_writer.add_summary(summaries, i)
