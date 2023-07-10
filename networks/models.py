import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as Layers
from activations import *
from losses import *


def accuracy(est, gt):
    with tf.name_scope('accuracy_op'):
        accuracy = RMSE(est, gt)
        tf.summary.scalar('accuracy', tf.reduce_mean(accuracy))
    return accuracy

def train_fn(loss, learning_rate, step, settings):
    with tf.name_scope('train'):
        learning_rate = apply_decay(step, learning_rate, settings)
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = apply_optimizer(settings.optimizer, learning_rate, loss)
    return train_step


class DenseNormalGamma(Layers.Layer):
    def __init__(self, units, name):
        super(DenseNormalGamma, self).__init__()
        self.units = int(units)
        self.dense = Layers.Dense(4 * self.units, activation=None, name=name)

    def evidence(self, x):
        return tf.nn.softplus(x)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config['units'] = self.units
        return base_config


class GraphATTNSP:
    """
    Attention Based Network
    """
    def __init__(self, settings, model_depth, layers, params, act='RELU'):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        mask = self.make_causal_mask(settings.sequence_length)
        tf.summary.image('causal_mask', tf.expand_dims(mask,-1))
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Embedding
        self.embbed = Layers.TimeDistributed(Layers.Dense( model_depth, use_bias=False, activation=None))(self.x)
        # Positional Embedding
        self.pcoded = self.positional_encoding(self.embbed, model_depth, settings.sequence_length)
        # Attention Projection
        with tf.name_scope('qkv_projection'):
            self.QW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='q_projection'))(self.pcoded)
            self.KW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='k_projection'))(self.pcoded)
            self.VW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='v_projection'))(self.pcoded)
        # Attention mechanism
        self.xr = self.attention(self.QW, self.KW, self.VW, float(model_depth), mask, name='full_attn')
        # FeedForward
        self.xc = self.xr
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.TimeDistributed(Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i)))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(self.drop_rate, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.ys_ = Layers.TimeDistributed(Layers.Dense(settings.output_dim, activation=None, name='outputs'))(self.xc)
        self.y_ = self.ys_[:,-1,:]
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

    def make_causal_mask(self, size):
        with tf.name_scope('causal_mask'):
            return tf.linalg.band_part(tf.ones((1,size, size)), -1, 0)
            #return tf.cast(np.tril(np.ones((1,size,size)),k=0),dtype=tf.float32)


    def attention(self, query, key, value, d_k, mask, name='attention'):
        with tf.variable_scope(name):
            with tf.name_scope('attention_weights'):
                scores = tf.divide(tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])),tf.math.sqrt(d_k))
                #masked_scores = tf.multiply(mask, scores)
                masked_scores = tf.multiply(scores, mask) - (1-mask)*1e9
                self.p_attn = tf.nn.softmax(masked_scores, axis = -1)
                tf.summary.image('attn_weights', tf.expand_dims(self.p_attn,-1))
            return tf.matmul(self.p_attn, value)

    def positional_encoding(self, x, model_depth, max_len=5000, name='positional_encoding'):
        with tf.variable_scope(name):
            encoding = np.zeros([max_len, model_depth], np.float32)
            position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
            div_term = np.exp(np.arange(0,model_depth,2, dtype=np.float32) * (-np.log(10000.0)/model_depth))
            encoding[:, 0::2] = np.sin(position*div_term)
            encoding[:, 1::2] = np.cos(position*div_term)
            return tf.math.add(x, tf.cast(encoding, dtype=tf.float32), name = 'encoding')

class GraphATTNMP(GraphATTNSP):
    """
    Seq2Seq Attention based network
    """
    def __init__(self, settings, alpha, model_depth, layers, params, act='RELU'):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        mask = self.make_causal_mask(settings.sequence_length)
        # Embedding
        self.embbed = Layers.TimeDistributed(Layers.Dense( model_depth, use_bias=False, activation=None))(self.x)
        # Positional Embedding
        self.pcoded = self.positional_encoding(self.embbed, model_depth, settings.sequence_length)
        # Attention Projection
        with tf.name_scope('qkv_projection'):
            self.QW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='q_projection'))(self.pcoded)
            self.KW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='k_projection'))(self.pcoded)
            self.VW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='v_projection'))(self.pcoded)
        # Attention mechanism
        self.xr = self.attention(self.QW, self.KW, self.VW, float(model_depth), mask, name='full_attn')
        # FeedForward
        self.xc = self.xr
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.TimeDistributed(Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i)))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(self.drop_rate, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.ys_ = Layers.TimeDistributed(Layers.Dense(settings.output_dim, activation=None, name='outputs'))(self.xc)
        self.y_ = self.ys_[:,-1,:]
        # Loss
        with tf.name_scope('loss_ops'):
            a = tf.range(settings.sequence_length,dtype=tf.float32)
            self.loss_weights = tf.pow(alpha,a)
            tf.summary.histogram('sequence_weights', self.loss_weights)
            self.diff = tf.square(tf.subtract(self.ys_, self.y))
            self.seq_loss = tf.multiply(tf.reduce_mean(self.diff, axis = -1),self.loss_weights)
            self.s_loss = tf.reduce_mean(self.seq_loss, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphATTNMPMH(GraphATTNSP):
    """
    Multi-Head Seq2Seq Attention based network
    """
    def __init__(self, settings, alpha, model_depth, layers, params, act='RELU'):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        state = self.x[:,:,:settings.input_dim]
        cmd = self.x[:,:,-settings.cmd_dim:]
        mask = self.make_causal_mask(settings.sequence_length)
        # Embedding
        self.embbed_state = tf.keras.layers.TimeDistributed(Layers.Dense(model_depth, use_bias=False, activation=None, name='state_embedding'))(state)
        self.embbed_cmd = tf.keras.layers.TimeDistributed(Layers.Dense(model_depth, use_bias=False, activation=None, name='cmd_embedding'))(cmd)
        # Positional Embedding
        self.pcoded_state = self.positional_encoding(self.embbed_state, model_depth, settings.sequence_length, name='state_positional_embedding')
        self.pcoded_cmd = self.positional_encoding(self.embbed_cmd, model_depth, settings.sequence_length, name='cmd_positional_embedding')
        # Attention Projection
        with tf.name_scope('state_qkv_projection'):
            self.SQW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='state_q_projection'))(self.pcoded_state)
            self.SKW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='state_k_projection'))(self.pcoded_state)
            self.SVW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='state_v_projection'))(self.pcoded_state)
        with tf.name_scope('cmd_qkv_projection'):
            self.CQW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='cmd_q_projection'))(self.pcoded_cmd)
            self.CKW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='cmd_k_projection'))(self.pcoded_cmd)
            self.CVW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='cmd_v_projection'))(self.pcoded_cmd)
        # Attention mechanism
        self.state_attn = self.attention(self.SQW, self.SKW, self.SVW, float(model_depth), mask, name='state_attn')
        self.p_attn_state = self.p_attn
        self.cmd_attn = self.attention(self.CQW, self.CKW, self.CVW, float(model_depth), mask, name='cmd_attn')
        self.p_attn_cmd = self.p_attn
        self.concat = tf.concat([self.state_attn, self.cmd_attn], axis=-1)
        # FeedForward
        self.xc = self.concat
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.TimeDistributed(Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i)))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(self.drop_rate, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.ys_ = Layers.TimeDistributed(Layers.Dense(settings.output_dim, activation=None, name='outputs'))(self.xc)
        self.y_ = self.ys_[:,-1,:]
        # Loss
        with tf.name_scope('loss_ops'):
            a = tf.range(settings.sequence_length,dtype=tf.float32)
            self.loss_weights = tf.pow(alpha,a)
            tf.summary.histogram('sequence_weights', self.loss_weights)
            self.diff = tf.square(tf.subtract(self.ys_, self.y))
            self.seq_loss = tf.multiply(tf.reduce_mean(self.diff, axis = -1),self.loss_weights)
            self.s_loss = tf.reduce_mean(self.seq_loss, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))/tf.reduce_mean(self.loss_weights)
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphATTNMPMH_PHY(GraphATTNSP):
    """
    Multi-Head Seq2Seq Attention based network
    """
    def __init__(self, settings, alpha, model_depth, layers, params, act='RELU'):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        state = self.x[:,:,:settings.input_dim]
        cmd = self.x[:,:,-settings.cmd_dim:]
        mask = self.make_causal_mask(settings.sequence_length)
        # Embedding
        self.embbed_state = tf.keras.layers.TimeDistributed(Layers.Dense(model_depth, use_bias=False, activation=None, name='state_embedding'))(state)
        self.embbed_cmd = tf.keras.layers.TimeDistributed(Layers.Dense(model_depth, use_bias=False, activation=None, name='cmd_embedding'))(cmd)
        # Positional Embedding
        self.pcoded_state = self.positional_encoding(self.embbed_state, model_depth, settings.sequence_length, name='state_positional_embedding')
        self.pcoded_cmd = self.positional_encoding(self.embbed_cmd, model_depth, settings.sequence_length, name='cmd_positional_embedding')
        # Attention Projection
        with tf.name_scope('state_qkv_projection'):
            self.SQW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='state_q_projection'))(self.pcoded_state)
            self.SKW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='state_k_projection'))(self.pcoded_state)
            self.SVW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='state_v_projection'))(self.pcoded_state)
        with tf.name_scope('cmd_qkv_projection'):
            self.CQW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='cmd_q_projection'))(self.pcoded_cmd)
            self.CKW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='cmd_k_projection'))(self.pcoded_cmd)
            self.CVW = Layers.TimeDistributed(Layers.Dense(model_depth, activation=get_activation(act), name='cmd_v_projection'))(self.pcoded_cmd)
        # Attention mechanism
        self.state_attn = self.attention(self.SQW, self.SKW, self.SVW, float(model_depth), mask, name='state_attn')
        self.p_attn_state = self.p_attn
        self.cmd_attn = self.attention(self.CQW, self.CKW, self.CVW, float(model_depth), mask, name='cmd_attn')
        self.p_attn_cmd = self.p_attn
        self.concat = tf.concat([self.state_attn, self.cmd_attn], axis=-1)
        # FeedForward
        self.xc = self.concat
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.TimeDistributed(Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i)))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(1-settings.dropout, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.nnl = Layers.TimeDistributed(Layers.Dense(settings.output_dim, activation=None, name='outputs'))(self.xc)
        self.A = Layers.TimeDistributed(Layers.Dense(settings.output_dim,activation=None,use_bias=False))(self.x[:,:,:settings.output_dim])
        self.B = Layers.TimeDistributed(Layers.Dense(settings.output_dim,activation=None,use_bias=False))(self.x[:,:,settings.output_dim:])
        self.linear_model = tf.add(self.A,self.B)
        self.ys_ = tf.add(self.linear_model,self.nnl)
        self.y_ = self.ys_[:,-1,:]
        # Loss
        with tf.name_scope('loss_ops'):
            a = tf.range(settings.sequence_length,dtype=tf.float32)
            self.loss_weights = tf.pow(alpha,a)
            tf.summary.histogram('sequence_weights', self.loss_weights)
            self.diff_linear = tf.square(tf.subtract(self.linear_model, self.y))
            self.diff_nnl = tf.square(tf.subtract(tf.subtract(self.y, self.linear_model),self.nnl))
            self.diff_full = tf.add(self.diff_nnl, self.diff_linear)
            self.seq_loss = tf.multiply(tf.reduce_mean(self.diff_full, axis = -1),self.loss_weights)
            self.s_loss = tf.reduce_mean(self.seq_loss, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))/tf.reduce_mean(self.loss_weights)
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        #self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMLP:
    """
    MLP.
    """
    def __init__(self, settings, layers, params, act='RELU'):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        self.xc = self.xr
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(self.drop_rate, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.y_ = Layers.Dense(settings.output_dim, activation=None, name='outputs')(self.xc)

        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        print(self.y_)
        print(self.s_loss)
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphEvdMLP:
    """
    MLP.
    """
    def __init__(self, settings, layers, params, act='RELU'):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        self.xc = self.xr
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(self.drop_rate, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.pred = DenseNormalGamma(settings.output_dim, name='outputs')(self.xc)
        self.y_ = tf.split(self.pred, 4, axis=-1)[0]
        # Loss
        with tf.name_scope('loss_ops'):
            self.s_loss = tf.reduce_mean(EvidentialRegression(self.y, self.pred, coeff=1e-2),axis=-1)
            print(self.s_loss)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        print(self.pred)
        print(self.y_)
        print(self.s_loss)
        #self.grad = tf.norm(tf.gradients(self.s_loss, self.pred),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMLP_PHY:
    """
    MLP with state space and adaptive control inspiration.
    """
    def __init__(self, settings, layers, params, act='RELU'):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        self.xc = self.xr
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(1-self.keep_prob, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.A = Layers.Dense(settings.output_dim,activation=None,use_bias=False)(self.x[:,-1,:settings.output_dim])
        self.B = Layers.Dense(settings.output_dim,activation=None,use_bias=False)(self.x[:,-1,settings.output_dim:])
        self.linear_model = tf.add(self.A,self.B)
        self.nnl = Layers.Dense(settings.output_dim, activation=None, name='outputs')(self.xc)
        self.y_ = tf.add(self.linear_model,self.nnl)
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff_linear = tf.square(tf.subtract(self.linear_model, self.yr))
            self.diff_nnl = tf.square(tf.subtract(tf.subtract(self.yr, self.linear_model),self.nnl))
            self.diff_full = tf.add(self.diff_nnl, self.diff_linear)
            self.s_loss = tf.reduce_mean(self.diff_full, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        #self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMLP_EN:
    """
    MLP.
    """
    def __init__(self, settings, layers, params, act='RELU'):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        self.nl = self.xr
        with tf.name_scope('non_linearity'):
            for i, layer_type in enumerate(layers):
                if layer_type == 'dense':
                    self.nl = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.nl)
                if layer_type == 'dropout':
                    self.nl = Layers.Dropout(1-self.keep_prob, name='drop_'+str(i))(self.nl, training=self.is_training)
            self.nl = Layers.Dense(settings.output_dim, activation=None, name='non_linearity')(self.nl)
        self.cmd = self.x[:,-1,settings.output_dim:]
        self.state = self.x[:,-1,:settings.output_dim]
        with tf.name_scope('cmd'):
            for i, layer_type in enumerate(layers):
                if layer_type == 'dense':
                    self.cmd = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.cmd)
                if layer_type == 'dropout':
                    self.cmd = Layers.Dropout(1-self.keep_prob, name='drop_'+str(i))(self.cmd, training=self.is_training)
            self.cmd = Layers.Dense(settings.output_dim, activation=None, name='input_commands')(self.cmd)
        #self.en = tf.identity(self.state)
        #with tf.name_scope('energy'):
        #    for i, layer_type in enumerate(layers):
        #        if layer_type == 'dense':
        #            self.en = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.en)
        #        if layer_type == 'dropout':
        #            self.en = Layers.Dropout(1-self.keep_prob, name='drop_'+str(i))(self.en, training=self.is_training)
        #    self.en = Layers.Dense(settings.output_dim, activation=None, name='resistance')(self.en)
        #self.decay = tf.identity(self.state)
        #with tf.name_scope('decay'):
        #    for i, layer_type in enumerate(layers):
        #        if layer_type == 'dense':
        #            self.decay = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.decay)
        #        if layer_type == 'dropout':
        #            self.decay = Layers.Dropout(1-self.keep_prob, name='drop_'+str(i))(self.decay, training=self.is_training)
        #    self.decay = Layers.Dense(1, activation=None, name='energy_decay')(self.decay)
        #w_init = tf.random_normal_initializer()
        #self.wA = tf.Variable(initial_value=w_init(shape=(self.state.shape[-1], settings.output_dim),dtype='float32'),trainable=True)
        self.A = Layers.Dense(settings.output_dim,activation=None,use_bias=False)(self.state)
        #self.A = tf.matmul(self.state,self.wA)#Layers.Dense(settings.output_dim,activation=None,use_bias=False)(self.state)
        self.B = Layers.Dense(settings.output_dim,activation=None,use_bias=False)(self.cmd)
        self.linear_model = tf.add(self.A,self.B)
        #self.y_ = tf.subtract(tf.add(self.linear_model,self.nl),self.en)
        self.y_ = tf.add(self.linear_model,self.nl)
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff_linear = tf.square(tf.subtract(self.linear_model, self.yr))
            self.diff_nnl = tf.square(tf.subtract(tf.subtract(self.yr, self.linear_model),self.nl))
            #self.energy_diff = tf.matmul(self.state,tf.pow(self.wA,25))
            #print(self.energy_diff)
            self.diff_full = tf.add(self.diff_nnl, self.diff_linear)
            self.s_loss = tf.reduce_mean(self.diff_full, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        #self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMLP_CPLX:
    """
    MLP for Complex valued number.
    """
    def __init__(self, settings, layer_type, params, act=tf.nn.relu):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Embed
        self.xe = tf.complex(self.x,self.x)
        # Reshape
        self.xr = tf.reshape(self.xe, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        for i, di in enumerate(layer_type):
            if di == 'dense':
                with tf.name_scope('dense_'+str(i)):
                    with tf.name_scope('Complex_Kernel'):
                        wr = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), params[i])))
                        wi = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), params[i])))
                        weight_matrix = tf.complex(wr,wi)
                    with tf.name_scope('Complex_Bias'):
                        br = tf.Variable(tf.zeros_initializer()(params[i]))
                        bi = tf.Variable(tf.zeros_initializer()(params[i]))
                        bias = tf.complex(br,bi)
                    with tf.name_scope('MatMul'):
                        matmul = tf.matmul(self.xr,weight_matrix)
                    with tf.name_scope('BiasAdd'):
                        biasadd = matmul + bias
                    with tf.name_scope('Activation'):
                        self.xr = tf.tanh(biasadd)
        with tf.name_scope('Complex_Kernel'):
            wr = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), settings.output_dim)))
            wi = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), settings.output_dim)))
            weight_matrix = tf.complex(wr,wi)
        with tf.name_scope('Complex_Bias'):
            br = tf.Variable(tf.zeros_initializer()(settings.output_dim))
            bi = tf.Variable(tf.zeros_initializer()(settings.output_dim))
            bias = tf.complex(br,bi)
        with tf.name_scope('MatMul'):
            matmul = tf.matmul(self.xr,weight_matrix)
        with tf.name_scope('BiasAdd'):
            biasadd = matmul + bias
        with tf.name_scope('Activation'):
            self.xr = tf.math.real(biasadd)

        self.y_ = tf.layers.dense(self.xr, settings.output_dim,  activation=None, name='outputs')
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', self.s_loss)
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphCNN:
    """
    CNN.
    """
    def __init__(self, settings, layers, params, act='RELU'):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        must_reshape = True
        self.xc = self.x
        for i, layer_type in enumerate(layers):
            if layer_type == 'conv':
                self.xc = tf.layers.Conv1D(params[i][0], params[i][1], padding='same', activation=get_activation(act), name='conv1D_'+str(i))(self.xc)
            if layer_type == 'pool':
                self.xc = Layers.MaxPool1D(params[i], params[i], padding='same', name='max_pool1D_'+str(i))(self.xc)
            if layer_type == 'dense':
                if must_reshape:
                    self.xc = Layers.Flatten(name='flatten')(self.xc)
                    must_reshape = False
                self.xc = Layers.Dense(params[i], activation=get_activation(act), name='dense_'+str(i))(self.xc)
            if layer_type == 'dropout':
                self.xc = Layers.Dropout(self.drop_rate, name='drop_'+str(i))(self.xc, training=self.is_training)
        self.y_ = Layers.Dense(settings.output_dim, activation=None, name='outputs')(self.xc)

        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphRNN:
    """
    RNN.
    """
    def __init__(self, settings, hidden_state, recurrent_layers, layer_type, params, act='RELU'):
        self.v_recurrent_layers = recurrent_layers
        self.v_hidden_state = hidden_state
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.input_dim], name = 'inputs')
        self.y = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.output_dim], name = 'target')
        self.hs = tf.placeholder(tf.float32, [recurrent_layers, None, hidden_state], name = 'hidden_state')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Hidden-State definition
        self.rnn_tuple_state = tuple([self.hs[idx] for idx in range(recurrent_layers)])
        stacked_rnn = []
        for _ in range(recurrent_layers):
            stacked_rnn.append(tf.nn.rnn_cell.BasicRNNCell(hidden_state, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Reshape data
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim], name="input-reformated")
        self.yr = tf.reshape(self.y, [-1, settings.sequence_length*settings.output_dim], name="target-reformated")
        self.inputs_series = tf.split(self.xr, settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.yr, settings.sequence_length, axis=1)
        # Send data in two passes
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        # Operations
        # Roll RNN in 2 steps
        # First to get the the hidden state after only one iteration (needed for multistep)
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        # Then roll the rest
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        # FeedForward layers
        self.tensor_state = tf.transpose(tf.convert_to_tensor(self.states_series),[1,0,2])
        self.xc = self.tensor_state
        for i, layer in enumerate(layer_type):
            if layer == 'dense':
                self.xc = tf.layers.dense(self.xc, params[i], activation=get_activation(act), name='dense_'+str(i))
            if layer == 'dropout':
                self.xc = tf.layers.dropout(self.xc, self.drop_rate, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='outputs')
        # Losses
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.y))
            self.s_loss =  tf.reduce_mean(tf.reduce_mean(self.diff, axis=1),axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_[-1], self.y[-1])
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

    def get_hidden_state(self, batch_size):
        return np.zeros((self.v_recurrent_layers, batch_size, self.v_hidden_state))

class GraphGRU:
    """
    GRU.
    """
    def __init__(self, settings, hidden_state, recurrent_layers, layer_type, params, act='RELU'):
        self.v_recurrent_layers = recurrent_layers
        self.v_hidden_state = hidden_state
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.input_dim], name = 'inputs')
        self.y = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.output_dim], name = 'target')
        self.hs = tf.placeholder(tf.float32, [recurrent_layers, None, hidden_state], name = 'hidden_state')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Hidden-State definition
        self.rnn_tuple_state = tuple([self.hs[idx] for idx in range(recurrent_layers)])
        stacked_rnn = []
        for _ in range(recurrent_layers):
            stacked_rnn.append(tf.nn.rnn_cell.GRUCell(hidden_state, activation=tf.nn.tanh, name='gru_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Reshape data
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim], name="input-reformated")
        self.yr = tf.reshape(self.y, [-1, settings.sequence_length*settings.output_dim], name="target-reformated")
        self.inputs_series = tf.split(self.xr, settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.yr, settings.sequence_length, axis=1)
        # Send data in two passes
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        # Operations
        # Roll RNN in 2 steps
        # First to get the the hidden state after only one iteration (needed for multistep)
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        # Then roll the rest
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        # FeedForward layers
        self.tensor_state = tf.transpose(tf.convert_to_tensor(self.states_series),[1,0,2])
        self.xc = self.tensor_state
        for i, layer in enumerate(layer_type):
            if layer == 'dense':
                self.xc = tf.layers.dense(self.xc, params[i], activation=get_activation(act), name='dense_'+str(i))
            if layer == 'dropout':
                self.xc = tf.layers.dropout(self.xc, self.drop_rate, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='outputs')
        # Losses
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.y))
            self.s_loss =  tf.reduce_mean(tf.reduce_mean(self.diff, axis=1),axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_[-1], self.y[-1])
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

    def get_hidden_state(self, batch_size):
        return np.zeros((self.v_recurrent_layers, batch_size, self.v_hidden_state))

class GraphLSTM:
    """
    LSTM.
    """
    def __init__(self, settings, hidden_state, recurrent_layers, layer_type, params, act='RELU'):
        self.v_recurrent_layers = recurrent_layers
        self.v_hidden_state = hidden_state
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.input_dim], name = 'inputs')
        self.y = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.output_dim], name = 'target')
        self.hs = tf.placeholder(tf.float32, [recurrent_layers, 2, None, hidden_state], name = 'hidden_state')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.drop_rate = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Hidden-State definition
        state_per_layer_list = tf.unstack(self.hs, axis=0)
        self.rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                for idx in range(recurrent_layers)])
        stacked_rnn = []
        for _ in range(recurrent_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(hidden_state, activation=tf.nn.tanh, name='lstm_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Reshape data
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim], name="input-reformated")
        self.yr = tf.reshape(self.y, [-1, settings.sequence_length*settings.output_dim], name="target-reformated")
        self.inputs_series = tf.split(self.xr, settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.yr, settings.sequence_length, axis=1)
        # Send data in two passes
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        # Operations
        # Roll RNN in 2 steps
        # First to get the the hidden state after only one iteration (needed for multistep)
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        # Then roll the rest
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        # FeedForward layers
        self.tensor_state = tf.transpose(tf.convert_to_tensor(self.states_series),[1,0,2])
        self.xc = self.tensor_state
        for i, layer in enumerate(layer_type):
            if layer == 'dense':
                self.xc = tf.layers.dense(self.xc, params[i], activation=get_activation(act), name='dense_'+str(i))
            if layer == 'dropout':
                self.xc = tf.layers.dropout(self.xc, self.drop_rate, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='outputs')
        # Losses
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.y))
            self.s_loss =  tf.reduce_mean(tf.reduce_mean(self.diff, axis=1),axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_[-1], self.y[-1])
        self.train_step = train_fn(self.w_loss, settings.learning_rate, self.step, settings)
        # Tensorboard
        self.merged = tf.summary.merge_all()

    def get_hidden_state(self, batch_size):
        return np.zeros((self.v_recurrent_layers, 2, batch_size, self.v_hidden_state))

