
import tensorflow as tf

def accuracy(estimation):
    accuracy = tf.reduce_mean(tf.cast(estimation, tf.float32),axis = 0)
    tf.summary.scalar('accuracy', tf.reduce_mean(accuracy))
    return accuracy

def train_fn(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    with tf.variable_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

class OneWayTransformer:
    def __init__(self, input_history, input_dim, output_forecast, output_dim, d, act=tf.nn.relu):
        self.g = rf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history,  input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

    def attention(self, query, key, d_k, value, dropout=None):
        with tf.variable_scope("attention"):
            scores = tf.divide(tf.matmul(query, tf.transpose(key, perm=[-2, -1])),tf.math.sqrt(d_k))
            p_attn = tf.nn.softmax(scores, axis = -1)
            if dropout is not None:
                p_attn = dropout(p_attn)
            return tf.matmul(p_attn, value), p_attn

    #def multi_head_attention(self, )

class GraphMLP_dX:
    def __init__(self, settings, d, act=tf.nn.relu):

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
        for i, di in enumerate(d):
            self.xr = tf.layers.dense(self.xr, di, activation=act, name='dense_'+str(i))
        self.y_ = tf.layers.dense(self.xr, settings.output_dim, activation=None, name='output')
        # Loss
        self.diff = tf.square(tf.subtract(self.y_, self.yr))
        self.s_loss = tf.reduce_mean(self.diff, axis=1)
        self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.diff)
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphCNN_kXcX_pX_dX:
    def __init__(self, settings, layers, params, act=tf.nn.relu):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        must_reshape = True
        self.xc = self.x
        for i, layer_type in enumerate(layers):
            if layer_type == 'conv':
                self.xc = tf.layers.conv1d(self.xc, params[i][0], params[i][1], padding='same', activation=act, name='conv1D_'+str(i))
            if layer_type == 'pool':
                self.xc = tf.layers.max_pooling1d(self.xc, params[i], params[i], padding='same', name='max_pool1D_'+str(i))
            if layer_type == 'dense':
                if must_reshape:
                    self.xc = tf.layers.flatten(self.xc, name='flatten')
                    must_reshape = False
                self.xc = tf.layers.dense(self.xc, params[i], activation=act, name='dense_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='output')

        # Loss
        self.diff = tf.square(tf.subtract(self.y_,self.yr))
        self.s_loss = tf.reduce_mean(self.diff, axis=1)
        self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.diff)
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMHCNN_k3cik3cip2k3cik3cid64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cip2k3cid64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ip2k3c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')
            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ip2k3c3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ip2k3c4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cik3cid64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_k3cik3cip2k3cik3cid128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cip2k3cid128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ip2k3c2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')
            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ip2k3c3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ip2k3c4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cik3cid128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_k3cik3cip2k3cik3cid128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cip2k3cid128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ip2k3c2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')
            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ip2k3c3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ip2k3c4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cik3cid128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_k3cXk3cXd64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            channel_list = [input_dim,2*input_dim,3*input_dim]
            for ch in channel_list:
                c1 = tf.layers.conv1d(self.rs, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ch))
                c2 = tf.layers.conv1d(c1, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ch))
                rs2 = tf.reshape(c2, [-1, ch*input_history],name='rs_ker_'+str(ch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc2ikXc2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc3ikXc3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc4ikXc4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cXk3cXd128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            channel_list = [input_dim,2*input_dim,3*input_dim]
            for ch in channel_list:
                c1 = tf.layers.conv1d(self.rs, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ch))
                c2 = tf.layers.conv1d(c1, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ch))
                rs2 = tf.reshape(c2, [-1, ch*input_history],name='rs_ker_'+str(ch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc2ikXc2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc3ikXc3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc4ikXc4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cXk3cXd128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            channel_list = [input_dim,2*input_dim,3*input_dim]
            for ch in channel_list:
                c1 = tf.layers.conv1d(self.rs, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ch))
                c2 = tf.layers.conv1d(c1, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ch))
                rs2 = tf.reshape(c2, [-1, ch*input_history],name='rs_ker_'+str(ch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc2ikXc2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc3ikXc3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_kXc4ikXc4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

