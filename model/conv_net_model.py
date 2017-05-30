import numpy as np
import tensorflow as tf

class Config():
    B, W, H, C = 32, 32,32, 9
    train_step = 25000
    lr = 1e-3
    weight_decay = 0.005
    # DISCREPANCY - paper uses 0.50 for keep probability
    drop_out = 0.1

def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")

def conv_relu_batch(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return r

def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, H])
        return tf.matmul(input_data, W, name="matmul") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class NeuralModel():
    def __init__(self, config, name):

        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])
        
        self.conv1_1 = conv_relu_batch(self.x, 128, 3,1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = conv_relu_batch(conv1_1_d, 128, 3,2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = conv_relu_batch(conv1_2_d, 256, 3,1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = conv_relu_batch(conv2_1_d, 256, 3,2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = conv_relu_batch(conv2_2_d, 512, 3,1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2= conv_relu_batch(conv3_1_d, 512, 3,1, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)
        conv3_3 = conv_relu_batch(conv3_2_d, 512, 3,2, name="conv3_3")
        conv3_3_d = tf.nn.dropout(conv3_3, self.keep_prob)

        dim = np.prod(conv3_3_d.get_shape().as_list()[1:])
        flattened = tf.reshape(conv3_3_d, [-1, dim])

        self.fc6 = dense(flattened, 2048, name="fc6")
        self.logits = tf.squeeze(dense(self.fc6, 1, name="dense"))

        # l2
        self.loss_err = tf.nn.l2_loss(self.logits - self.y)

        with tf.variable_scope('dense') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')
        with tf.variable_scope('conv1_1/conv2d') as scope:
            scope.reuse_variables()
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')

        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = self.loss_err+self.loss_reg

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
