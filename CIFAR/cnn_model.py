# Python 2.7
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, '
                     'weight_decay_rate')

# Defines common operations for different models. Overwrite as needed.
class Net(object):
    def __init__(self, hps, images, labels, mode):
        self.hps = hps # Hyperparameters
        self._images = images
        self.labels = labels
        self.mode = mode # 'train' or 'eval'
        
    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == 'train':
          self._build_train_op()
        self.summaries = tf.summary.merge_all()
        
    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]
        
    def _build_model(self):
        raise NotImplementedError("_build_model not implemented")
        
    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        # May want to use another optimizer?
        optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=self.global_step, name='train_step')

        # No extra train ops for now
        train_ops = [apply_op]
        self.train_op = tf.group(*train_ops)
      
    # To use weight, put 'DW' in the name of variable  
    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
          if var.op.name.find(r'DW') > 0:            
            costs.append(tf.nn.l2_loss(var))
            # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))
        
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
          n = filter_size * filter_size * out_filters
          # use weight decay; larger matrix leads to lower stddev when init
          kernel = tf.get_variable(
              'DW', [filter_size, filter_size, in_filters, out_filters],
              tf.float32, initializer=tf.random_normal_initializer(
                  stddev=np.sqrt(2.0/n)))
          return tf.nn.conv2d(x, kernel, strides, padding='SAME')
          
    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        
    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        
        # factor = 1.0 for linear layer
        #        = ~1.43 for conv layer
        # https://www.tensorflow.org/api_docs/python/tf/uniform_unit_scaling_initializer
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)
        
        
        
        
        
        
        
          
                          
