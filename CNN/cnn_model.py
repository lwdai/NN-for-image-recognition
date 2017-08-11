# Python 2.7
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

HParams = namedtuple('HParams',
                     'batch_size, num_classes, lrn_rate, min_lrn_rate, '
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
          
    def _add_bias(self, name, x, shape):
        biases = tf.get_variable(name, shape, 
            initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            
        return tf.nn.bias_add(x, biases)
          
    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
        
    def _fully_connected(self, x, out_dim, init_bias=0.0):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        
        # factor = 1.0 for linear layer
        #        = ~1.43 for conv layer
        # https://www.tensorflow.org/api_docs/python/tf/uniform_unit_scaling_initializer
        w = tf.get_variable(
            'DW', [x.get_shape()[1], out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer(init_bias))
        return tf.nn.xw_plus_b(x, w, b)
        
                
class CNN_base(Net):
    #def __init__(self, hps, images, labels, mode):
    #    super(CNN_base, self).__init__(hps, images, labels, mode)
        
    # Basically the model at models/tutorials/image/cifar10/cifar10.py
    def _build_model(self):
        with tf.variable_scope('conv1'):
            x = self._images
            x = self._conv('conv', x, 5, 3, 64, self._stride_arr(1))
            x = self._add_bias('conv_bias', x, [64])
            x = self._relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool')
            x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
          
        with tf.variable_scope('conv2'):
            x = self._conv('conv', x, 5, 64, 64, self._stride_arr(1))
            x = self._add_bias('conv_bias', x, [64])
            x = self._relu(x)
            x = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm')
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool')
                             
        with tf.variable_scope('local3') as scope:
            x = self._fully_connected(x, 384, init_bias=0.1)
            x = self._relu(x)
            
        with tf.variable_scope('local4') as scope:
            x = self._fully_connected(x, 192, init_bias=0.1)
            x = self._relu(x)           
            
        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(logits)
            
        with tf.variable_scope('costs'):
            # loss
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self.labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()

            tf.summary.scalar('cost', self.cost)
            
        

        
        
        
        
        
        
        
          
                          
