# coding=utf-8
import tensorflow as tf
from config import cfg

class VFELayer(object):

    def __init__(self, out_channels, name):
        super(VFELayer, self).__init__()
        self.units = int(out_channels / 2)
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            self.dense = tf.layers.Dense(
                self.units, tf.nn.relu, name='dense', _reuse=tf.AUTO_REUSE, _scope=scope)
            self.batch_norm = tf.layers.BatchNormalization(
                name='batch_norm', fused=True, _reuse=tf.AUTO_REUSE, _scope=scope)

    def apply(self, inputs, mask, training):
        # [K, T, 6] tensordot [6, units] = [K, T, units]
        pointwise = self.batch_norm.apply(self.dense.apply(inputs), training)
        # TODO:to be rewrite,and this method is not effective for using all of the point including poings that is not existing
        # [K, 1, units]
        aggregated = tf.reduce_max(pointwise, axis=1, keep_dims=True)

        # [K, T, units]
        repeated = tf.tile(aggregated, [1, cfg.VOXEL_POINT_COUNT, 1])

        # [K, T, 2 * units]
        concatenated = tf.concat([pointwise, repeated], axis=2)

        mask = tf.tile(mask, [1, 1, 2 * self.units])

        concatenated = tf.multiply(concatenated, tf.cast(mask, tf.float32))

        return concatenated


class Net(object):

    def __init__(self):
        super(Net, self).__init__()

    def conv2d(self,input, num_kernels=1, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', has_bias=True,
               name='conv', detas=None):

        input_shape = input.get_shape().as_list()
        assert len(input_shape) == 4
        C = input_shape[3]
        H = kernel_size[0]
        W = kernel_size[1]
        K = num_kernels
        if detas is not None:
            C = detas
        ##[filter_height, filter_width, in_channels, out_channels]
        # tf.truncated_normal_initializer(stddev=0.1)
        w = tf.get_variable(name=name + '_weight', shape=[H, W, C, K],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv2d(input, w, strides=stride, padding=padding, name=name)
        if has_bias:
            b = tf.get_variable(name=name + '_bias', shape=[K], initializer=tf.constant_initializer(0.0))
            conv = conv + b

        return conv

    def relu(self,input, name='relu'):
        act = tf.nn.relu(input, name=name)
        return act

    def bn(self,input, IS_TRAIN_PHASE, decay=0.9, eps=1e-5, name='bn'):
        with tf.variable_scope(name) as scope:
            bn = tf.cond(IS_TRAIN_PHASE,
                         lambda: tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                                                              is_training=1, reuse=None, fused=True,
                                                              updates_collections=None, scope=scope),
                         lambda: tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                                                              is_training=0, reuse=True, fused=True,
                                                              updates_collections=None, scope=scope))
            # bn = tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
            #                              is_training=IS_TRAIN_PHASE, fused=True,
            #                              updates_collections=None, scope=scope)
        return bn

    def concat(self,input, axis=3, name='cat'):
        cat = tf.concat(axis=axis, values=input, name=name)
        return cat

    def concat_relu(self,input, axis=3, name='cat'):
        concat = tf.concat(input, axis=axis, name=name)
        concat = self.relu(concat)
        return concat

    def conv2d_transpose_strided(self,input, W, b, output_shape=None, stride1=2, stride2=2):
        # print x.get_shape()
        # print W.get_shape()
        if output_shape is None:
            output_shape = input.get_shape().as_list()
            output_shape[1] *= stride1
            output_shape[2] *= stride2
            output_shape[3] = W.get_shape().as_list()[2]
        # print output_shape
        conv = tf.nn.conv2d_transpose(input, W, output_shape, strides=[1, stride1, stride2, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)

    def weight_variable(self,shape, stddev=0.02, name=None):
        # print(shape)
        # initial = tf.truncated_normal(shape, stddev=stddev)
        initial = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name,shape,dtype=tf.float32,initializer=initial)

    def bias_variable(self,shape, name=None):
        initial = tf.constant(0.0, shape=shape)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)

    def conv2d_bn_relu(self,input, IS_TRAIN_PHASE, num_kernels=1, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME',
                       name='conv'):
        with tf.variable_scope(name) as scope:
            block = self.conv2d(input, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding,
                           has_bias=True)
            block = self.bn(block, IS_TRAIN_PHASE)
            block = self.relu(block)
        return block

    def conv2d_relu(self,input, num_kernels=1, kernel_size=(1, 1), stride=[1, 1, 1, 1], padding='SAME', name='conv'):
        with tf.variable_scope(name) as scope:
            block = self.conv2d(input, num_kernels=num_kernels, kernel_size=kernel_size, stride=stride, padding=padding,
                           has_bias=True)
            block = self.relu(block)
        return block

    def load_weigths(self, data_path, session, saver):
        import numpy as np
        try:
            if data_path.endswith('.ckpt'):
                saver.restore(session, data_path)
            else:
                data_dict = np.load(data_path).item()
                for key in data_dict:
                    with tf.variable_scope(key, reuse=True):
                        for subkey in data_dict[key]:
                            try:
                                var = tf.get_variable(subkey)
                                session.run(var.assign(data_dict[key][subkey]))
                                print "assign pretrain model " + subkey + " to " + key
                            except ValueError:
                                print "ignore " + key
        except RuntimeError:
            from tensorflow.python import pywrap_tensorflow
            reader = pywrap_tensorflow.NewCheckpointReader(data_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            with tf.variable_scope('', reuse=tf.AUTO_REUSE) as scope:
                for key in var_to_shape_map:
                    try:
                        var = tf.get_variable(key, trainable=False)
                        session.run(var.assign(reader.get_tensor(key)))
                        print "    Assign pretrain model: " + key
                    except ValueError:
                        print "    Ignore variable:" + key


