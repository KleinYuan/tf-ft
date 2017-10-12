import tensorflow as tf


# We usually use the same kernel_height and kernel_width.
# Therefore, if you wanna use a more customized kernel shape, you may wanna create a new function.
def conv(x, kernel_size, depth, name, stride=1, input_channel=None, padding='SAME'):

    if input_channel is None:
        input_channel = int(x.get_shape()[3])

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name='weights',
                                  shape=[kernel_size, kernel_size, input_channel, depth]
                                  )
        biases = tf.get_variable(name='biases',
                                 shape=[depth]
                                 )
        conv_layer = tf.nn.conv2d(input=x,
                                  filter=weights,
                                  strides=[1, stride, stride, 1],
                                  padding=padding)
    conv_layer = tf.reshape(tf.nn.bias_add(conv_layer, biases), tf.shape(conv_layer))
    relu = tf.nn.relu(conv_layer, name=scope.name)
    return relu


def fc(x, name, output_size, relu=True):
    with tf.variable_scope(name) as scope:
        input_size = x.get_shape()[1]
        weights = tf.get_variable(name='weights', shape=[input_size, output_size], trainable=True)
        biases = tf.get_variable(name='biases', shape=[output_size], trainable=True)
        fc_layer = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    if relu:
        fc_layer = tf.nn.relu(fc_layer)
    return fc_layer


def dropout(x, keep_prob):
    return tf.nn.dropout(x=x, keep_prob=keep_prob)


def max_pool(x, kernel_size, stride, name, padding='SAME'):
    return tf.nn.max_pool(value=x,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding,
                          name=name)


def lr_norm(x, depth_radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input=x,
                                              depth_radius=depth_radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)


def conv_with_groups(x, kernel_size, depth, name, stride=1, padding='SAME', num_groups=2):
    assert num_groups > 1, '[Error] Num of groups is too small.'
    input_channel = int(x.get_shape()[3])
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride, stride, 1],
                                         padding=padding)
    with tf.variable_scope(name) as scope:

        input_groups = tf.split(value=x, num_or_size_splits=num_groups, axis=3)
        weights = tf.get_variable(name='weights',
                                  shape=[kernel_size, kernel_size, input_channel/num_groups, depth]
                                  )
        biases = tf.get_variable(name='biases',
                                 shape=[depth]
                                 )
        weight_groups = tf.split(value=weights, num_or_size_splits=num_groups, axis=3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
        conv_layer = tf.concat(axis=3, values=output_groups)
    bias = tf.reshape(tf.nn.bias_add(conv_layer, biases), tf.shape(conv_layer))
    relu = tf.nn.relu(bias, name=scope.name)
    return relu
