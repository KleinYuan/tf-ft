import tensorflow as tf


# We usually use the same kernel_height and kernel_width.
# Therefore, if you wanna use a more customized kernel shape, you may wanna create a new function.
def conv(x, kernel_size, depth, stride=1, input_channel=None, padding='SAME'):

    if input_channel is None:
        input_channel = int(x.get_shape()[3])

    weights = tf.get_variable(name='weights',
                              shape=[kernel_size, kernel_size, input_channel, depth]
                              )
    biases = tf.get_variable(name='biases',
                             shape=[depth]
                             )
    conv_layer = tf.nn.conv2d(input=x,
                              filter=weights,
                              strides=[1, stride, stride, 1],
                              padding=padding) + biases
    return conv_layer


def fc(x, output_size):
    input_size = x.get_shape()[1]
    weights = tf.get_variable(name='weights', shape=[input_size, output_size], trainable=True)
    biases = tf.get_variable(name='biases', shape=[output_size], trainable=True)
    return tf.matmul(a=x, b=weights) + biases


def relu(x):
    return tf.nn.relu(x)


def dropout(x, keep_prob):
    return tf.nn.dropout(x=x, keep_prob=keep_prob)


def max_pool(x, kernel_size, stride, padding='SAME'):
    return tf.nn.max_pool(value=x,
                          ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


def lr_norm(x, depth_radius, alpha, beta, bias=1.0):
    return tf.nn.local_response_normalization(input=x,
                                              depth_radius=depth_radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)


def conv_with_groups(x, kernel_size, depth, stride=1, padding='SAME', num_groups=2):
    assert num_groups > 1, '[Error] Num of groups is too small.'
    input_channel = int(x.get_shape()[3])
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride, stride, 1],
                                         padding=padding)
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
    conv_layer = tf.reshape(tf.nn.bias_add(conv_layer, biases), tf.shape(conv_layer))
    return conv_layer
