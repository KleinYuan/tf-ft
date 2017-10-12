import tensorflow as tf
import numpy as np


def load_alexnet_pre_trained_weights(weight_path, skip_layers, session):
    with session.graph.as_default():
        print 'Loading pre-trained weights in graph: ', session.graph
        weights_dict = np.load(weight_path, encoding='bytes').item()
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in skip_layers:

                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:
                        # Biases
                        if len(data.shape) == 1:
                            print 'Assigning [Biases] value to %s' % op_name
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))
                        # Weights
                        else:
                            print 'Assigning [Weights] value to %s' % op_name
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))
