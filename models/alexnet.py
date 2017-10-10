'''
A detailed AlexNet explaination can be found here:
http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
This model is to define an AlexNet class so that you can easily use it in other models/apps/services.

Lots of other existing project on AlexNet have been done:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/blob/master/model.py
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py
https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d/alexnet.py
...

However, I do it again since I feel, we should do it in a cleaner way. Dope.

'''

import tensorflow as tf
from models import blocks


# Lets not use any tf placeholder in this [Net] model, which more emphasize on architecture
# and keep all placeholders in [Graph] model, which more emphasize on computation
# In general, I think this is a good practice, potentially resolving a lot of troubles in future iterations
# And that's why we require you pass in [tf_input]

class AlexNet:
    def __init__(self, num_classes):
        self.output = None
        self.num_classes = num_classes
        self.net = None

    def init_networks(self, x, keep_prob):
        with tf.variable_scope('conv1'):
            conv1 = blocks.conv(x=x, kernel_size=11, depth=96, stride=4, input_channel=3, padding='VALID')
            relu1 = blocks.relu(x=conv1)
            pool1 = blocks.max_pool(x=relu1, kernel_size=3, stride=2, padding='VALID')
            norm1 = blocks.lr_norm(x=pool1, depth_radius=2, alpha=2e-05, beta=0.75)

        with tf.variable_scope('conv2'):
            conv2 = blocks.conv_with_groups(x=norm1, kernel_size=5, depth=256, stride=1, num_groups=2)
            relu2 = blocks.relu(x=conv2)
            pool2 = blocks.max_pool(x=relu2, kernel_size=3, stride=2, padding='VALID')
            norm2 = blocks.lr_norm(x=pool2, depth_radius=2, alpha=2e-05, beta=0.75)

        with tf.variable_scope('conv3'):
            conv3 = blocks.conv(x=norm2, kernel_size=3, depth=384, stride=1)
            relu3 = blocks.relu(x=conv3)

        with tf.variable_scope('conv4'):
            conv4 = blocks.conv_with_groups(x=relu3, kernel_size=3, depth=384, stride=1, num_groups=2)
            relu4 = blocks.relu(x=conv4)

        with tf.variable_scope('conv5'):
            conv5 = blocks.conv_with_groups(x=relu4, kernel_size=3, depth=256, stride=1, num_groups=2)
            relu5 = blocks.relu(x=conv5)
            pool5 = blocks.max_pool(x=relu5, kernel_size=3, stride=2, padding='VALID')

        with tf.variable_scope('flatten'):
            flattened_shape = pool5.get_shape().as_list()
            flattened = tf.reshape(tensor=pool5, shape=[-1, flattened_shape[1] * flattened_shape[2] * flattened_shape[3]])

        with tf.variable_scope('fc6'):
            fc6 = blocks.fc(x=flattened, output_size=4096)
            relu6 = blocks.relu(x=fc6)
            dropout6 = blocks.dropout(x=relu6, keep_prob=keep_prob)

        with tf.variable_scope('fc7'):
            fc7 = blocks.fc(x=dropout6, output_size=4096)
            relu7 = blocks.relu(x=fc7)
            dropout7 = blocks.dropout(x=relu7, keep_prob=keep_prob)

        with tf.variable_scope('fc8'):
            fc8 = blocks.fc(x=dropout7, output_size=self.num_classes)
            self.net = fc8
