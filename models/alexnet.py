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
# And that's why we require you pass in [x]

class AlexNet:
    def __init__(self, num_classes):
        self.output = None
        self.num_classes = num_classes
        self.net = None

    def init_networks(self, x, keep_prob):

        conv1 = blocks.conv(x=x, kernel_size=11, depth=96, stride=4, input_channel=3, padding='VALID', name='conv1')
        pool1 = blocks.max_pool(x=conv1, kernel_size=3, stride=2, padding='VALID', name='pool1')
        norm1 = blocks.lr_norm(x=pool1, depth_radius=2, alpha=2e-05, beta=0.75, name='norm1')

        conv2 = blocks.conv_with_groups(x=norm1, kernel_size=5, depth=256, stride=1, num_groups=2, name='conv2')
        pool2 = blocks.max_pool(x=conv2, kernel_size=3, stride=2, padding='VALID', name='pool2')
        norm2 = blocks.lr_norm(x=pool2, depth_radius=2, alpha=2e-05, beta=0.75, name='norm2')

        conv3 = blocks.conv(x=norm2, kernel_size=3, depth=384, stride=1, name='conv3')
        conv4 = blocks.conv_with_groups(x=conv3, kernel_size=3, depth=384, stride=1, num_groups=2, name='conv4')

        conv5 = blocks.conv_with_groups(x=conv4, kernel_size=3, depth=256, stride=1, num_groups=2, name='conv5')
        pool5 = blocks.max_pool(x=conv5, kernel_size=3, stride=2, padding='VALID', name='pool5')

        flattened_shape = pool5.get_shape().as_list()
        flattened = tf.reshape(tensor=pool5, shape=[-1, flattened_shape[1] * flattened_shape[2] * flattened_shape[3]])

        fc6 = blocks.fc(x=flattened, output_size=4096, name='fc6')
        dropout6 = blocks.dropout(x=fc6, keep_prob=keep_prob)

        fc7 = blocks.fc(x=dropout6, output_size=4096, name='fc7')
        dropout7 = blocks.dropout(x=fc7, keep_prob=keep_prob)

        fc8 = blocks.fc(x=dropout7, output_size=self.num_classes, relu=False, name='fc8')
        self.net = fc8
