import tensorflow as tf
import numpy as np
from models.alexnet import AlexNet
from models.finetune_graph import FineTuneGraph
from nolearn.lasagne import BatchIterator

BVLC_ALEXNET_FP = 'data/bvlc_alexnet.npy'
SUMMARY_PATH = 'data/alexnet_train'
NUM_CLASSES = 2
FINE_TUNE_LAYERS = ['fc7', 'fc8']
MODEL_NAME = 'alexnet_fintune'

# Hyper Param
BATCH_SIZE = 32
IMG_HEIGHT = 244
IMG_WIDTH = 244
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
NUM_EPOCHS = 1000
KEEP_PROB = 0.5

VALIDATE_STEP = 300


def load_pre_trained_weights(weight_path, skip_layers, session):
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

                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))
                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))

# TODO: a trainer model should be added

def train(x_train, y_train):

    alexnet = AlexNet(num_classes=NUM_CLASSES)
    fine_tune_graph = FineTuneGraph(model=alexnet,
                                    fine_tune_layers=FINE_TUNE_LAYERS,
                                    num_classes=NUM_CLASSES,
                                    summary_path=SUMMARY_PATH,
                                    batch_size=BATCH_SIZE,
                                    img_height=IMG_HEIGHT,
                                    img_width=IMG_WIDTH,
                                    num_channels=NUM_CHANNELS,
                                    learning_rate=LEARNING_RATE)

    graph = fine_tune_graph.get_graph()
    x_placeholder, y_placeholder, keep_prob_placeholder = fine_tune_graph.get_placeholders()
    writer = fine_tune_graph.get_writer()
    summary = fine_tune_graph.get_summary()
    ops = fine_tune_graph.get_ops()

    with tf.Session(graph=graph) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer.add_graph(sess.graph)
        load_pre_trained_weights(weight_path=BVLC_ALEXNET_FP,
                                 skip_layers=FINE_TUNE_LAYERS,
                                 session=sess)
        batch_iterator = BatchIterator(batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(NUM_EPOCHS):

            step = 1

            for x_train_batch, y_train_batch in batch_iterator(x_train, y_train):
                sess.run(ops, feed_dict={
                    x_placeholder: x_train_batch,
                    y_placeholder: y_train_batch,
                    keep_prob_placeholder: KEEP_PROB
                })

                if step % VALIDATE_STEP == 0:
                    current_summary = sess.run(summary, feed_dict={
                        x_placeholder: x_train_batch,
                        y_placeholder: y_train_batch,
                        keep_prob_placeholder: 1.
                    })
                    writer.add_summary(current_summary, epoch * BATCH_SIZE + step)
                    save_path = saver.save(sess, MODEL_NAME + '_%s' % epoch)
                    print 'Taking snapshot at [Epoch = %s] [Step = %s] [Path = %s]' % (epoch, step, save_path)
            step += 1

                # TODO: validation to be added
