import tensorflow as tf
from services.weights_load_services import load_alexnet_pre_trained_weights
from models.alexnet import AlexNet
from models.finetune_graph import FineTuneGraph
from nolearn.lasagne import BatchIterator
from models.train import Trainer

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
TEST_STEP = 400
DATA_SPLIT_RATIOS = [0.7, 0.2, 0.1]


def run_alexnet_session(self):

    with tf.Session(graph=self.graph) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.writer.add_graph(sess.graph)
        load_alexnet_pre_trained_weights(weight_path=BVLC_ALEXNET_FP,
                                         skip_layers=FINE_TUNE_LAYERS,
                                         session=sess)
        batch_iterator = BatchIterator(batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(NUM_EPOCHS):

            step = 1

            for x_train_batch, y_train_batch in batch_iterator(self.x_train, self.y_train):
                sess.run(self.ops, feed_dict={
                    self.x_placeholder: x_train_batch,
                    self.y_placeholder: y_train_batch,
                    self.keep_prob_placeholder: KEEP_PROB
                })

                if step % VALIDATE_STEP == 0:
                    current_summary, train_loss = sess.run([self.summary, self.loss], feed_dict={
                        self.x_placeholder: x_train_batch,
                        self.y_placeholder: y_train_batch,
                        self.keep_prob_placeholder: 1.
                    })
                    val_loss = sess.run(self.loss, feed_dict={
                        self.x_placeholder: self.x_val,
                        self.y_placeholder: self.y_val,
                        self.keep_prob_placeholder: 1.
                    })

                    print '[EPOCH -- %s] Train loss: %s\nVal loss: %s' % (epoch, train_loss, val_loss)
                    self.writer.add_summary(current_summary, epoch * BATCH_SIZE + step)
                    save_path = saver.save(sess, MODEL_NAME + '_%s' % epoch)
                    print 'Taking snapshot at [Epoch = %s] [Step = %s] [Path = %s]' % (epoch, step, save_path)

                if step % TEST_STEP == 0:
                    test_loss = sess.run(self.loss, feed_dict={
                        self.x_placeholder: self.x_test,
                        self.y_placeholder: self.y_test,
                        self.keep_prob_placeholder: 1.
                    })
                    print '[EPOCH -- %s] Test loss: %s' % (epoch, test_loss)
            step += 1


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
alexnet_trainer = Trainer(graph_model=fine_tune_graph)
alexnet_trainer.feed_trainer(x='', y='', data_split_ratio=DATA_SPLIT_RATIOS)
alexnet_trainer.run_session = run_alexnet_session
alexnet_trainer.run_session()
