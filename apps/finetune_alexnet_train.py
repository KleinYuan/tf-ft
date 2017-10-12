import tensorflow as tf
import cv2
import numpy as np
import os
from sklearn.utils import shuffle
from nolearn.lasagne import BatchIterator
from pandas.io.parsers import read_csv
from services.weights_load_services import load_alexnet_pre_trained_weights
from models.alexnet import AlexNet
from models.finetune_graph import FineTuneGraph
from models.train import Trainer
from models.data import DataSets
from config.config import alexnet as alexnet_config


class GenericDataSets(DataSets):
    @staticmethod
    def _img_pre_process(img):
        new_img = cv2.resize(img,
                             (alexnet_config['hyperparams']['img_width'],
                              alexnet_config['hyperparams']['img_height']),
                             interpolation=cv2.INTER_CUBIC)
        new_img = cv2.normalize(new_img, new_img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return new_img

    @staticmethod
    def _normalize_data(data, norm_factor):
        data = (data - norm_factor) / norm_factor
        return data

    def _load(self, mode, fn, cloud=False):
        x_data = []
        print 'Loading from fp : %s' % fn
        df = read_csv(os.path.expanduser(fn))
        img_fps = df[alexnet_config['img_csv_col_name']].values

        print 'Loading imgs, resize and normalize it! '
        for index, img_fp in enumerate(img_fps):
            img = cv2.imread(img_fp)
            img = self._img_pre_process(img=img)

            print 'Loading ', img_fp
            x_data.append(img)

        x_data = np.array(x_data)
        x_data = x_data.astype(np.float32)

        if mode == 'train':
            print 'Loading features and normalize them!'

            y_data = df[df.columns[1:alexnet_config['num_classes'] + 1]].values
            y_data = y_data.astype(np.float32)
            #y_data = self._normalize_data(y_data, norm_factor=50)
            x_data, y_data = shuffle(x_data, y_data, random_state=42)
        else:
            y_data = None

        print 'Assign to instance object.'
        x_data = x_data.reshape(-1, alexnet_config['hyperparams']['img_width'], alexnet_config['hyperparams']['img_height'], alexnet_config['hyperparams']['num_channels'])
        self.x_data = x_data
        self.y_data = y_data


def run_alexnet_session(self):

    with tf.Session(graph=self.graph) as sess:

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        self.writer.add_graph(sess.graph)
        load_alexnet_pre_trained_weights(weight_path=alexnet_config['pre_trained_weights_fp'],
                                         skip_layers=alexnet_config['fine_tune_layers'],
                                         session=sess)
        batch_iterator = BatchIterator(batch_size=alexnet_config['hyperparams']['batch_size'], shuffle=True)
        current_x_train_batch = None
        current_y_train_batch = None
        step = 1

        for epoch in range(alexnet_config['hyperparams']['num_epochs']):
            print '[EPOCH -- %s/%s] In Progress ...' % (epoch, alexnet_config['hyperparams']['num_epochs'])
            for x_train_batch, y_train_batch in batch_iterator(self.x_train, self.y_train):
                sess.run(self.ops, feed_dict={
                    self.x_placeholder: x_train_batch,
                    self.y_placeholder: y_train_batch,
                    self.keep_prob_placeholder: alexnet_config['hyperparams']['keep_prob']
                })
                current_x_train_batch = x_train_batch
                current_y_train_batch = y_train_batch
                step += 1

            if step % alexnet_config['hyperparams']['validation_period'] == 0:
                current_summary, train_loss = sess.run([self.summary, self.loss], feed_dict={
                    self.x_placeholder: current_x_train_batch,
                    self.y_placeholder: current_y_train_batch,
                    self.keep_prob_placeholder: 1.
                })
                val_loss = sess.run(self.loss, feed_dict={
                    self.x_placeholder: self.x_val,
                    self.y_placeholder: self.y_val,
                    self.keep_prob_placeholder: 1.
                })

                print '[EPOCH -- %s] Train loss: %s\nVal loss: %s' % (epoch, train_loss, val_loss)
                self.writer.add_summary(current_summary, epoch * alexnet_config['hyperparams']['batch_size'] + step)
                save_path = saver.save(sess, alexnet_config['model_save_path'] + '_%s' % epoch)
                print 'Taking snapshot at [Epoch = %s] [Step = %s] [Path = %s]' % (epoch, step, save_path)

            if step % alexnet_config['hyperparams']['test_period'] == 0:
                test_loss = sess.run(self.loss, feed_dict={
                    self.x_placeholder: self.x_test,
                    self.y_placeholder: self.y_test,
                    self.keep_prob_placeholder: 1.
                })
                print '[EPOCH -- %s] Test loss: %s' % (epoch, test_loss)

data_sets = GenericDataSets()
data_sets.load(mode='train', fn=alexnet_config['csv_fp'])
alexnet = AlexNet(num_classes=alexnet_config['num_classes'])
fine_tune_graph = FineTuneGraph(model=alexnet,
                                fine_tune_layers=alexnet_config['fine_tune_layers'],
                                num_classes=alexnet_config['num_classes'],
                                summary_path=alexnet_config['tensorboard_dir'],
                                batch_size=alexnet_config['hyperparams']['batch_size'],
                                img_height=alexnet_config['hyperparams']['img_height'],
                                img_width=alexnet_config['hyperparams']['img_width'],
                                num_channels=alexnet_config['hyperparams']['num_channels'],
                                learning_rate=alexnet_config['hyperparams']['learning_rate'])
alexnet_trainer = Trainer(graph_model=fine_tune_graph)
alexnet_trainer.feed_trainer(x=data_sets.x_data, y=data_sets.y_data, data_split_ratio=alexnet_config['hyperparams']['data_split_ratios'])
Trainer.run_session = run_alexnet_session
alexnet_trainer.run_session()
