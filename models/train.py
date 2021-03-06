import tensorflow as tf
from sklearn.cross_validation import train_test_split


class Trainer:

    def __init__(self, graph_model):
        self.graph_model = graph_model

        self.x_train = None
        self.x_val = None
        self.x_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.graph = None
        self.x_placeholder = None
        self.y_placeholder = None
        self.keep_prob_placeholder = None
        self.writer = None
        self.summary = None
        self.ops = None
        self.loss = None

        self.session = None
        self.checkpoint = None

    def _check_point_exists(self, check_point_dir):
        self.checkpoint = tf.train.get_checkpoint_state(check_point_dir)
        if self.checkpoint is None:
            return False
        else:
            return True

    def feed_trainer(self, x, y, data_split_ratio):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=(data_split_ratio[1] + data_split_ratio[2]))
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test, self.y_test, test_size=(data_split_ratio[2] / (data_split_ratio[1] + data_split_ratio[2])))

        self.graph = self.graph_model.get_graph()
        self.x_placeholder, self.y_placeholder, self.keep_prob_placeholder = self.graph_model.get_placeholders()
        self.writer = self.graph_model.get_writer()
        self.summary = self.graph_model.get_summary()
        self.ops = self.graph_model.get_ops()
        self.loss = self.graph_model.get_loss()

    def run_session(self):
        print 'To be overriden'
