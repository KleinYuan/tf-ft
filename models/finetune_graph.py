import tensorflow as tf


class FineTuneGraph:

    def __init__(self, model, fine_tune_layers, num_classes, summary_path, batch_size, img_height, img_width, num_channels, learning_rate):
        self.model = model
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.fine_tune_layers = fine_tune_layers
        self.learning_rate = learning_rate
        self.summary_path = summary_path

        self.net = None
        self.var_list = None
        self.x = None
        self.y = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.train_op = None
        self.grads_and_vars = None
        self.summary = None
        self.writer = None
        self.graph = None
        self.gradients = None

        self._init_graph()

    def _init_graph(self):

        self.graph = tf.Graph()
        with self.graph.as_default():

            self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.img_width, self.img_height, self.num_channels])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes])
            self.keep_prob = tf.placeholder(tf.float32)
            self.model.init_networks(x=self.x, keep_prob=self.keep_prob)
            self.net = self.model.net
            self.var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.fine_tune_layers]

            with tf.name_scope("cross_ent"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.net, labels=self.y))

            with tf.name_scope("train"):
                self.gradients = tf.gradients(self.loss, self.var_list)
                self.grads_and_vars = list(zip(self.gradients, self.var_list))
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                self.train_op = self.optimizer.apply_gradients(grads_and_vars=self.grads_and_vars)

            for gradient, var in self.grads_and_vars:
                tf.summary.histogram(var.name + '/gradient', gradient)

            for var in self.var_list:
                tf.summary.histogram(var.name, var)

            tf.summary.scalar('cross_entropy', self.loss)

            with tf.name_scope("accuracy"):
                correct_pred = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            tf.summary.scalar('accuracy', self.accuracy)

            self.summary = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(logdir=self.summary_path)

    def get_writer(self):
        return self.writer

    def get_summary(self):
        return self.summary

    def get_graph(self):
        return self.graph

    def get_placeholders(self):
        return self.x, self.y, self.keep_prob

    def get_ops(self):
        return self.train_op

    def get_loss(self):
        return self.loss