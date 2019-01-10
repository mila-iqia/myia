import os
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger('bigcat')

class MiniVGG:
    """
    A mini version of VGG.
    
    With hyper-parameters (lr=003, batch_size=128, dropout=0.1, epochs=10), the 
    performance is test accuracy: 0.681600, test loss: 1.024199.
    """
    def __init__(self, lr=.003, dropout=0.5, name='mini_vgg'):
        self.sess = tf.Session()
        self.lr = lr
        self.dropout = dropout
        self.name = name

        # build the network
        with tf.variable_scope(self.name):
            self.images = tf.placeholder(tf.float32, [None, 32, 32, 3])
            self.labels = tf.placeholder(tf.int64, [None])
            self.training = tf.placeholder(tf.bool)
            
            # inference
            logits = self._build_net(self.images, self.labels,
                    self.training, self.dropout)
            self.probs = tf.nn.softmax(logits)

            # optimization
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=logits)
            self.loss = tf.reduce_mean(cross_entropy)
            self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            # metrics
            correct_pred = tf.equal(tf.argmax(self.probs,1), self.labels)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        logger.info('tensorflow vgg net is created.')
        logger.info('number of vgg parameters: %d.' %(self.get_var_count()))

    def update(self, images, labels):
        """update the model with one mini-batch data
        """
        feed = {self.images: images, self.labels: labels, self.training: True} 
        loss, _ = self.sess.run([self.loss, self.opt], feed_dict=feed)
        return loss

    
    def predict(self, images):
        """update the model with one mini-batch data
        """
        feed = {self.images: images, self.training: False}
        probs_pred = self.sess.run(self.probs, feed_dict=feed)
        return probs_pred


    def evaluate(self, images, labels):
        """evaluate the model with one mini-batch data
        """
        feed = {self.images: images, self.labels: labels, self.training: False}
        loss, acc = self.sess.run([self.loss, self.acc], feed_dict=feed)
        return loss, acc
 

    @staticmethod
    def _build_net(images, labels, training, dropout):
        """build the network
        """
        x = images

        # conv-8
        conv1_1 = tf.layers.conv2d(x, 8, (3,3), 
                activation='relu', padding='same', name="conv1_1")
        conv1_2 = tf.layers.conv2d(conv1_1, 8, (3,3), 
                activation='relu', padding='same', name="conv1_2")
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2, padding='same', name='pool1')

        # conv-16
        conv2_1 = tf.layers.conv2d(pool1, 16, (3,3), 
                activation='relu', padding='same', name='conv2_1')
        conv2_2 = tf.layers.conv2d(conv2_1, 16, (3,3), 
                activation='relu', padding='same', name='conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2, padding='same', name='pool2')

        # fc-2
        flatten1 = tf.layers.flatten(pool2, name='flatten1')

        fc1 = tf.layers.dense(flatten1, 256, activation=tf.nn.relu, 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
                name="fc1")  
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=training)

        fc2 = tf.layers.dense(fc1, 256, activation=tf.nn.relu, 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                name="fc2")
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=training)

        logits = tf.layers.dense(fc2, 10, activation=None, 
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
                name="logits")
        return logits


    def save(self, model_dir='./saved_models/'):
        """save model
        """
        filepath = model_dir + 'tf_vgg.ckpt'
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.saver.save(self.sess, filepath)
        logger.info('saved model to %s' %(filepath))

    def load(self, model_dir='./saved_models/'):
        """load model
        """
        filepath = model_dir + 'tf_vgg.ckpt'
        if os.path.exists(filepath):
            self.saver.restore(self.sess, filepath)
            logger.info('loaded model from  %s' %(filepath))

    def get_var_count(self):
        count = 0
        for var in tf.trainable_variables():
            count += np.prod(var.get_shape().as_list())
        return count
