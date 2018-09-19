import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as tfk
from tensorflow.keras.initializers import RandomNormal, TruncatedNormal
from tensorflow.keras.layers import (Input, Dense, Activation, Layer, Lambda,
                                     Concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam


class TrainableVar(Layer):
    '''Creates variable that's trainable with keras model. Needs to be attached
    to some node that conditions optimizer op.'''
    def __init__(self, name, shape, **kwargs):
        super(TrainableVar, self).__init__()
        self.kernel = self.add_variable(name=name, shape=shape, **kwargs)

    def call(self, input):
        return self.kernel

class LeftBranch(Layer):
    def call(self, input):
        return input[0] * (1 - input[1])

class RightBranch(Layer):
    def call(self, input):
        return input[0] * input[1]

class Scale(Layer):
    def call(self, input):
        return input[0] * input[1]

class Node(object):
    def __init__(self, id, depth, pathprob, tree):
        self.id = id
        self.depth = depth
        self.pathprob = pathprob
        self.isLeaf = self.depth == tree.max_depth
        self.leftChild = None
        self.rightChild = None

        if self.isLeaf:
            self.phi = TrainableVar(
                name='phi_'+self.id, shape=(1, tree.n_classes),
                dtype='float32', initializer=TruncatedNormal())(pathprob)
        else:
            self.dense = Dense(
                units=1, name='dense_'+self.id, dtype='float32',
                kernel_initializer=RandomNormal(),
                bias_initializer=TruncatedNormal())(tree.input_layer)

    def build(self, tree):
        '''Defines the output probability of the node and builds child nodes.'''
        self.prob = self.forward(tree)
        if not self.isLeaf:
            leftprob = LeftBranch()([self.pathprob, self.prob])
            rightprob = RightBranch()([self.pathprob, self.prob])
            self.leftChild = Node(id=self.id+'0', depth=self.depth+1,
                                  pathprob=leftprob, tree=tree)
            self.rightChild = Node(id=self.id+'1', depth=self.depth+1,
                                   pathprob=rightprob, tree=tree)

    def forward(self, tree):
        '''Defines the output probability.'''
        if not self.isLeaf:
            self.dense_scaled = Scale()([tree.inv_temp, self.dense])
            return Activation('sigmoid', name='prob_' + self.id)(
                self.dense_scaled)
        else:
            return Activation('softmax', name='pdist_' + self.id)(self.phi)

    def get_penalty(self, tree):
        '''From paper: "... we can maintain an exponentially decaying running
        average of the actual probabilities with a time window that is
        exponentially proportional to the depth of the node."
        So here we track EMAs of batches of P^i and p_i and calculate:
            alpha = sum(ema(P^i) * ema(p_i)) / sum(ema(P^i))
            penalty = -0.5 * (log(alpha) + log(1-alpha))
        '''
        # Keep track of running average of probabilities (batch-wise)
        # with exponential growth of time window w.r.t. the  depth of the node
        self.ema = tf.train.ExponentialMovingAverage(
            decay=0.9999, num_updates=tree.ema_win_size*2**self.depth)
        self.ema_apply_op = self.ema.apply([self.pathprob, self.prob])
        self.ema_P = self.ema.average(self.pathprob)
        self.ema_p = self.ema.average(self.prob)
        # Calculate alpha by summing probs and pathprobs over batch
        self.alpha = (tf.reduce_sum(self.ema_P * self.ema_p) + tree.eps) / (
            tf.reduce_sum(self.ema_P) + tree.eps)
        # Calculate penalty for this node using running average of alpha
        self.penalty = (- 0.5 * tf.log(self.alpha + tree.eps)
                        - 0.5 * tf.log(1. - self.alpha + tree.eps))
        # Replace possible NaN values with zeros
        self.penalty = tf.where(
            tf.is_nan(self.penalty), tf.zeros_like(self.penalty), self.penalty)
        return self.penalty

    def get_loss(self, y, tree):
        if self.isLeaf:
            # Cross-entropies (batch) of soft labels with output of this leaf
            leaf_ce = - tf.reduce_sum(y * tf.log(tree.eps + self.prob), axis=1)
            # Mean of cross-entropies weighted by path probability of this leaf
            self.leaf_loss = tf.reduce_mean(self.pathprob * leaf_ce)
            # Return leaf contribution to the loss
            return self.leaf_loss
        else:
            # Return decayed penalty term of this (inner) node
            return tree.penalty_strength * self.get_penalty(tree) * (
                tree.penalty_decay**self.depth) # decay


class Constant(Layer):
    def __init__(self, value=1, **kwargs):
        self.value = value
        super(Constant, self).__init__(**kwargs)
    def call(self, input):
        return tfk.constant(self.value, shape=(1,), dtype='float32')


class OutputLayer(Layer):
    def call(self, input):
        opinions, weights = input
        opinions = Concatenate(axis=0)(opinions) # shape=(n_bigots,n_classes)
        weights = Concatenate(axis=1)(weights) # shape=(batch_size,n_bigots)
        elems = tfk.argmax(weights, axis=1) # shape=(batch_size,)
        def from_keras_tensor(opinions, elems=None):
            return tfk.map_fn(lambda x: opinions[x], elems, dtype=tf.float32)
        outputs = Lambda(
            from_keras_tensor, arguments={'elems': elems})(opinions)
        return outputs # shape=(batch_size,n_classes)


class SoftBinaryDecisionTree(object):
    def __init__(self, max_depth, n_features, n_classes,
                 penalty_strength=10.0, penalty_decay=0.5, inv_temp=0.01,
                 ema_win_size=100, learning_rate=3e-4, metrics=['acc']):
        '''Initialize model instance by saving parameter values
        as model properties and creating others as placeholders.
        '''

        # save hyperparameters
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_classes = n_classes

        self.penalty_strength = penalty_strength
        self.penalty_decay = penalty_decay
        self.inv_temp = inv_temp
        self.ema_win_size = ema_win_size

        self.nodes = list()
        self.bigot_opinions = list()
        self.bigot_weights = list()
        self.ema_apply_ops = list()

        self.loss = 0.0
        self.loss_leaves = 0.0
        self.loss_penalty = 0.0

        self.optimizer = Adam(lr=learning_rate)
        self.metrics = metrics

        self.eps = tfk.constant(1e-8, shape=(1,), dtype='float32')
        self.initialized = False

    def build_model(self):
        self.input_layer = Input(shape=(self.n_features,), dtype='float32')

        if self.inv_temp:
            self.inv_temp = Constant(value=self.inv_temp)(self.input_layer)
        else:
            self.inv_temp = TrainableVar(
                name='beta', shape=(1,), dtype='float32',
                initializer=RandomNormal())(self.input_layer)

        self.root = Node(
            id='0', depth=0, pathprob=Constant()(self.input_layer), tree=self)
        self.nodes.append(self.root)

        for node in self.nodes:
            node.build(tree=self)
            if node.isLeaf:
                self.bigot_opinions.append(node.prob)
                self.bigot_weights.append(node.pathprob)
            else:
                self.nodes.append(node.leftChild)
                self.nodes.append(node.rightChild)

        def tree_loss(y_true, y_pred):
            for node in self.nodes:
                if node.isLeaf:
                    self.loss_leaves += node.get_loss(y=y_true, tree=self)
                else:
                    self.loss_penalty += node.get_loss(y=None, tree=self)
                    self.ema_apply_ops.append(node.ema_apply_op)

            with tf.control_dependencies(self.ema_apply_ops):
                self.loss = tf.log(
                    self.eps + self.loss_leaves) + self.loss_penalty
            return self.loss

        self.output_layer = OutputLayer()(
            [self.bigot_opinions, self.bigot_weights])

        print('Built tree has {} leaves out of {} nodes'.format(
            sum([node.isLeaf for node in self.nodes]), len(self.nodes)))

        self.model = Model(inputs=self.input_layer, outputs=self.output_layer)

        self.model.compile(optimizer=self.optimizer, loss=tree_loss,
                           metrics=self.metrics)

        self.saver = tf.train.Saver()


    def initialize_variables(self, sess, x, batch_size):
        '''Since tf.ExponentialMovingAverage generates variables that
        depend on other variables being initialized first, we need to
        perform customized, 2-step initialization.
        Importantly, initialization of EMA variables also requires
        a single input batch of size that will be used for evaluation
        of loss, in order to create compatible shapes. Therefore,
        model will be constrained to initial batch size.
        '''

        if not self.model:
            print('Missing model instance.')
            return

        ema_vars = [v for v in tf.global_variables() if
                    'ExponentialMovingAverage' in v.name and
                    'Const' not in v.name]
        independent_vars = [v for v in tf.global_variables() if
                            v not in ema_vars]
        feed_dict = {self.input_layer: x[:batch_size]}

        init_indep_vars_op = tf.variables_initializer(independent_vars)
        init_ema_vars_op = tf.variables_initializer(ema_vars)

        sess.run(init_indep_vars_op)
        sess.run(init_ema_vars_op, feed_dict=feed_dict)
        self.initialized = True

    def save_variables(self, sess, path):
        '''Keras saving methods such as model.save() or model.save_weights()
        are not suitable, since Keras won't serialize tf.Tensor objects which
        get included into saving process as arguments of Lambda layers.
        '''
        self.saver.save(sess, path)

    def load_variables(self, sess, path):
        self.saver.restore(sess, path)
        self.initialized = True

    def maybe_train(self, sess, data_train, data_valid,
                    batch_size, epochs, callbacks=None, distill=False):

        DIR_ASSETS = 'assets/'
        DIR_DISTILL = 'distilled/'
        DIR_NON_DISTILL = 'non-distilled/'
        DIR_MODEL = DIR_ASSETS + (DIR_DISTILL if distill else DIR_NON_DISTILL)
        PATH_MODEL = DIR_MODEL + 'tree-model'

        try:
            print('Loading trained model from {}.'.format(PATH_MODEL))
            self.load_variables(sess, PATH_MODEL)
            return
        except ValueError as e:
            print('{} is not a valid checkpoint. Training from scratch.'.format(
                PATH_MODEL))
            x_train, y_train = data_train
            self.initialize_variables(sess, x_train, batch_size)
            self.model.fit(
                x_train, y_train, validation_data=data_valid,
                batch_size=batch_size, epochs=epochs, callbacks=callbacks)
            print('Saving trained model to {}.'.format(PATH_MODEL))
            if not os.path.isdir(DIR_MODEL):
                os.mkdir(DIR_MODEL)
            self.save_variables(sess, PATH_MODEL)

    def evaluate(self, x, y, batch_size):
        if self.model and self.initialized:
            score = self.model.evaluate(x, y, batch_size)
            print('accuracy: {:.2f}% | loss: {}'.format(100*score[1], score[0]))
        else:
            print('Missing initialized model instance.')

    def predict(self, x):
        if self.model and self.initialized:
            return self.model.predict(x, verbose=1)
        else:
            print('Missing initialized model instance.')
