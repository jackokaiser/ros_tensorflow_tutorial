#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.dense_layer1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,))
        self.dense_layer2 = tf.keras.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        return x


class ModelWrapper():
    def __init__(self, input_dim, output_dim):
        self.session = tf.compat.v1.keras.backend.get_session()

        self.model = Model(input_dim, output_dim)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.)
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=sgd, metrics=['accuracy'])

    def predict(self, x):
        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)

            out = self.model.predict(x)
            winner = np.argmax(out[0])
            confidence = out[0, winner]
        return winner, confidence

    def train(self, x_train, y_train, n_epochs=100, callbacks=[]):
        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)

            self.model.fit(x_train, y_train,
                           batch_size=32,
                           epochs=n_epochs,
                           callbacks=callbacks)

class StopTrainOnCancel(tf.keras.callbacks.Callback):
    def __init__(self, check_preempt):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.check_preempt = check_preempt
    def on_batch_end(self, batch, logs={}):
        self.model.stop_training = self.check_preempt()

class EpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, cb):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.cb = cb
    def on_epoch_end(self, epoch, logs):
        self.cb(epoch, logs)
