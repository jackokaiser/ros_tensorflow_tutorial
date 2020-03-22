#!/usr/bin/env python3

import rospy
import actionlib
from ros_tensorflow_msgs.msg import TrainAction, TrainFeedback
from ros_tensorflow_msgs.srv import Predict, PredictResponse

from ros_tensorflow.model import ModelWrapper, StopTrainOnCancel, EpochCallback

import numpy as np

class RosInterface():
    def __init__(self):
        self.input_dim = 10
        self.output_dim = 2

        # This dictionary is only used for synthetic data generation
        self.fake_hidden_params = {}

        self.wrapped_model = ModelWrapper(input_dim=self.input_dim, output_dim=self.output_dim)
        self.train_as = actionlib.SimpleActionServer('train', TrainAction, self.train_cb, False)
        self.train_as.start()

        self.predict_srv = rospy.Service('predict', Predict, self.predict_cb)

    def make_samples(self, i_class, n_samples):
        # This function generates synthetic data
        if not i_class in self.fake_hidden_params:
            # our hidden parameters are the mean and scale of a gaussian distribution
            self.fake_hidden_params[i_class] = np.random.rand(2)
        y = np.ones(n_samples) * i_class
        y[i_class] = 1.
        return np.random.normal(*self.fake_hidden_params[i_class], size=(n_samples, self.input_dim)), y

    def make_synthetic_dataset(self, n_samples_per_class):
        x1, y1 = self.make_samples(i_class=0, n_samples=n_samples_per_class)
        x2, y2 = self.make_samples(i_class=1, n_samples=n_samples_per_class)
        x_train = np.concatenate([x1, x2])
        y_train = np.concatenate([y1, y2])
        return x_train, y_train

    def train_cb(self, goal):
        if goal.epochs <= 0:
            rospy.logerr("Number of epochs needs to be greater than 0! Given: {}".format(goal.epochs))

        stop_on_cancel = StopTrainOnCancel(check_preempt=lambda : self.train_as.is_preempt_requested())
        pub_feedback = EpochCallback(lambda epoch, logs: self.train_as.publish_feedback(TrainFeedback(i_epoch=epoch, loss=logs['loss'], acc=logs['accuracy'])))

        # ... load x_train and y_train
        # There you could load files from a path specified in a rosparam
        # For the sake of demonstration I generate a synthetic dataset
        n_samples_per_class = 1000
        x_train, y_train = self.make_synthetic_dataset(n_samples_per_class)

        self.wrapped_model.train(x_train, y_train,
                                 n_epochs=goal.epochs,
                                 callbacks=[
                                     stop_on_cancel,
                                     pub_feedback])

        # Training finished either because it was done or because it was cancelled
        if self.train_as.is_preempt_requested():
            self.train_as.set_preempted()
        else:
            self.train_as.set_succeeded()

    def predict_cb(self, req):
        rospy.loginfo("Prediction from service")
        x = np.array(req.data).reshape(-1, self.input_dim)
        i_class, confidence = self.wrapped_model.predict(x)
        return PredictResponse(i_class=i_class, confidence=confidence)

def main():
    rospy.init_node("ros_tensorflow")
    rospy.loginfo("Creating the Tensorflow model")
    ri = RosInterface()
    rospy.loginfo("ros_tensorflow node initialized")
    rate = rospy.Rate(0.5)
    while not rospy.is_shutdown():
        test_class = 0
        x, _ = ri.make_samples(i_class=test_class, n_samples=1)
        y, confidence = ri.wrapped_model.predict(x.reshape(1, -1))
        rospy.loginfo("Prediction from loop: class {} was successfully predicted: {} (confidence: {})"\
                      .format(test_class, y==test_class, confidence))

        rate.sleep()

if __name__ == "__main__":
    main()
