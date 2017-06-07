import tensorflow as tf
import numpy as np

RANDOM_SEED = 42

class Model:
    def __init__(self, gamma, actions_number, states_number, hidden_nodes_number):
        self.gamma = gamma
        self.states_number = states_number
        self.actions_number = actions_number
        self.hidden_nodes_number = hidden_nodes_number

        self.states = 1
        tf.set_random_seed(RANDOM_SEED)


    def init_weights(shape):
        weights = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(weights)
