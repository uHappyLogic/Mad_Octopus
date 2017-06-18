import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import sgd
import keras.backend as kerasBackend


class ExperienceReplay(object):
    def __init__(self, action_count, state_count, max_memory=100, discount=.5):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.action_count = action_count
        self.state_count = state_count

    def remember(self, states):
        self.memory.append([states])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size, final_reward):
        len_memory = len(self.memory)
        env_dim = self.state_count

        poll_size = min(len_memory, batch_size)

        inputs = np.zeros((poll_size, env_dim))
        targets = np.zeros((inputs.shape[0], self.action_count))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=poll_size)):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            inputs[i:i + 1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            targets[i, action_t] = reward_t + self.discount * Q_sa
            # print(targets[i, action_t])
        return inputs, targets


class Model:
    def __init__(self, learning_rate, actions_count, states_count, hidden_nodes_count):
        self.gamma = learning_rate
        self.states_count = states_count
        self.actions_count = actions_count
        self.hidden_nodes_number = hidden_nodes_count

        self.states = 1

        self.model = Sequential()
        self.model.add(Dense(self.hidden_nodes_number, input_shape=(self.states_count,), activation='relu'))
        self.model.add(Dense(self.actions_count, activation='sigmoid'))
        self.model.compile(sgd(lr=learning_rate), "mse")
        max_memory = 100

        self.exp_replay = ExperienceReplay(self.actions_count, self.states_count, max_memory=max_memory, discount=self.gamma)

    def remember(self, state_before_move, action, reward, state_after_move):
        self.exp_replay.remember([np.array([state_before_move]), action, reward, np.array([state_after_move])])

    def clear_session(self):
        kerasBackend.clear_session()
