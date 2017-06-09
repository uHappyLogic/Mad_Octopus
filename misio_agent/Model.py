import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


class Model:
    def __init__(self, gamma, actions_number, states_number, hidden_nodes_number):
        self.gamma = gamma
        self.states_number = states_number
        self.actions_number = actions_number
        self.hidden_nodes_number = hidden_nodes_number

        self.states = 1

        hidden_size = 100

        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(self.states_number,), activation='relu'))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(self.actions_number, activation='sigmoid'))
        model.compile(sgd(lr=.2), "mse")
        max_memory = 10

        self.exp_replay = ExperienceReplay(max_memory=max_memory)

    def remember(self, input_tm1, action, reward, input_t, game_over):
        self.exp_replay.remember([input_tm1, action, reward, input_t], game_over)


