import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd


class ExperienceReplay(object):
    def __init__(self, action_count, state_count, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.action_count = action_count
        self.state_count = state_count

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        env_dim = self.state_count
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], self.action_count))
        print(targets)
        for i, idx in enumerate(np.random.randint(0, len_memory, size=[2])):
            print('iterator: {0}'.format(i))
            print('idx: {0}'.format(idx))
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            print("action_t: {0}".format(action_t))
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
    def __init__(self, gamma, actions_count, states_count, hidden_nodes_count):
        self.gamma = gamma
        self.states_count = states_count
        self.actions_count = actions_count
        self.hidden_nodes_number = hidden_nodes_count

        self.states = 1

        self.model = Sequential()
        self.model.add(Dense(self.hidden_nodes_number, input_shape=(self.states_count, ), activation='relu'))
        self.model.add(Dense(self.hidden_nodes_number, activation='relu'))
        self.model.add(Dense(self.actions_count, activation='sigmoid'))
        self.model.compile(sgd(lr=.2), "mse")
        max_memory = 10

        self.exp_replay = ExperienceReplay(self.actions_count, self.states_count, max_memory=max_memory, discount=0.9)

    def remember(self, state_before_move, action, reward, state_after_move, game_over):
        self.exp_replay.remember([np.array([state_before_move]), action, reward, np.array([state_after_move])], game_over)
