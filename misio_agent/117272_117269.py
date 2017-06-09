import random
from array import array
import numpy as np
import math

from pytools import flatten

from Logger import Logger
from Model import Model


class Agent:
    __name = '117272_117269'

    def __init__(self, stateDim, actionDim, agentParams):

        print("===========================================")
        print('params {0}'.format(agentParams))
        print('state dim {0} actionDim {1}'.format(stateDim, actionDim))
        print("===========================================")
        # self.__logger = Logger()
        self.__terget_point = [9, -1]
        self.__stateDim = stateDim
        self.__actionDim = actionDim
        self.__action = array('d', [0 for x in range(actionDim)])
        self.batch_size = 50

        random.seed()
        self.step_id = 0
        self.counter = 0

        self.neural_output_translator = {
            0: [i for i in range(15) if i % 3 == 0]
            , 1: [i for i in range(15) if i % 3 == 1]
            , 2: [i for i in range(15) if i % 3 == 2]
            , 3: [i for i in range(15, 30) if i % 3 == 0]
            , 4: [i for i in range(15, 30) if i % 3 == 1]
            , 5: [i for i in range(15, 30) if i % 3 == 2]
        }

        self.model = Model(0.5, len(self.neural_output_translator), 42, 10)

        self.state_before_move = []

    def __randomAction(self):
        for i in range(self.__actionDim):
            self.__action[i] = random.random()

    def __curlAction(self):
        for i in range(self.__actionDim):
            if self.counter % 100 < 50:
                if i % 3 == 0:
                    self.__action[i] = 1
                if i % 3 == 2:
                    self.__action[i] = 0
                if i % 3 == 1:
                    self.__action[i] = 0
            else:
                if i % 3 == 0:
                    self.__action[i] = 0
                if i % 3 == 2:
                    self.__action[i] = 1
                if i % 3 == 1:
                    self.__action[i] = 0

    # return minimal between snake and target
    def get_reward(self, simple_state):
        dists = []
        for i in range(len(simple_state)):
            dists.append(
                math.hypot(self.__terget_point[0] - simple_state[i][0], self.__terget_point[1] - simple_state[i][1]))
        return np.min(dists)

    def unwind_action(self, neural_output):

        result = np.zeros(30)

        for i in range(len(neural_output)):
            for k in self.neural_output_translator[i]:
                result[k] = neural_output[i]

        return result

    def start(self, state):
        print('agent started')
        "Given starting state, agent returns first action"
        self.__randomAction()
        # self.__logger.log_state(state)

        self.state_before_move = self.get_flat_simple_state(list(state))

        print(self.state_before_move)

        return self.__action

    def step(self, reward, state):
        state_list = list(state)
        if self.counter % self.batch_size == 1:
            inputs, targets = self.model.exp_replay.get_batch(self.model.model, self.batch_size)
            print('====================================\n{0}\n====================================\n{1}\n===================================='
                  .format(inputs, targets))

        enhanced_reward = 30 - (self.get_reward(self.get_simple_state(state_list)) + reward)

        self.step_id += 1
        # self.__logger.log_reward(enhanced_reward, self.step_id)
        print(enhanced_reward)

        after_state = self.get_flat_simple_state(state_list)
        self.model.remember(self.state_before_move, self.__action, enhanced_reward, after_state, False)

        self.state_before_move = after_state
        self.__curlAction()

        self.counter += 1

        return self.__action

    def get_flat_simple_state(self, state):
        return list(flatten(self.get_simple_state(state)))

    def get_simple_state(self, state):
        simple_state = []
        l_state = state
        # average points
        for i in range(2, 42, 4):
            print(i)
            x = (l_state[i] + l_state[i + 40]) / 2
            y = (l_state[i + 1] + l_state[i + 41]) / 2
            v_x = (l_state[i + 2] + l_state[i + 42]) / 2
            v_y = (l_state[i + 3] + l_state[i + 43]) / 2
            state_part = [x, y, v_x, v_y]
            simple_state.append(state_part)
        # start_angle_and_velocity
        simple_state.append([l_state[0], l_state[1]])
        return simple_state

    def end(self, reward):
        pass
        # self.__logger.log_reward(reward, self.step_id)
        # self.__logger.close()

    def cleanup(self):
        pass

    def getName(self):
        return self.__name
