from array import array
import numpy as np
import math
import json
import os

from pytools import flatten
from Model import Model


class Agent:
    __name = '117272_117269'

    def __init__(self, stateDim, actionDim, agentParams):
        self.__terget_point = [9, -1]
        self.__stateDim = stateDim
        self.__actionDim = actionDim
        self.__action = array('d', [0 for x in range(actionDim)])
        self.batch_size = 50

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
        self.neural_action_count = int(math.pow(2, len(self.neural_output_translator)))

        self.model = Model(0.5, self.neural_action_count, 42, 10)

        self.model_dump_file_name = "model.h5"

        if os.path.exists("model.h5"):
            self.model.model.load_weights(self.model_dump_file_name)
        else:
            print("Model cannot be loaded")

        self.previous_state = []

    def __randomAction(self):
        self.action_idx = np.random.randint(0, self.neural_action_count)
        bin_action = bin(self.action_idx)
        self.__action_set = [int(i) for i in list(bin_action)[2:]]
        self.__action = self.unwind_action(self.__action_set)

    def __neural_action(self, state):
        neural_output = self.model.model.predict(np.array([state]))[0]

        indexed_neural_output = [[i, neural_output[i]] for i in range(len(neural_output))]
        sorted_indexed_neural_output = sorted(indexed_neural_output, key=lambda x: x[1], reverse=True)
        bests_count = 8
        bests_neural_outputs = sorted_indexed_neural_output[0:bests_count]
        best_probabilites = [x[1] for x in bests_neural_outputs]
        prob_sum = np.sum(best_probabilites)
        normalized_best_probs = [i/prob_sum for i in best_probabilites]
        print("norm best probs {0}".format(normalized_best_probs))
        print("best neural output {0}".format([ bests_neural_outputs]))

        current_prob = 0
        treshold = np.random.random()
        best_id = 0
        for i in range(0, bests_count):
            current_prob += normalized_best_probs[i]
            if treshold < current_prob:
                best_id = i
                break

        print("choosen action {0}".format(bests_neural_outputs[best_id][0]))

        self.action_idx = bests_neural_outputs[best_id][0]
        bin_action = bin(self.action_idx)
        self.__action_set = [int(i) for i in list(bin_action)[2:]]
        self.__action = self.unwind_action(self.__action_set)

    # def __curlAction(self):
    #     for i in range(self.__actionDim):
    #         if self.counter % 100 < 50:
    #             if i % 3 == 0:
    #                 self.__action[i] = 1
    #             if i % 3 == 2:
    #                 self.__action[i] = 0
    #             if i % 3 == 1:
    #                 self.__action[i] = 0
    #         else:
    #             if i % 3 == 0:
    #                 self.__action[i] = 0
    #             if i % 3 == 2:
    #                 self.__action[i] = 1
    #             if i % 3 == 1:
    #                 self.__action[i] = 0

    def get_distance_from_target(self, simple_state):
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
        self.previous_state = self.get_flat_simple_state(list(state))

        self.__neural_action(self.previous_state)

        return self.__action

    def step(self, reward, state):
        print("step idx {0}".format(self.step_id))
        state_list = list(state)
        if self.counter % self.batch_size == 1:
            inputs, targets = self.model.exp_replay.get_batch(self.model.model, self.batch_size)
            self.model.model.train_on_batch(inputs, targets)

        enhanced_reward = pow(10 - (self.get_distance_from_target(self.get_simple_state(state_list))), 2) / 100

        self.step_id += 1
        current_state = self.get_flat_simple_state(state_list)

        self.model.remember(self.previous_state, self.action_idx, enhanced_reward, current_state)

        self.previous_state = current_state
        self.__neural_action(current_state)
        self.counter += 1

        return self.__action

    def get_flat_simple_state(self, state):
        return list(flatten(self.get_simple_state(state)))

    def get_simple_state(self, state):
        simple_state = []
        l_state = state

        for i in range(2, 42, 4):
            x = (l_state[i] + l_state[i + 40]) / 2
            y = (l_state[i + 1] + l_state[i + 41]) / 2
            v_x = (l_state[i + 2] + l_state[i + 42]) / 2
            v_y = (l_state[i + 3] + l_state[i + 43]) / 2
            state_part = [x, y, v_x, v_y]
            simple_state.append(state_part)

        simple_state.append([l_state[0], l_state[1]])
        return simple_state

    def end(self, reward):
        pass

    def cleanup(self):
        inputs, targets = self.model.exp_replay.get_batch(self.model.model, self.batch_size)
        self.model.model.train_on_batch(inputs, targets)

        print("saving dump")
        self.model.model.save_weights(self.model_dump_file_name, overwrite=True)

        print("end")

    def getName(self):
        return self.__name

