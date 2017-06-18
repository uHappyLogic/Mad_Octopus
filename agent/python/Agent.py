from array import array
import numpy as np
import math
import json
import copy
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
        self.batch_size = 30
        self.hidden_nodes_count = 84
        self.random_actions_count = 1
        self.final_reward = -100.0
        self.step_id = 0
        self.counter = 0
        self.max_steps = 1000
        self.states_count = 42
        self.learning_on = True
        self.last_reward = 0.0
        self.neural_action_on = True
        self.neural_output_translator = {
            0: [i for i in range(15) if i % 3 == 0]
            , 1: [i for i in range(15) if i % 3 == 1]
            , 2: [i for i in range(15) if i % 3 == 2]
            , 3: [i for i in range(15, 30) if i % 3 == 0]
            , 4: [i for i in range(15, 30) if i % 3 == 1]
            , 5: [i for i in range(15, 30) if i % 3 == 2]
        }
        self.part_actions_count = len(self.neural_output_translator)
        self.neural_action_count = int(math.pow(2, self.part_actions_count))

        self.model = Model(0.9, self.neural_action_count, self.states_count, self.hidden_nodes_count)

        self.model_dump_file_name = "model_batch_reward.h27"

        if os.path.exists(self.model_dump_file_name):
            self.model.model.load_weights(self.model_dump_file_name)
        else:
            print("Model cannot be loaded")

        self.previous_state = []

    def __randomAction(self):
        idx = np.random.randint(0, self.neural_action_count)
        self.__set_action(idx)

    def __set_action(self, idx):
        self.action_idx = idx
        bin_action = bin(self.action_idx)[2:].zfill(self.part_actions_count)
        self.__action_set = [int(i) for i in list(bin_action)[:]]
        self.__action = self.unwind_action(self.__action_set)

    def __good_action(self):
        action_step = self.step_id % 150
        # if action_step< 28:
        #     self.__set_action(33)
        # else:
        #     self.__set_action(30)
        wait_steps = 15+self.init_state[37]*2
        if action_step < wait_steps:
            self.__set_action(int('001001', 2))
        else:
            if action_step < 65:
                self.__set_action(int('010010',2))  # 011101 33
            else:
                if action_step < 100:
                    self.__set_action(int('100110',2))  # 100100
                else:
                    self.__set_action(9)
        #
        # if self.previous_state[41] < -1:
        #     if action_step < 30:
        #         self.__set_action(54)
        #     else:
        #         if action_step < 65:
        #             print('tutaj')
        #             self.__set_action(33)  # 011101 33
        #         else:
        #             if action_step < 100:
        #                 self.__set_action((36))  # 100100
        #             else:
        #                 self.__set_action(9)
        #
        # else:
        #     if action_step < 30:
        #         self.__set_action(int('100100', 2))
        #     else:
        #         if action_step < 65:
        #             self.__set_action(int('010010', 2))  #
        #         else:
        #             if action_step < 100:
        #                 self.__set_action(int('110110', 2))
        #             else:
        #                 self.__set_action(25)
    def __get_best_neural_action(self, state):
        neural_output = self.model.model.predict(np.array([state]))[0]
        self.__set_action(np.argmax(neural_output))

    def __neural_action(self, state):
        if self.random_actions_count == 1:
            self.__get_best_neural_action(state)
            return
        neural_output = self.model.model.predict(np.array([state]))[0]

        indexed_neural_output = [[i, neural_output[i]] for i in range(len(neural_output))]
        sorted_indexed_neural_output = sorted(indexed_neural_output, key=lambda x: x[1], reverse=True)
        bests_count = self.random_actions_count
        bests_neural_outputs = sorted_indexed_neural_output[0:bests_count]
        best_probabilites = [x[1] for x in bests_neural_outputs]
        prob_sum = np.sum(best_probabilites)
        normalized_best_probs = [i / prob_sum for i in best_probabilites]
        # print("norm best probs {0}".format(normalized_best_probs))
        # print("best neural output {0}".format([bests_neural_outputs]))

        current_prob = 0
        treshold = np.random.random()
        best_id = 0
        for i in range(0, bests_count):
            current_prob += normalized_best_probs[i]
            if treshold < current_prob:
                best_id = i
                break

        # print("choosen action {0}".format(bests_neural_outputs[best_id][0]))
        self.__set_action(bests_neural_outputs[best_id][0])

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
        self.init_state = copy.deepcopy(self.previous_state)
        #print(self.init_state)
        return self.__action

    def get_enhanced_reward(self, state_list):
        actual_reward = (-1 * self.get_distance_from_target(self.get_simple_state(state_list))) /10
        self.last_reward = actual_reward
        return actual_reward

    def step(self, reward, state):
        # print("step idx {0}".format(self.step_id))

        state_list = list(state)
        if self.learning_on:
            if (self.counter % self.batch_size == 0) and self.counter > 0:
                inputs, targets = self.model.exp_replay.get_batch(self.model.model, self.batch_size, self.final_reward)
                self.model.model.train_on_batch(inputs, targets)
                self.final_reward = -100.0
        enhanced_reward = self.get_enhanced_reward(state_list)
        self.final_reward = max(self.final_reward, enhanced_reward)
        #print("enhanced_reward {0}".format(enhanced_reward))
        self.step_id += 1
        current_state = self.get_flat_simple_state(state_list)

        if self.learning_on:
            self.model.remember(self.previous_state, self.action_idx, enhanced_reward, current_state)

        self.previous_state = current_state
        if self.neural_action_on:
            self.__neural_action(current_state)
        else:
            self.__good_action()
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
        self.final_reward = (self.max_steps - self.step_id) / 100
        self.model.remember(self.previous_state, self.action_idx, self.final_reward, self.previous_state)
        inputs, targets = self.model.exp_replay.get_batch(self.model.model, self.batch_size, self.final_reward)
        self.model.model.train_on_batch(inputs, targets)

        print("final reward {0}".format(self.final_reward))
        print("saving dump")
        self.model.model.save_weights(self.model_dump_file_name, overwrite=True)
        self.model.clear_session()
        print("end")

    def getName(self):
        return self.__name
