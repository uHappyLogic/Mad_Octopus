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

        self.previous_state = []

    def __randomAction(self):
        self.action_idx = np.random.randint(0, self.neural_action_count)
        bin_action = bin(self.action_idx)
        self.__action_set = [int(i) for i in list(bin_action)[2:]]
        self.__action = self.unwind_action(self.__action_set)
        #for i in range(self.__actionDim):
        #    self.__action[i] = random.random()

    def __neural_action(self, state):
        neural_output = self.model.model.predict(np.array([state]))[0]

        indexed_neural_output = [[i, neural_output[i]] for i in range(len(neural_output))]
        sorted_indexed_neural_output = sorted(indexed_neural_output, key=lambda x: x[1], reverse=True)
        bests_count = 8
        bests_neural_outputs = sorted_indexed_neural_output[0:bests_count]
        best_probabilites = [x[1] for x in bests_neural_outputs]
        prob_sum = np.sum(best_probabilites)
        print("prob sum {0}".format(prob_sum))
        normalized_best_probs = [i/prob_sum for i in best_probabilites]
        print("norm best probs {0}".format(normalized_best_probs))
        print("best neural output {0}".format(bests_neural_outputs))

        current_prob = 0
        treshold = np.random.random()
        best_id = 0
        for i in range(0, bests_count):
            current_prob += normalized_best_probs[i]
            if treshold < current_prob:
                best_id = i
                break

        print(neural_output)
        self.action_idx = bests_neural_outputs[best_id][0]
        print('argmax = {0}'.format(self.action_idx))
        bin_action = bin(self.action_idx)
        self.__action_set = [int(i) for i in list(bin_action)[2:]]
        self.__action = self.unwind_action(self.__action_set)

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
        print('agent started')
        "Given starting state, agent returns first action"
        self.__randomAction()
        # self.__logger.log_state(state)

        self.previous_state = self.get_flat_simple_state(list(state))

        print(self.previous_state)

        return self.__action

    def step(self, reward, state):
        print("step idx {0}".format(self.step_id))
        state_list = list(state)
        if self.counter % self.batch_size == 1:
            inputs, targets = self.model.exp_replay.get_batch(self.model.model, self.batch_size)
            print("inputs")
            print(inputs)
            print("targets")
            print(targets)
            loss = self.model.model.train_on_batch(inputs, targets)
            print("loss: {0}".format(loss))

        enhanced_reward = pow(10 - (self.get_distance_from_target(self.get_simple_state(state_list))), 2) / 100

        self.step_id += 1
        # self.__logger.log_reward(enhanced_reward, self.step_id)
        print("enhanced reward {0}".format(enhanced_reward))

        current_state = self.get_flat_simple_state(state_list)
        self.model.remember(self.previous_state, self.action_idx, enhanced_reward, current_state, False)

        self.previous_state = current_state
        #self.__curlAction()
        #self.__randomAction()
        self.__neural_action(current_state)
        self.counter += 1

        return self.__action

    def get_flat_simple_state(self, state):
        return list(flatten(self.get_simple_state(state)))

    def get_simple_state(self, state):
        simple_state = []
        l_state = state
        # average points
        for i in range(2, 42, 4):
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

    def __str__(self) -> str:
        return super().__str__()
