import random
from array import array
import numpy as np
import math


class Agent:

    __name = '117272_117269'

    def __init__(self, stateDim, actionDim, agentParams):

        print("===========================================")
        print('params {0}'.format(agentParams))
        print('state dim {0} actionDim {1}'.format(stateDim, actionDim))
        print("===========================================")

        self.__stateDim = stateDim
        self.__actionDim = actionDim
        self.__action = array('d', [0 for x in range(actionDim)])
        # we ignore agentParams because our agent does not need it.
        # agentParams could be a parameter file needed by the agent.
        random.seed()
        self.step_id = 0
        self.run_log_filename = './runlogs/run_log_1'
        self.run_log_file = open(self.run_log_filename, "w+")

        self.last_step_log_line = []
        self.counter = 0

        self.neural_output_translator = {
              0: [i for i in range(15) if i % 3 == 0]
            , 1: [i for i in range(15) if i % 3 == 1]
            , 2: [i for i in range(15) if i % 3 == 2]
            , 3: [i for i in range(15, 30) if i % 3 == 0]
            , 4: [i for i in range(15, 30) if i % 3 == 1]
            , 5: [i for i in range(15, 30) if i % 3 == 2]
        }

    def __randomAction(self):
        for i in range(self.__actionDim):
            self.__action[i] = random.random()

    def __curlAction(self):
        for i in range(self.__actionDim):

            if i % 3 == 1:
                if self.counter % 100 < 50:
                    self.__action[i] = 1
                else:
                    self.__action[i] = 0
            else:
                self.__action[i] = 1

            #if i % 3 == 2:
            #    self.__action[i] = 0
            #else:
            #    self.__action[i] = 1

    def minus_vector(self, vec1, vec2):
        return [vec1[0] - vec2[0], vec1[1] - vec2[1]]


    # return minimal between snake and target
    def get_reward(self, simple_state):
        dists = []
        p3 = [9, 4]

        for i in range(0, len(simple_state) - 4, 4):
            p2p1 = self.minus_vector(simple_state[i + 4], simple_state[i])
            p1p3 = self.minus_vector(simple_state[i], p3)

            dists.append(
                np.linalg.norm(np.cross(p2p1, p1p3)) / np.linalg.norm(p2p1)
            )

        return np.min(dists)


    def __str__(self) -> str:
        return super().__str__()

    # if i % 3 == 0 or i % 3 == 2:
    #    self.__action[i] = 1
    # else:
    #    self.__action[i] = 1

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

        self.log_state(state)

        return self.__action

    def step(self, reward, state):
        print('step {0}'.format(self.step_id))
        self.step_id += 1

        self.log_reward(reward)
        self.counter += 1
        "Given current reward and state, agent returns next action"
        self.__curlAction()

        self.log_state(state)

        return self.__action

    def get_simple_state(self, state):

        simple_state = []
        l_state = list(state)

        # first coords
        simple_state.append((l_state[2] + l_state[42]) / 2)
        simple_state.append((l_state[3] + l_state[43]) / 2)

        # first velocity
        simple_state.append((l_state[4] + l_state[44]) / 2)
        simple_state.append((l_state[5] + l_state[45]) / 2)

        for i in range(36):
            if i % 8 == 0:
                # coords coordinates
                simple_state.append((l_state[i + 6] + l_state[i + 46]) / 2)
                simple_state.append((l_state[i + 7] + l_state[i + 47]) / 2)
                # velocities coordinates
                simple_state.append((l_state[i + 8] + l_state[i + 48]) / 2)
                simple_state.append((l_state[i + 9] + l_state[i + 49]) / 2)

        return simple_state

    def log_state(self, state):
        l_state = list(state)

        pos = []
        vel = []

        # position
        pos.append((l_state[2] + l_state[42]) / 2)
        pos.append((l_state[3] + l_state[43]) / 2)

        for i in range(36):
            if i % 8 == 0:
                pos.append((l_state[i + 6] + l_state[i + 46]) / 2)
                pos.append((l_state[i + 7] + l_state[i + 47]) / 2)

        vel.append((l_state[4] + l_state[44]) / 2)
        vel.append((l_state[5] + l_state[45]) / 2)

        for i in range(36):
            if i % 8 == 0:
                vel.append((l_state[i + 8] + l_state[i + 48]) / 2)
                vel.append((l_state[i + 9] + l_state[i + 49]) / 2)

        state_rep = ','.join(['{: 3.2f}'.format(el) for el in pos]) + '\n'
        state_rep += ','.join(['{: 3.2f}'.format(el) for el in vel]) + '\n'

        # self.last_step_log_line = ["%.2f" % round(el, 2) for el in self.__action]
        # self.last_step_log_line = ["%.2f" % el for el in state]

        self.last_step_log_line = [state_rep]

    def log_reward(self, reward):
        self.last_step_log_line.append(reward)

        self.run_log_file.write(','.join(map(str, self.last_step_log_line)) + ('\n--{0}-\n'.format(self.step_id)))

    def end(self, reward):

        self.log_reward(reward)

        self.run_log_file.close()

    def cleanup(self):
        pass

    def getName(self):
        return self.__name
