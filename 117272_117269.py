import random
from array import array


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
        self.run_log_filename = '/home/uhappylogic/Workspace/misio-labs/octopus_arm/runlogs/run_log_1'
        self.run_log_file = open(self.run_log_filename, "w+")

        self.last_step_log_line = []

    def __randomAction(self):
        for i in range(self.__actionDim):
            self.__action[i] = random.random()

    def __curlAction(self):
        for i in range(self.__actionDim):
            if i % 3 == 2:
                self.__action[i] = 1
            else:
                self.__action[i] = 0

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

        "Given current reward and state, agent returns next action"
        self.__randomAction()

        self.log_state(state)

        return self.__action

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
