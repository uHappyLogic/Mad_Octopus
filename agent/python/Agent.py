import random
from array import array

class Agent:
    "A template agent acting randomly"

    __name = 'Python_Random'

    def __init__(self, stateDim, actionDim, agentParams):
        "Initialize agent assuming floating point state and action"
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
            if (i % 3 == 2):
                self.__action[i] = 1
            else:
                self.__action[i] = 0

    def start(self, state):
        print('agent started')
        "Given starting state, agent returns first action"
        self.__randomAction()

        self.last_step_log_line = [self.__action, state]

        return self.__action

    def step(self, reward, state):
        print('step {0}'.format(self.step_id))
        self.step_id += 1

        self.last_step_log_line.append(reward)

        self.run_log_file.write(','.join(map(str, self.last_step_log_line)) + '\n')

        "Given current reward and state, agent returns next action"
        self.__curlAction()

        self.last_step_log_line = [self.__action, state]

        return self.__action

    def end(self, reward):
        self.run_log_file.close()

    def cleanup(self):
        pass

    def getName(self):
        return self.__name


