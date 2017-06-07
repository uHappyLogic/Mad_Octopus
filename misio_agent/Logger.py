class Logger:
    def __init__(self):
        self.run_log_filename = './runlogs/run_log_1'
        self.run_log_file = open(self.run_log_filename, "w+")
        self.last_step_log_line = []

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

    def log_reward(self, reward, step_id):
        self.last_step_log_line.append(reward)

        self.run_log_file.write(','.join(map(str, self.last_step_log_line)) + ('\n--{0}-\n'.format(step_id)))

    def close(self):
        self.run_log_file.close()
