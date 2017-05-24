# Echo client program
import socket
import sys
import random
import Agent

LINE_SEPARATOR = '\n'
BUF_SIZE = 4096 #in bytes

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def getArgs():
    try:
        return sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    except ValueError:
        print('usage: python agent_handler <server> <port> <num-episodes> [<agent-specific parameters>]')
        sys.exit(1)
        
def connect(host, port):
    try:
        sock.connect((host, port))
    except socket.error:
        print('Unable to contact environment at the given host/port.')
        sys.exit(1)
        
def sendStr(s):
    sock.send(bytes(s + LINE_SEPARATOR, encoding='utf-8'))
    
def receive(numTokens):
    data = ['']
    # print('receiving data')
    while len(data) <= numTokens:
        rawData = data[-1] + (sock.recv(BUF_SIZE)).decode('utf-8')
        del data[-1]
        data = data + rawData.split(LINE_SEPARATOR)
        
    # print('data received')
    del data[-1]
    return data
    
def sendAction(action):
    #sends all the components of the action one by one
    for a in action:
        sendStr(str(a))

#main procedure that handles the protocol
host, port, numEpisodes, agentParams = getArgs()

connect(host, port)
print('agent connected')

sendStr('GET_TASK')
data = receive(2)
stateDim = int(data[0])
actionDim = int(data[1])

print('instantiate agent')
agent = Agent.Agent(stateDim, actionDim, agentParams)


sendStr('START_LOG')
sendStr(agent.getName())

for i in range(numEpisodes):
    sendStr('START')
    data = receive(2+stateDim)
    
    terminalFlag = int(data[0])
    state = map(float, data[2:])
    action = agent.start(state)
    
    while not terminalFlag:
        sendStr('STEP')
        sendStr(str(actionDim))
        sendAction(action)
        
        data = receive(3 + stateDim)
        if not(len(data) == stateDim + 3):
            print('Communication error: calling agent.cleanup()')
            agent.cleanup()
            sys.exit(1)
            
        reward = float(data[0])
        terminalFlag = int(data[1])
        state = map(float, data[3:])
        
        action = agent.step(reward, state)
