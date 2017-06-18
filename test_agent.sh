#!/bin/bash

PORT=65399
PYTHON=python3 # Modify to you needs
SLEEP_TIME=0.5   # You might have to increase it in case of "unable to connect to port" problems

if [ "$#" -ne 3 ]; then
    echo "usage:"
    echo "  test_agent.sh <TestDir> <AgentName.py> external_gui"
    exit 1
fi

TESTDIR=$1
GUITYPE=$3
AGENTFILE=$2
AGENT=`basename ${AGENTFILE%.*}`

# Create directories and remove old logs
RESDIR=results/${AGENT}
mkdir -p $RESDIR 2>/dev/null
rm *.log $RESDIR/*.log 2>/dev/null
cp $AGENTFILE agent/python/Agent.py
cp ./misio_agent/Logger.py agent/python/Logger.py
cp ./misio_agent/Model.py agent/python/Model.py

# Kill all servers that might have not been closed properly earlier
#kill `ps -x |grep environment/octopus-environment.jar|head -n1 |cut -d " " -f1`
#kill `ps|grep environment/octopus-environment.jar|cut -d' ' -f1|xargs` 2>/dev/null
kill `ps -x |grep environment/octopus-environment.jar|head -n1 |awk '{print $1}'`
sleep 1

# Mean of each line (sum for each value in line)
meansum()
{
    python3 -c "import sys; from numpy import mean; print(mean([sum(float(r) for r in line.split()) for line in sys.stdin]))"
}

while :
do
    for instance in $TESTDIR/*.xml; do
        test=`basename ${instance%.*}`
        printf "%-12s" "$test "

        # Run the server
        java -Djava.endorsed.dirs=environment/lib -jar environment/octopus-environment.jar $3 $instance $PORT &
        server_pid=$!
        printf $server_pid

        #printf "server running\n"

        # run the agent
        pushd agent/python >/dev/null
        exit_status=1
        while true; do
            $PYTHON agent_handler.py localhost $PORT 1
            if [ $? -eq 0 ]; then break; fi
            sleep $SLEEP_TIME # I need some time here to let the port be created
            printf "sleeping time\n"
        done
        popd >/dev/null
        sleep $SLEEP_TIME # Waste of time
        #printf "killing server\n"
        # Kill the server
        #kill $server_pid
        kill `ps -x |grep environment/octopus-environment.jar|head -n1 |awk '{print $1}'`
        sleep $SLEEP_TIME # Waste of time, but I need to wait for OS to reclaim the port

        # Move the results to results/agent/*.log and print the sum of rewards
        logfile=`ls *.log`
        #printf "%6.2f\n" `cat "$logfile" | meansum`
        mv $logfile $RESDIR/${test}.log
    done
done

# Average reward over all tests
printf "\nAverage reward "
cat $RESDIR/*.log | meansum
