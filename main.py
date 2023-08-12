import random


# Frozen Lake Environment ##############################################################################################

# This class creates the frozen lake environment where the robot will attempt to solve the frozen lake problem. It comes
# with a 4x4 and a 10x10 grid for the environment. This environment has functions that can help the various techniques
# interact with the environment or obtain relevant information about the state of the environment.
class FrozenLakeEnv:

    # 4x4 frozen lake map used for the basic implementation of the techniques. State numbers are labelled as shown. This
    # labelling convention will be used to convert the state location on the 2D grid to a number 0-15 for easier
    # implementation of computations.
    frozenLakeMap4x4 = [['S', 'F', 'F', 'F'],      # # State Numbers: 0  1  2  3
                        ['F', 'H', 'F', 'H'],      # #                4  5  6  7
                        ['F', 'F', 'F', 'H'],      # #                8  9 10 11
                        ['H', 'F', 'F', 'G']]      # #               12 13 14 15

    # 10x10 frozen lake map used for the basic implementation of the techniques. State numbers are labelled as shown.
    # This labelling convention will be used to convert the state location on the 2D grid to a number 0-99 for easier
    # implementation of computations.
    frozenLakeMap10x10 = [['S', 'H', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],      # # State Numbers:  0 to  9
                          ['F', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'F'],      # #                10 to 19
                          ['F', 'H', 'H', 'F', 'F', 'F', 'H', 'F', 'F', 'F'],      # #                20 to 29
                          ['F', 'H', 'F', 'H', 'F', 'H', 'F', 'F', 'H', 'F'],      # #                30 to 39
                          ['H', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'H'],      # #                40 to 49
                          ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],      # #                50 to 59
                          ['F', 'F', 'H', 'H', 'F', 'H', 'H', 'F', 'F', 'F'],      # #                60 to 69
                          ['H', 'H', 'F', 'F', 'F', 'F', 'H', 'F', 'H', 'F'],      # #                70 to 79
                          ['F', 'F', 'F', 'H', 'F', 'F', 'F', 'F', 'F', 'F'],      # #                80 to 89
                          ['H', 'F', 'F', 'F', 'F', 'F', 'H', 'F', 'F', 'G']]      # #                90 to 99

    # initialise variables to create the environment
    def __init__(self, mapSize):
        self.frozenLakeMap = []
        if mapSize == "4x4":
            self.frozenLakeMap = self.frozenLakeMap4x4  # set map to 4x4 version
        elif mapSize == "10x10":
            self.frozenLakeMap = self.frozenLakeMap10x10  # set map to 10x10 version
        self.numberOfStates = len(self.frozenLakeMap) * len(self.frozenLakeMap[0])  # set number of states on map
        self.numberOfActions = 4  # 4 actions represented by numbers as follows: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
        self.currentState = 0  # set current state of robot to starting point represented by number 0
        self.stateSequence = [0]  # initialise array to store sequence of states visited (used for rendering robot path)

    # renders the environment such that it shows the past and present states of the robot on the frozen lake map in
    # pictorial view
    def render(self):
        # render top border of frozen lake based on size of map
        print("------", end="")
        for i in range(len(self.frozenLakeMap[0])):
            print("---", end="")
        print()

        # render the frozen lake
        for i in range(len(self.frozenLakeMap[0])):
            print("|  ", end="")  # render left border
            for j in range(len(self.frozenLakeMap)):
                if self.currentState == i * len(self.frozenLakeMap[0]) + j:
                    print("(" + str(self.frozenLakeMap[i][j]) + ")", end='')  # show current state with a ( )
                elif i * len(self.frozenLakeMap[0]) + j in self.stateSequence:
                    print("[" + str(self.frozenLakeMap[i][j]) + "]", end='')  # show past states with  a [ ]
                else:
                    print(" " + str(self.frozenLakeMap[i][j]) + " ", end='')  # show unvisited states with proper spaces
            print("  |")  # render right border

        # render the bottom border of frozen lake based on size of map
        print("------", end="")
        for i in range(len(self.frozenLakeMap[0])):
            print("---", end="")

        # Provide legend to understand the various letters/symbols used in frozen map
        print("\nS = Start(Safe), F = Frozen Surface(Safe), H = Hole(Danger), G = Goal(Frisbee)")
        print("( ) indicates current state, [ ] indicates past states")
        print("\n")

    # moves the robot to the next state based on the action provided
    def step(self, action):
        if action == 0 and self.currentState >= len(self.frozenLakeMap[0]):
            self.currentState = self.currentState - len(self.frozenLakeMap[0])  # Move robot based on 'UP' action
        elif action == 1 and self.currentState < self.numberOfStates - len(self.frozenLakeMap[0]):
            self.currentState = self.currentState + len(self.frozenLakeMap[0])  # Move robot based on 'DOWN' action
        elif action == 2 and self.currentState % len(self.frozenLakeMap[0]) != 0:
            self.currentState = self.currentState - 1  # Move robot based on 'LEFT' action
        elif action == 3 and self.currentState % len(self.frozenLakeMap[0]) != len(self.frozenLakeMap[0]) - 1:
            self.currentState = self.currentState + 1  # Move robot based on 'RIGHT' action
        self.stateSequence.append(self.currentState)

    # resets the environment and moves robot back to starting point
    def reset(self):
        self.currentState = 0  # move robot back to starting point
        self.stateSequence = [0]  # clears the state sequence of the robot

    # checks and returns true if the robot is currently located at a terminal state, returns false otherwise
    def isTerminalState(self):
        stateRow = self.currentState // len(self.frozenLakeMap[0])  # get row of current state on map
        stateCol = self.currentState % len(self.frozenLakeMap[0])  # get column of current state on map
        stateCharacter = self.frozenLakeMap[stateRow][stateCol]  # get character of the current state
        if stateCharacter == 'G':
            return True  # True as 'G' represents the goal which is a terminal state
        elif stateCharacter == 'H':
            return True  # True as 'H' represents the holes which are terminal states
        else:
            return False  # No other terminal states exist besides 'G' and 'H'

    # returns the reward that the robot receives for reaching its current state
    def getReward(self):
        stateRow = self.currentState // len(self.frozenLakeMap[0])  # get row of current state on map
        stateCol = self.currentState % len(self.frozenLakeMap[0])  # get column of current state on map
        stateCharacter = self.frozenLakeMap[stateRow][stateCol]  # get character of the current state
        if stateCharacter == 'G':
            return 1  # Robot gets reward '+1' for reaching the goal
        elif stateCharacter == 'H':
            return -1  # Robots gets reward '-1' for falling into a hole
        else:
            return 0  # All other states give a reward of '0'

    # returns the current state of the robot
    def getCurrentState(self):
        return self.currentState


# FIRST VISIT MONTE CARLO CONTROL WITHOUT EXPLORING STARTS #############################################################

# attempts to solve the frozen lake problem  of specified size using the Monte Carlo technique with provided parameters
def MonteCarlo(e, discountVal, numberOfEpisodes, mapSize):
    # Parameters #####
    env = FrozenLakeEnv(mapSize)  # creates an instance of the frozen lake environment
    numberOfStates = 0
    if mapSize == "4x4":
        numberOfStates = 16  # set number of states for 4x4 map
    elif mapSize == "10x10":
        numberOfStates = 100  # set number of states for 10x10 map
    numberOfActions = 4  # 4 actions represented by numbers as follows: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
    # initialise 2D list to store q-values for each state-action pair
    q = [[0 for i in range(numberOfActions)] for j in range(numberOfStates)]
    # initialise 3D list to store a list of all returns for each state-action pair
    returns = [[[] for i in range(numberOfActions)] for j in range(numberOfStates)]
    # probabilities computed based on epsilon greedy policy and epsilon value e provided as parameter to MonteCarlo()
    nonGreedyProb = e / numberOfActions  # for non-greedy actions, probability = e / number of actions
    greedyProb = 1 - e + nonGreedyProb  # for greedy actions, probability = 1 - e + (e / number of actions)

    # returns the calculated returns for the previous state based on the formula: G_t = R_t+1 + (gamma x G_t+1) , where
    # G_t is the returns of the previous state, G_t+1 is the returns of the current state, R_t+1 is the reward for
    # reaching the current state & gamma is the discount value provided as a parameter to MonteCarlo()
    def getPreviousReturns(currentReturns, reward):
        return (currentReturns * discountVal) + reward

    # returns the next action that the robot should take either based on the epsilon greedy policy, or the policy that
    # always takes the greedy action. If isGreedy is true, policy that always takes greedy action is used, otherwise
    # uses epsilon greedy policy instead.
    def getNextAction(state, isGreedy):
        highestQValue = max(q[state][0], q[state][1], q[state][2], q[state][3])  # get highest q-value of all 4 actions
        # determine which actions have the highest q-value and are the greedy action
        greedyActions = []
        if q[state][0] == highestQValue:
            greedyActions.append(0)
        if q[state][1] == highestQValue:
            greedyActions.append(1)
        if q[state][2] == highestQValue:
            greedyActions.append(2)
        if q[state][3] == highestQValue:
            greedyActions.append(3)
        # if more than 1 greedy action exists, randomly select 1 to be the greedy action by eliminating the rest
        if len(greedyActions) > 1:
            while len(greedyActions) > 1:
                randomNumber = random.randint(0, len(greedyActions) - 1)
                greedyActions.pop(randomNumber)
        # returns greedy action based on policy that always takes greedy action
        if isGreedy:
            return greedyActions[0]
        # returns action based on epsilon greedy policy
        randomNumber = random.random()
        if randomNumber <= greedyProb:
            return greedyActions[0]  # return greedy action
        elif randomNumber <= greedyProb + nonGreedyProb:
            return (greedyActions[0] + 1) % 4  # return a non-greedy action
        elif randomNumber <= greedyProb + (nonGreedyProb * 2):
            return (greedyActions[0] + 2) % 4  # return a non-greedy action
        elif randomNumber <= greedyProb + (nonGreedyProb * 3):
            return (greedyActions[0] + 3) % 4  # return a non-greedy action

    # Updates q-value for provided state-action pair based on the list of returns of the particular state-action pair
    # based on formula: Q(s, a) = Average of Returns(s, a), where Q(s, a) is q-value for the state-action pair and
    # Returns(s, a) is the list of returns for the state-action pair
    def updateQValue(state, action):
        returnValues = returns[state][action]  # extracts the list of returns for the state-action pair
        q[state][action] = sum(returnValues) / len(returnValues)  # updates q-value as the average of the extracted list

    # Learning #####
    for i in range(numberOfEpisodes):  # iterate through episodes
        # Generate an episode #
        env.reset()  # reset environment for the start of episode
        stateSequence = [0]  # initialise the state sequence to store the order of states visited and add starting state
        actionSequence = []  # initialise the action sequence to store the order of actions taken
        rewardSequence = []  # initialise the reward sequence to store the order of rewards received
        endOfSequence = False  # flag that breaks the loop when robot reaches a terminal state and episode ends
        while not endOfSequence:  # run episode as long as robot has not reached a terminal state
            actionSequence.append(getNextAction(stateSequence[-1], isGreedy=False))  # choose action based on e-greedy
            env.step(actionSequence[-1])  # let robot take the chosen action in the environment
            rewardSequence.append(env.getReward())  # get reward from environment for reaching a state
            stateSequence.append(env.getCurrentState())  # update current state of robot in state sequence
            if env.isTerminalState():
                endOfSequence = True  # flag true as robot has reached a terminal state
        # Compute Returns #
        currentReturns = 0  # initialise current returns of last state
        tempReturns = []  # initialise a list to store returns in reverse order (from last state to first state)
        for j in range(len(actionSequence)):
            # compute previous returns to update current returns due to working backwards, and update returns list
            currentReturns = getPreviousReturns(currentReturns, rewardSequence.pop())
            tempReturns.append(currentReturns)
        tempReturns.reverse()  # reverse order to restore first state to last state order
        # Update q values #
        # Initialise a 2D list to remember which state-action pairs have been visited before, true if visited before,
        # false otherwise
        stateActionVisitedBefore = [[False for j in range(numberOfActions)] for k in range(numberOfStates)]
        for j in range(len(actionSequence)):
            if not stateActionVisitedBefore[stateSequence[j]][actionSequence[j]]:
                returns[stateSequence[j]][actionSequence[j]].append(tempReturns[j])  # update returns for first visit
                stateActionVisitedBefore[stateSequence[j]][actionSequence[j]] = True  # state-action pair visited now
                updateQValue(stateSequence[j], actionSequence[j])  # update q-values based on first visit return values

    # Final policy #####
    env.reset()  # reset environment before displaying final policy
    for i in range(numberOfStates):  # loop a finite number of times to cut off policies which never go terminate states
        # move robot according to optimal policy by always taking greedy action
        env.step(getNextAction(env.getCurrentState(), isGreedy=True))
        if env.isTerminalState():
            break  # exit loop as robot has reached terminal state
    env.render()  # render environment to show path taken by robot


# SARSA/Q-LEARNING #####################################################################################################

# attempts to solve the frozen lake problem  of specified size using a temporal difference technique with provided
# parameters and provided choice of algorithm between SARSA & Q-learning
def TemporalDifference(e, discountVal, learningRate, numberOfEpisodes, mapSize, algorithm):
    # parameters #####
    env = FrozenLakeEnv(mapSize)  # creates an instance of the frozen lake environment
    numberOfStates = 0
    if mapSize == "4x4":
        numberOfStates = 16  # set number of states for 4x4 map
    elif mapSize == "10x10":
        numberOfStates = 100  # set number of states for 10x10 map
    numberOfActions = 4  # 4 actions represented by numbers as follows: 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
    # initialise 2D list to store q-values for each state-action pair
    q = [[0 for i in range(numberOfActions)] for j in range(numberOfStates)]
    # probabilities computed based on epsilon greedy policy and epsilon value e provided as parameter to
    # TemporalDifference()
    nonGreedyProb = e / numberOfActions  # for non-greedy actions, probability = e / number of actions
    greedyProb = 1 - e + nonGreedyProb  # for greedy actions, probability = 1 - e + (e / number of actions)

    # returns the next action that the robot should take either based on the epsilon greedy policy, or the policy that
    # always takes the greedy action. If isGreedy is true, policy that always takes greedy action is used, otherwise
    # uses epsilon greedy policy instead.
    def getNextAction(state, isGreedy):
        highestQValue = max(q[state][0], q[state][1], q[state][2], q[state][3])  # get highest q-value of all 4 actions
        # determine which actions have the highest q-value and are the greedy action
        greedyActions = []
        if q[state][0] == highestQValue:
            greedyActions.append(0)
        if q[state][1] == highestQValue:
            greedyActions.append(1)
        if q[state][2] == highestQValue:
            greedyActions.append(2)
        if q[state][3] == highestQValue:
            greedyActions.append(3)
        # if more than 1 greedy action exists, randomly select 1 to be the greedy action by eliminating the rest
        if len(greedyActions) > 1:
            while len(greedyActions) > 1:
                randomNumber = random.randint(0, len(greedyActions) - 1)
                greedyActions.pop(randomNumber)
        # returns greedy action based on policy that always takes greedy action
        if isGreedy:
            return greedyActions[0]
        # returns action based on epsilon greedy policy
        randomNumber = random.random()
        if randomNumber <= greedyProb:
            return greedyActions[0]  # return greedy action
        elif randomNumber <= greedyProb + nonGreedyProb:
            return (greedyActions[0] + 1) % 4  # return a non-greedy action
        elif randomNumber <= greedyProb + (nonGreedyProb * 2):
            return (greedyActions[0] + 2) % 4  # return a non-greedy action
        elif randomNumber <= greedyProb + (nonGreedyProb * 3):
            return (greedyActions[0] + 3) % 4  # return a non-greedy action

    # Updates q-value for provided state-action pair based on the update rule formula:
    # Q(s, a) = Q(s, a) + alpha x (R + (gamma x Q(s',a')) - Q(s, a)) for SARSA
    # Q(s, a) = Q(s, a) + alpha x (R + (gamma x maxQ(s',a')) - Q(s, a)) for Q-learning
    # where Q(s, a) is the q-value of the current state-action pair, Q(s', a') is the q-value of the next state-action
    # pair, maxQ(s', a') is the q-value of the next state-action pair when action taken is greedy, alpha is the learning
    # rate passed as parameter to TemporalDifference() & gamma is the discount value also passed as parameter to
    # TemporalDifference(). Function does not need to know whether algorithm is SARSA or Q-learning as the next
    # action passed to function will correspond to next action for SARSA and greedy action for Q-learning
    def updateQValue(currentState, currentAction, reward, nextState, nextAction):
        tempQ = reward
        tempQ += discountVal * q[nextState][nextAction]
        tempQ -= q[currentState][currentAction]
        tempQ = q[currentState][currentAction] + (learningRate * tempQ)
        q[currentState][currentAction] = tempQ

    # Learning #####
    for i in range(numberOfEpisodes):  # iterate through episodes
        # Generate an episode
        env.reset()  # reset environment for the start of episode
        currentState = env.getCurrentState()  # initialise current state
        currentAction = getNextAction(currentState, isGreedy=False)  # initialise current action by choosing an action
        while True:  # run episode until loop is broken upon reaching terminal state
            env.step(currentAction)  # let robot take the chosen action in the environment
            nextState = env.getCurrentState()  # initialise next state with current state as robot has already moved
            nextAction = getNextAction(nextState, isGreedy=False)  # initialise next action by choosing an action
            if algorithm == "SARSA":
                # q-value updated based on update rule for SARSA
                updateQValue(currentState, currentAction, env.getReward(), nextState, nextAction)
            elif algorithm == "QLearning":
                # q-value updated based on update rule for Q-learning
                greedyNextAction = getNextAction(nextState, isGreedy=True)  # get greedy action used to update q-value
                updateQValue(currentState, currentAction, env.getReward(), nextState, greedyNextAction)
            currentState = nextState  # update the current state to the next state
            currentAction = nextAction  # update the current action to the next action
            if env.isTerminalState():
                break  # break loop as robot has reached a terminal state and episode has ended

    # Final policy #####
    env.reset()  # reset environment before displaying final policy
    for i in range(numberOfStates):  # loop a finite number of times to cut off policies which never go terminate states
        # move robot according to optimal policy by always taking greedy action
        env.step(getNextAction(env.getCurrentState(), isGreedy=True))
        if env.isTerminalState():
            break  # exit loop as robot has reached terminal state
    env.render()  # render environment to show path taken by robot


# attempts to solve the frozen lake problem  of specified size using the SARSA technique with provided parameters.
# Technique implementation is in the TemporalDifference() function.
def SARSA(e, discountVal, learningRate, numberOfEpisodes, mapSize):
    return TemporalDifference(e, discountVal, learningRate, numberOfEpisodes, mapSize, algorithm="SARSA")


# attempts to solve the frozen lake problem  of specified size using the Q-learning technique with provided parameters
# Technique implementation is in the TemporalDifference() function.
def QLearning(e, discountVal, learningRate, numberOfEpisodes, mapSize):
    return TemporalDifference(e, discountVal, learningRate, numberOfEpisodes, mapSize, algorithm="QLearning")


# MAIN CODE ############################################################################################################

print("\n\nThese are the policies for the 4x4 and 10x10 map.")
print("Note that some techniques may not find solutions 100% of the time.")
print("In particular, Monte Carlo never finds a solution for the 10x10 map.")

# Basic Implementation #####
print("\n\n< < < < < BASIC IMPLEMENTATION (4 x 4) > > > > >\n\n")

print("\nFIRST VISIT MONTE CARLO CONTROL WITHOUT EXPLORING STARTS")
# Monte Carlo technique on 4x4 map
MonteCarlo(e=0.1, discountVal=1, numberOfEpisodes=1000, mapSize="4x4")

print("\nSARSA")
# SARSA technique on 4x4 map
SARSA(e=0, discountVal=1, learningRate=0.1, numberOfEpisodes=1000, mapSize="4x4")

print("\nQ-LEARNING")
# Q-learning technique on 4x4 map
QLearning(e=0, discountVal=1, learningRate=0.1, numberOfEpisodes=1000, mapSize="4x4")


# Extended Implementation #####
print("\n\n< < < < < EXTENDED IMPLEMENTATION (10 x 10) > > > > >\n\n")

print("\nFIRST VISIT MONTE CARLO CONTROL WITHOUT EXPLORING STARTS")
# Monte Carlo technique on 10x10 map
MonteCarlo(e=0.1, discountVal=1, numberOfEpisodes=1000, mapSize="10x10")

print("\nSARSA")
# SARSA technique on 10x10 map
SARSA(e=0, discountVal=1, learningRate=0.1, numberOfEpisodes=1000, mapSize="10x10")

print("\nQ-LEARNING")
# Q-learning technique on 10x10 map
QLearning(e=0, discountVal=1, learningRate=0.1, numberOfEpisodes=1000, mapSize="10x10")

# END OF CODE #
