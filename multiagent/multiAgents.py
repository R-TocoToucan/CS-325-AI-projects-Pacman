#THIS  CODE  WAS MY OWN WORK , IT WAS  WRITTEN  WITHOUT  CONSULTING  ANY
#SOURCES  OUTSIDE  OF  THOSE  APPROVED  BY THE  INSTRUCTOR - MINSEOK CHOI
#MCHOI49@emory.edu/mchoi49

# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
#
# Modified by Eugene Agichtein for CS325 Sp 2014 (eugene@mathcs.emory.edu)
#

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        Note that the successor game state includes updates such as available food,
        e.g., would *not* include the food eaten at the successor state's pacman position
        as that food is no longer remaining.
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        currentFood = currentGameState.getFood() #food available from current state
        newFood = successorGameState.getFood() #food available from successor state (excludes food@successor) 
        currentCapsules=currentGameState.getCapsules() #power pellets/capsules available from current state
        newCapsules=successorGameState.getCapsules() #capsules available from successor (excludes capsules@successor)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0.0

        # if distance to ghost is 1, heavily reduce score to avoid ghost unless eaten pallet
        # when found food, give score in eating it.

        # set initial newGhostDistance value with large number to find min distance between ghosts

        if successorGameState.isWin():
            score += 100000

        newGhostDistance = 99999
        for ghost in successorGameState.getGhostPositions():
            ghostDistance = manhattanDistance(newPos, ghost)
            newGhostDistance = min(ghostDistance, newGhostDistance)

        if newGhostDistance <= 2:
            if newScaredTimes[0] != 0:
                score += 500
            else:
                score -= 500

        # find min distance to food and add weight based on distance
        newFoodDistance = 99999
        for food in newFood.asList():
            foodDistance = manhattanDistance(newPos, food)
            newFoodDistance = min(foodDistance, newFoodDistance)


        # if pacman eats food, add to score
        if  currentGameState.getNumFood() > successorGameState.getNumFood():
            score += 100
        else:
            score -= 2*newFoodDistance

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # for both max/min return evaluation function at 0 depth/terminal state
        # then, find all possible moves from getLegalActions and return Max/Min values for pacman/ghosts
        # value = current value vs opponent's value on successor node.
        def maxValue(gameState, depth):

            value = -float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(0)
            for moves in nextMove:
                value = max(value, minValue(gameState.generateSuccessor(0, moves), depth, 1))

            return value

        def minValue(gameState, depth, agent):

            value = float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(agent)
            # there can be more than 1 ghost. if then, pass the value to the next ghost
            if agent == gameState.getNumAgents() - 1:
                for moves in nextMove:
                    value = min(value, maxValue(gameState.generateSuccessor(agent, moves), depth-1))
            else:
                for moves in nextMove:
                    value = min(value, minValue(gameState.generateSuccessor(agent, moves), depth, agent+1))

            return value

        value = -float("inf")
        action = Directions.STOP
        nextMove = gameState.getLegalActions(0)
        for moves in nextMove:
            value2 = value
            value = max(value2, minValue(gameState.generateSuccessor(0, moves), self.depth, 1))
            if value > value2:
                action = moves

        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # add alpha, beta values set to -/+ infinity compare with max/min score
        # alpha = maximizer's known best choice, beta = minimizer's known best choice
        # if alpha >= beta, the agent can assume a better choice was made
        # because at maximizer's turn, finding a value greater than alpha updates alpha and lesser updates beta
        # this works similarly for minimizer.
        def maxValue(gameState, depth, alpha, beta):

            value = -float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(0)
            for moves in nextMove:
                value = max(value, minValue(gameState.generateSuccessor(0, moves), depth, 1, alpha, beta))
                alpha = max(alpha, value)
                if value > beta:
                    return value

            return value

        def minValue(gameState, depth, agent, alpha, beta):

            value = float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(agent)
            # there can be more than 1 ghost. if then, pass the value to the next ghost
            if agent == gameState.getNumAgents() - 1:
                for moves in nextMove:
                    value = min(value, maxValue(gameState.generateSuccessor(agent, moves), depth - 1, alpha, beta))
                    beta = min(beta, value)
                    if value < alpha:
                        return value
            else:
                for moves in nextMove:
                    value = min(value, minValue(gameState.generateSuccessor(agent, moves), depth, agent + 1, alpha, beta))
                    beta = min(beta, value)
                    if value < alpha:
                        return value

            return value

        alpha = -float("inf")
        beta = float("inf")
        value = -float("inf")

        action = Directions.STOP
        nextMove = gameState.getLegalActions(0)

        for moves in nextMove:
            value2 = value
            value = max(value2, minValue(gameState.generateSuccessor(0, moves), self.depth, 1, alpha, beta))
            if value > value2:
                action = moves

            if value >= beta:
                return action

            alpha = max(alpha, value)

        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def maxValue(gameState, depth):

            value = -float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(0)
            for moves in nextMove:
                value = max(value, expectValue(gameState.generateSuccessor(0, moves), depth, 1))

            return value

        def expectValue(gameState, depth, agent):

            value = 0
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(agent)

            if agent == gameState.getNumAgents()-1:
                for moves in nextMove:
                    value += maxValue(gameState.generateSuccessor(agent, moves), depth-1)
            else:
                for moves in nextMove:
                    value += expectValue(gameState.generateSuccessor(agent, moves), depth, agent+1)

            prob = value/len(nextMove)
            return prob

        value = -float("inf")
        action = Directions.STOP
        nextMove = gameState.getLegalActions(0)

        for moves in nextMove:
            value2 = expectValue(gameState.generateSuccessor(0, moves), self.depth, 1)
            if value2 > value:
                value = value2
                action = moves

        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

    - The better evaluation function will be based on the evaluation function used on reflexagent
    The goal is to make pacman more active on capsule use and ghost chasing than the original evaluation function.


    Considering the settings of the map where ghosts cooperate better and having more capsules available for use
    where safer approach of avoiding ghosts have little chance of survival,
    the pacman will take more cavalier approach and get involved in risky moves if rewards are worth it.
    Taking capsules and chasing off scared ghosts will be much more heavily weighted compared to getting food,
    yet if the distance between the ghosts and other capsules are far, the pacman will act more
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()  # food available from current state
    currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
    newCapsules = currentGameState.getCapsules()  # capsules available from successor (excludes capsules@successor)
    newGhostStates = currentGameState.getGhostStates()

    moves = currentGameState.getLegalActions(0)

    score = 0.0

    foodDistance = []
    capsuleDistance = []

    # if distance to ghost is 1, heavily reduce score to avoid ghost unless eaten pallet
    # when found food, give score in eating it.

    # set initial newGhostDistance value with large number to find min distance between ghosts


    for ghost in newGhostStates:
        ghostDistance = manhattanDistance(currentPos, ghost.getPosition())

        if ghostDistance <= 2:
            if ghost.scaredTimer == 0:
                score -= 1000/(ghostDistance+1)
            elif ghost.scaredTimer != 0:
                score += 1000/(ghostDistance+1)


    # find min distance to food and add weight based on distance
    for food in currentFood.asList():
        foodDistance.append(manhattanDistance(currentPos, food))
        if food == currentPos:
            score += 300

    minFoodDist = min(foodDistance + [10])

    for capsule in newCapsules:
        capsuleDistance.append(manhattanDistance(currentPos, capsule))
        if capsule == currentPos:
            score += 500

    minCapsuleDist = min(capsuleDistance + [10])
    if minCapsuleDist < 5:
        score += 500/minCapsuleDist

    return score + 100/minFoodDist - len(foodDistance)* 150 -len(capsuleDistance)*100

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # add alpha, beta values set to -/+ infinity compare with max/min score
        # alpha = maximizer's known best choice, beta = minimizer's known best choice
        # if alpha >= beta, the agent can assume a better choice was made
        # because at maximizer's turn, finding a value greater than alpha updates alpha and lesser updates beta
        # this works similarly for minimizer.

        def evaluationFunction(gameState, action):

            currentPos = currentGameState.getPacmanPosition()
            currentFood = currentGameState.getFood()  # food available from current state
            currentCapsules = currentGameState.getCapsules()  # power pellets/capsules available from current state
            newCapsules = currentGameState.getCapsules()  # capsules available from successor (excludes capsules@successor)
            newGhostStates = currentGameState.getGhostStates()

            for ghost in newGhostStates:
                ghostDistance = min(mazeDistance(currentPos, ghost.getPosition(), currentGameState))

                if ghostDistance <= 2:
                    if ghost.scaredTimer == 0:
                        score -= 2000 / (ghostDistance + 1)
                    elif ghost.scaredTimer != 0:
                        score += 1000 / (ghostDistance + 1)

                # find min distance to food and add weight based on distance
            for food in currentFood.asList():
                foodDistance = min(mazeDistance(currentPos, food, currentGameState))
                score -= foodDistance*100- len(foodDistance) * 100

            for capsule in newCapsules:
                capsuleDistance = min(mazeDistance(currentPos, capsule, currentGameState))
                if capsule == currentPos:
                    score += 500

            if capsuleDistance < 5:
                score += 100 / minCapsuleDist

            score = score  - len(capsuleDistance) * 100

            return score



        def maxValue(gameState, depth, alpha, beta):

            value = -float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(0)
            for moves in nextMove:
                value = max(value, minValue(gameState.generateSuccessor(0, moves), depth, 1, alpha, beta))
                alpha = max(alpha, value)
                if value > beta:
                    return value

            return value

        def minValue(gameState, depth, agent, alpha, beta):

            value = float("inf")
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            nextMove = gameState.getLegalActions(agent)
            # there can be more than 1 ghost. if then, pass the value to the next ghost
            if agent == gameState.getNumAgents() - 1:
                for moves in nextMove:
                    value = min(value, maxValue(gameState.generateSuccessor(agent, moves), depth - 1, alpha, beta))
                    beta = min(beta, value)
                    if value < alpha:
                        return value
            else:
                for moves in nextMove:
                    value = min(value,
                                minValue(gameState.generateSuccessor(agent, moves), depth, agent + 1, alpha, beta))
                    beta = min(beta, value)
                    if value < alpha:
                        return value

            return value

        alpha = -float("inf")
        beta = float("inf")
        value = -float("inf")

        action = Directions.STOP
        nextMove = gameState.getLegalActions(0)

        for moves in nextMove:
            value2 = value
            value = max(value2, minValue(gameState.generateSuccessor(0, moves), self.depth, 1, alpha, beta))
            if value > value2:
                action = moves

            if value >= beta:
                return action

            alpha = max(alpha, value)

        return action