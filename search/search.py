#THIS  CODE  WAS MY OWN WORK , IT WAS  WRITTEN  WITHOUT  CONSULTING  ANY
#SOURCES  OUTSIDE  OF  THOSE  APPROVED  BY THE  INSTRUCTOR - MINSEOK CHOI
#MCHOI49@emory.edu/mchoi49

# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    fringe = util.Stack()
    currentState = problem.getStartState()
    # push starting position to fringe. Fringe contains current node/directions
    fringe.push((currentState, []))
    visited = []

    visited.append(currentState)
    # follows generic iterative dfs algorithm and insert successors to fringe to get directions
    while not fringe.isEmpty():
        (node, directions) = fringe.pop()

        if problem.isGoalState(node):
            return directions

        for successor, actions, cost in problem.getSuccessors(node):
            if not successor in visited:
                fringe.push((successor, directions + [actions]))
                visited.append(node)


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    currentState = problem.getStartState()
    # push starting position to fringe. Fringe contains current node/directions
    fringe.push((currentState, []))
    visited = []

    # added starting node to visited nodes list since the code below does not append starting node to visited list
    visited.append(currentState)
    while not fringe.isEmpty():
        (node, directions) = fringe.pop()

        if problem.isGoalState(node):
            return directions

        for successor, actions, cost in problem.getSuccessors(node):
            if not successor in visited:
                fringe.push((successor, directions + [actions]))
                visited.append(successor)

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    fringe = util.PriorityQueue()
    currentState = problem.getStartState()
    # push starting position to fringe. Fringe contains current (node/directions)/cost
    fringe.push((currentState, []), 0)
    # visited is set to dictionary so that it is easier to store cost values
    visited = dict()
    while not fringe.isEmpty():
        (node, directions) = fringe.pop()

        curCost = problem.getCostOfActions(directions)
        visited[node] = curCost

        if problem.isGoalState(node):
            return directions

        for successor, actions, cost in problem.getSuccessors(node):
            # check if lowest cost is less than cost of the already visited successor node
            if (not successor in visited) or (successor in visited and visited[successor] > curCost + cost):
                fringe.push((successor, directions + [actions]), curCost + cost)
                visited[successor] = curCost + cost

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."

    fringe = util.PriorityQueue()
    currentState = problem.getStartState()
    # push starting position to fringe. Fringe contains current node/directions/cost and heuristic
    fringe.push((currentState, [], 0), 0)
    # using dictionary causes some syntex errors.. back to list
    visited = []

    while not fringe.isEmpty():
        (node, directions, curCost) = fringe.pop()
        if not node in visited:
            visited.append(node)
            if problem.isGoalState(node):
                return directions
            # update heuristic cost
            for successor, actions, cost in problem.getSuccessors(node):
                heuristicCost = curCost + heuristic(successor, problem)

                fringe.push((successor, directions + [actions], curCost + cost), heuristicCost + cost)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
