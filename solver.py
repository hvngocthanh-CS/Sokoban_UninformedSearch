import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False


def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    
    # get initial positions of player and boxes
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    # create initial state
    startState = (beginPlayer, beginBox) # store position boxes and player positions
    frontier = collections.deque([[startState]]) # store lists of states
    exploredSet = set() # Create a set to store the explored states
    actions = [[0]] # Create a list to store the actions (path), with the initial state being 0 (first node)
    temp = [] # store the final result to be returned
    
    #DFS algorithm: 
    while frontier: # While there are nodes to explore
        node = frontier.pop() # Get the last node in the frontier queue and remove it
        node_action = actions.pop() # Get the last action (path) in the actions list and remove it
        if isEndState(node[-1][-1]): # If the current state is the goal state, append the actions to the temp list and break out of the loop
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet: # check if the current state has not been explored before
            exploredSet.add(node[-1]) # Add the current state to the explored set
            
            # For each legal action that the player can take in the current state
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # update positions of player and boxes using the action
                # check if the resulting box positions lead to a failed state
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp # Return the actions lead to the goal state.


def breadthFirstSearch(gameState):
    """The BFS algorithm is the same as DFS but the only difference is how to open the next node.
    BFS is opening a node from left to right, opening a shallow node, not a deep node."""
    
    beginBox = PosOfBoxes(gameState) # get initial positions of boxes
    beginPlayer = PosOfPlayer(gameState) # get initial positions of player

    startState = (beginPlayer, beginBox) # Create initial state to store positions of player and box
    frontier = collections.deque([[startState]]) # Create a frontier queue, with the initial point being the state of the game.
    actions = collections.deque([[0]]) # store actions
    exploredSet = set() # Create a set to store the explored states
    temp = [] # create an empty list to store the resulting actions
    
    #BFS algorithm:
    while frontier: # While there are nodes to explore
        node = frontier.popleft() # Get the leftside state in the frontier and remove it
        node_action = actions.popleft() # Get the leftside action (path) in the actions list and remove it
        if isEndState(node[-1][-1]): # If the current node is the goal state
            temp += node_action[1:] # Append all actions into temp list
            break # break out of the loop
        if node[-1] not in exploredSet: # check if the current node has not been visited.
            exploredSet.add(node[-1]) # Add the current node to the set of visited nodes.
            for action in legalActions(node[-1][0], node[-1][1]): # Loop through the set of legal action, for example (UP, DOWN, LEFT, RIGHT)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # update positions of player and boxes using the action
                if isFailed(newPosBox): # if the resulting box positions lead to a failed state
                    continue # Skip and go to the next action
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Add new position to frontier queue
                actions.append(node_action + [action[-1]]) # Add new action to the actions
    return temp # Return the actions lead to the goal state

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()]) # cost of actions when not pushing box

def uniformCostSearch(gameState):
    """The UCS algorithm finds the optimal path based on the minimum cost"""
    beginBox = PosOfBoxes(gameState) # get initial positions of boxes
    beginPlayer = PosOfPlayer(gameState) # get initial positions of player

    startState = (beginPlayer, beginBox) # Create initial state to store positions of player and box
    frontier = PriorityQueue() # create a priority queue for the frontier
    frontier.push([startState], 0) # the initial point being the state of the game
    exploredSet = set() # create a set to store the explored states
    actions = PriorityQueue() # create a priority queue for the actions 
    actions.push([0], 0) # add the initial action to the actions queue with priority 0 (the cost of first node is 0)
    temp = [] # create an empty list to store the resulting actions
    
    #UCS algorithm:
    while not frontier.isEmpty(): # While there are nodes to explore
        node = frontier.pop() # Pop the node from the frontier with the lowest priority (last position) and remove it
        node_action = actions.pop() # Pop the last action in actions queue and remove it
        if isEndState(node[-1][-1]):  # If the current node is the goal state
            temp += node_action[1:] # Append all actions into temp list, except the first (0).
            break # break out of the loop
        if node[-1] not in exploredSet: # If the current node has not been visited
            exploredSet.add(node[-1]) # Add the current node to the set of visited nodes.
            Cost_of_Action = cost(node_action[1:]) # Calculate the cost of actions (the initial position to the current node) when not pushing
            for action in legalActions(node[-1][0], node[-1][1]): # Loop through the set of legal action (UP, DOWN, LEFT, RIGHT)
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Update the new player and box positions.
                if isFailed(newPosBox): # If the new box position is a fail state.
                    continue # Skip and go to the next action
                frontier.push(node + [(newPosPlayer, newPosBox)], Cost_of_Action) # Add the new node with the new player and box positions to the frontier with the priority of the cost.
                actions.push(node_action + [action[-1]], Cost_of_Action) # Add the new node to the actions with the priority of the cost.
    return temp # Return the actions lead to the goal state."""
    

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    print('Number of steps:', len(result))
    return result
