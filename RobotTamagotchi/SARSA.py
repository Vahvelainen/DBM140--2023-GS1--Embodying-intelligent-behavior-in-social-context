
import numpy as np

'''
Classes fo running SARSA algorithm including following classes:
- SARSA
- Policy
- Action
- State
- StateVar (for State Variable)
- StateVarOption (Which is not in use and very useless)

Additionally includes function createStateSpace() that return every
possible state. However, State.getStateSpace() can be used instead

INTENDED USE:

  #Set intitial reward and loop
  reward = 0
  while running:
  
    #calculate changes to Q table for last action taken and return new action
    action = sarsa.update(reward)

    #use action selected in last update
    doAction( action )

    #wait for the effect
    sleep(10s)

    #read the sensors etc. and update state space variables
    #state is automatically updates when variables change
    updateState()

    #calculate reward, use whatever variables needed
    reward = countReward()
'''


class Action:
  '''
  Class for actions for SARSA
  Name is used to differentiate and description is used for narrative purposes if needed
  '''
  def __init__(self, name: str, description: str) -> None:
    self.name = name
    self.description = description

  def __hash__(self) -> int:
    return hash(self.name)
  
  def __eq__(self, other: object) -> bool:
    return str(self) == str(other)

class State:
  '''
  Class for states for SARSA
  '''
  def __init__(self, variables: list) -> None:
    self.variables = variables

  def __hash__(self) -> int:
    return hash( self.__str__ )

  def __str__(self) -> str:
    # Convert each element to a string using list comprehension
    string_list = [str(variable) for variable in self.variables]
    return '(' + ', '.join(string_list) + ')'
  
  def __eq__(self, other: object) -> bool:
    return str(self) == str(other)
  
  def getStateSpace(self) -> list:
      '''
      Creates a state space from all the possible variations of the variables in State
      Return
      '''
      return createStateSpace(self.variables)
  
  def copy(self):
    return State( list( var.copy() for var in self.variables ) )

    
class Policy:
  '''
  Representation of the Q-table as a policy
  very hard to decide how to implement this
  Needs:
  - states (list, can generate from a state?)
  - actions (list)
  - Q-matrix np.matrix of [state][action]
  '''
  # Maybe dictionary is better than matrix or idk
  # Matrix works well anough
  def __init__(self, stateSpace: list[ State ], actions: list[ Action ], initialPolicy: np.matrix ) -> None:
    self.actions = actions
    self.stateSpace = stateSpace
    self.Q = initialPolicy

  def bestAction(self, state: State) -> Action:
    '''
    Select the highest scoring action for state
    '''
    stateIndex = self.stateSpace.index(state)
    return self.actions[ np.argmax( self.Q[stateIndex] ) ]

  def __str__(self) -> str:
    return str(self.Q)
  
class SARSA:
  '''
  Class for running the SARSA algorithm
  use .update() to adjust Q table and select next action
  '''
  def __init__(self, state: State, initialPolicy: Policy, epsilon = 0.9, alpha = 0.85, gamma = 0.95) -> None:
    self.policy = initialPolicy
    self.actions = initialPolicy.actions
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

    self.state = state #Current state
    self.nextAction = None 
    self.prevState = None #Previous state (before action)
    self.prevAction = None #Previous previousAction

  def chooseAction(self) -> Action:
    '''
    Epsilon greedy choosing of action
    Should not be used outside update function
    '''
    # Exploration vs exploitation
    if np.random.uniform(0, 1) < self.epsilon:
      # Explore: select a random action
      action = np.random.choice( self.actions )
    else:
      # Exploit: select the action with the highest Q-value
      action = self.policy.bestAction(self.state)
    return action


  def update(self, reward: float) -> Action:
    '''
    Updates the Q values based on reward
    Return action to be used next
    Assumes:
    1.) Last returned action is happened and observed before running
    2.) State is updated
    3.) Reward is somewhat related to the action
    '''
    self.prevAction = self.nextAction
    self.nextAction = self.chooseAction()

    #Skips the first run, but should it?
    if self.prevAction and self.prevState:
      Q = self.policy.Q
      s = self.policy.stateSpace.index(self.prevState)
      a = self.actions.index(self.prevAction)
      s2 = self.policy.stateSpace.index(self.state)
      a2 = self.actions.index(self.nextAction)
      #Update Q values
      predict = Q[s, a]
      target = reward + self.gamma * Q[s2, a2]
      Q[s, a] = Q[s, a] + self.alpha * (target - predict)

    self.prevState = self.state.copy()
    return self.nextAction

  def __str__(self) -> str:
    return str( self.policy) 

class StateVar:
  '''
  Class for state variables for States in SARSA
  '''
  def __init__(self, name: str, options: list, currentIndex = 0) -> None:
    self.name = name
    self.options = options
    self.current = options[ currentIndex ]

  def setIndex(self, index) -> bool:
    '''
    Returns false if index is out of bounds
    '''
    if not index in range( len( self.options ) ):
      return False
    self.current = self.options[ index ]
    return True

  def getIndex(self) -> int:
    '''
    Returns index of current option
    '''
    return self.options.index(self.current)
  
  def copy(self):
    '''
    Returns new instance of it self
    '''
    return StateVar( self.name, self.options, self.getIndex() )

  def __hash__(self) -> int:
    #Not sure if this is a.) necessary b.) working with only the name
    return hash(self.name)
  
  def __str__(self) -> str:
    #Might add bracets later
    return self.name + ': ' +  self.current

  
class StateVarOption:
  '''
  Basically a string,
  Not in use currently
  '''
  def __init__(self, name) -> None:
    self.name = name

  def __hash__(self) -> int:
    return hash(self.name)
  
  def __str__(self) -> str:
    return self.name + ':' 
  

def createStateSpace( stateVariables: list[ StateVar ] ) -> list[ State ]:
  '''
  Creates a state space from all the possible variations of the variable
  Does not alter the original
  1. param is a lsit of State variables
  '''
  vars = [] #Copied list to not alter the original
  stateCount = 1
  for var in stateVariables:
    stateCount = stateCount * len(var.options)
    vars.append( StateVar(var.name, var.options, 0) )    

  states = []
  for _ in range(stateCount):
    # Copy states to a state 
    state = State( list( var.copy() for var in vars ) )
    states.append(state)
    # Alter variables
    rollOver = True
    for var in vars:
      if rollOver:
        rollOver = False
        inRange = var.setIndex( var.getIndex() + 1 )
        if not inRange:
          var.setIndex(0)
          rollOver = True

  return states



