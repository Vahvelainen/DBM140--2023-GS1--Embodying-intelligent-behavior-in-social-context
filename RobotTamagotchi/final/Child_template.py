import warnings


'''
  Template for user agents with different behaviors
  Should react to different bunny actions + have some varying behaviour logic
  Outputs valence and engagement values

  In the future could be extended to give values to reward function outside the state
'''

# Import state space variables
from SARSA import Action, State, StateVar #Import classes for types

class Child:
  def __init__(self) -> None:
    #Init function can be used eg. to store previous states and actions
    self.prevState = None
    pass

  def parseReactionDescription(self, beginning, prevState: State, state: State):
    '''
    Method for parsing together a narrative descritption of how the state of the child got changed
    '''

    responses = []
    curr_valence = state.getVar('Valence').getIndex()
    prev_valence = prevState.getVar('Valence').getIndex()
    curr_engagement = state.getVar('Engagement').getIndex()
    prev_engagement = prevState.getVar('Engagement').getIndex()
    curr_task = state.getVar('Task').getIndex()
    prev_task = prevState.getVar('Task').getIndex()

    if curr_valence == 0 and prev_valence != 0:
      responses.append('sad')
    if curr_valence == 1 and prev_valence != 1:
      responses.append('feeling neutral')
    if curr_valence == 2 and prev_valence != 2:
      responses.append('happy')

    if curr_engagement == 0 and prev_engagement != 0:
      responses.append('losing interest')
    if curr_engagement == 1 and prev_engagement != 1:
      responses.append('noticing the robot')
    if curr_engagement == 2 and prev_engagement != 2:
      responses.append('engaging with the robot')

    if curr_task == 1 and prev_task != 1:
      responses.append('petting the robot!')

    response = beginning
    if len(responses):
      response = beginning + ' and is now ' + ', and '.join(responses)

    if curr_task == 0:
      response = response + ' ❌'
    else:
      response = response + ' ✅'

    return response

  def react(self, action: Action, state: State) -> str:
    '''
    React to the action chosen by SARSA by altering state
    Actions and states needs to be mathed to the ones from Bunny.py
    '''
    
    #To know the difference in state for the parser function, a starting state should be saved
    startingState = state.copy()

    #Print warning if behaviour is not defined
    warnings.warn('!!!WARNING!!!: Using template. Child behaviour is not properly defined.')

    return self.parseReactionDescription( 'Child is not reacting due to lack of programming :) ', startingState, state)

