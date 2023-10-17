import numpy as np

'''
  Template for user agents with different behaviors
  Should react to different bunny actions + have some varying behaviour logic
  Outputs valence and engagement values

  In the future could be extended to give values to reward function outside the state
'''

# Import state space variables
from SARSA import Action, State, StateVar #Import classes for types

class Child1:
  def __init__(self) -> None:
    #Init function can be used eg. to store previous states and actions
    self.prevState = None
    pass

  def parseReactionDescription(self, beginning, prevState: State, state: State):

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

    if len(responses):
      return beginning + ' and is now ' + ', and '.join(responses)
    else:
      return beginning

  def react(self, action: Action, state: State) -> str:
    '''
    React to the action chosen by SARSA by altering state
    Actions and states needs to be mathed to the ones from Bunny.py
    '''
    engagement = state.getVar('Engagement')
    task = state.getVar('Task')
    valence = state.getVar('Valence')

    taskPropability = 0

    startingState = state.copy()
    childDoing = 'minding their own business'

    ## Getting attention: Engagement unaware, Task undone 
    if ( engagement.value == 'Unaware' and task.value == 'Undone' ):
      childDoing = 'not paying attention'
      if action == 'Needy':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Demanding':
        engagement.set('Aware')
        valence.set('Pos')

      if action == 'Content':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Curious':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Focused':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Anticipating':
        engagement.set('Aware')
        valence.set('Neg')

      if action == 'Happy':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Grateful':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Joyful':
        engagement.set('Aware')
        valence.set('Neutral')

      if action == 'Excited':
        engagement.set('Aware')
        valence.set('Neg')


    ## Keeping Attention: Engagement aware or engaged, Valence Neutral or Positive, Task undone
    elif ( ( not engagement.value == 'Unaware') and ( not valence.value == 'Neg' ) and task.value == 'Undone' ):
      childDoing = 'interacting with robot'
      if action == 'Needy':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Demanding':
        engagement.set('Aware')
        valence.set('Neg')

      if action == 'Content':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Curious':
        engagement.set('Aware')
        valence.set('Neutral')

      if action == 'Focused':
        engagement.set('Aware')
        valence.set('Pos')
        taskPropability = 0.2

      if action == 'Anticipating':
        engagement.set('Engaged')
        valence.set('Pos')
        taskPropability = 0.6

      if action == 'Happy':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Grateful':
        engagement.set('Unaware')
        valence.set('Neutral')

      if action == 'Joyful':
        engagement.set('Aware')
        valence.set('Pos')
        taskPropability = 0.2

      if action == 'Excited':
        engagement.set('Engaged')
        valence.set('Neutral')
        taskPropability = 0.4

    ## Improve Mood: Valence Neg, Engagement aware or engaged,
    elif ( valence.value == 'Neg' and ( not engagement.value == 'Unaware') ):
      childDoing = 'in bad mood'
      if action == 'Needy':
        engagement.set('Unaware')
        valence.set('Neg')

      if action == 'Demanding':
        engagement.set('Aware')
        valence.set('Neg')

      if action == 'Content':
        engagement.set('Unaware')
        valence.set('Neg')

      if action == 'Curious':
        engagement.set('Aware')
        valence.set('Neg')

      if action == 'Focused':
        engagement.set('Aware')
        valence.set('Neutral')

      if action == 'Anticipating':
        engagement.set('Aware')
        valence.set('Pos')

      if action == 'Happy':
        engagement.set('Unaware')
        valence.set('Neg')
      
      if action == 'Grateful':
        engagement.set('Unaware')
        valence.set('Neg')

      if action == 'Joyful':
        engagement.set('Engaged')
        valence.set('Pos')
        taskPropability = 0.4
      
      if action == 'Excited':
        engagement.set('Engaged')
        valence.set('Neutral')
        taskPropability = 0.2

    ## Gratitude: Task done, Engagement aware or engaged, Valence Neutral or Positive
    elif ( task.value == 'Done' and ( not engagement.value == 'Unaware' ) and ( not valence.value == 'Neg' ) ):
      childDoing = 'playing with robot after petting'
      if action == 'Needy':
        engagement.set('Unaware')
        valence.set('Neg')
      
      if action == 'Demanding':
        engagement.set('Aware')
        valence.set('Neg')

      if action == 'Content':
        engagement.set('Unaware')
        valence.set('Neutral')
    
      if action == 'Curious':
        engagement.set('Aware')
        valence.set('Neutral')

      if action == 'Focused':
        engagement.set('Aware')
        valence.set('Neutral')

      if action == 'Anticipating':
        engagement.set('Aware')
        valence.set('Pos')

      if action == 'Happy':
        engagement.set('Unaware')
        valence.set('Neutral')
    
      if action == 'Grateful':
        engagement.set('Aware')
        valence.set('Neg')

      if action == 'Joyful':
        engagement.set('Aware')
        valence.set('Neutral')

      if action == 'Excited':
        engagement.set('Aware')
        valence.set('Pos')

    ## Change for children do actually do the task
    rndFloat = np.random.uniform(0, 1)
    if (rndFloat < taskPropability):
      task.set('Done')

    return self.parseReactionDescription( 'Child was ' + childDoing, startingState, state)

