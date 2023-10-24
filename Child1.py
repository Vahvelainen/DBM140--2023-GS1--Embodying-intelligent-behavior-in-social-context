import numpy as np

'''
  Extension to template for user agents with different behaviors
  Should react to different bunny actions + have some varying behaviour logic
  Outputs valence and engagement values
'''

# Import state space variables
from Child_template import Child
from SARSA import Action, State, StateVar #Import classes for types

class Child1(Child):
  '''
  Child with low attention span
  '''

  def react(self, action: Action, state: State) -> str:
    '''
    React to the action chosen by SARSA by altering state
    Actions and states needs to be matched to the ones from Bunny.py
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

