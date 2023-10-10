
'''
  Template for user ugents with different behaviors
  Should react to different bunny actions + have some varying behaviour logic
  Outputs valence and engagement values
'''

# Import state space variables
from SARSA import Action, State, StateVar #Import classes of types

class Agent:
  def __init__(self) -> None:
    pass

  def react(self, action: Action, state: State) -> str:
    '''
    React to the action chosen by sarsa
    Actions and states needs to be mathed to the ones from Bunny.py
    '''
    engagement = state.getVar('Engagement')
    task = state.getVar('Task')

    if ( action == 'Sleep' ):
      engagement.set('Lo')
      return 'Child lost interest'
    
    if ( action == 'Whining' ):
      engagement.set('Hi')
      return 'Child is now paying attention'

    if ( action == 'Exited' and engagement.value == 'Hi' ):
      if (task.value != 'Done'):
        task.set('Done')
        self.task_done_rounds = 0
        return 'Child feeds the bunny, task is done!!!'
      else:
        return 'Child feeds the bunny but its not hungry :('
      
    return 'Nothing happens'


