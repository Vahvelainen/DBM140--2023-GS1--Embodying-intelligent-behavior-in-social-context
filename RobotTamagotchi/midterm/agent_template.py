
'''
  Template for user agents with different behaviors
  Should react to different bunny actions + have some varying behaviour logic
  Outputs valence and engagement values

  In the future could be extended to give values to reward function outside the state
'''

# Import state space variables
from SARSA import Action, State, StateVar #Import classes for types

class Agent:
  def __init__(self) -> None:
    #Init function can be used eg. to store previous states and actions
    pass

  def react(self, action: Action, state: State) -> str:
    '''
    React to the action chosen by SARSA by altering state
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


