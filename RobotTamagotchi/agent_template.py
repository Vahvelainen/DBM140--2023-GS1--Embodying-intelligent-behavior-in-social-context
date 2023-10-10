
'''
  Template for user ugents with different behaviors
  Should react to different bunny actions + have some varying behaviour logic
  Outputs valence and engagement values
'''

# Import state space variables
from SARSA import Action, State, StateVar #Import classes fo types

class Agent:
  def __init__(self) -> None:
    self.task_done_rounds = 0

  def updateTask(self, task: StateVar):
    '''
    Reset task after 10 rounds its being done
    Isnt really necessary to be included in the agent
    Maybe should be made to dedicated class? Propably.
    '''
    if (task.value == 'Done'):
      self.task_done_rounds += 1
    if(self.task_done_rounds > 10):
      task.set('Undone')

  def react(self, action: Action, state: State) -> str:
    '''
    React to the action chosen by sarsa
    Actions and states needs to be mathed to the ones from Bunny.py
    '''
    engagement = state.getVar('Engagement')
    task = state.getVar('Task')

    self.updateTask(task)

    if ( action == 'Sleep' ):
      engagement.set('Lo')
      return 'Child lost interest'
    
    if ( action == 'Whining' ):
      engagement.set('Hi')
      return 'Child is now paying attention'

    if ( action == 'Exited' and engagement.value == 'Hi' ):
      if (task != 'Done'):
        task.set('Done')
        self.task_done_rounds = 0
        return 'Child feeds the bunny, task is done!!!'
      else:
        return 'Child feeds the bunny but its not hungry :('
      
    return 'Nothing happens'


