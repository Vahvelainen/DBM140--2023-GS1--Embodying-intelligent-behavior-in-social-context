'''
  Configuration of the variables of sarsa.
  Presented as extension to the sarsa class

  Adds:
  - init function with parameters defined
  - .reward() function
'''

from SARSA import SARSA, State, StateVar, Action
import numpy as np

class Bunny(SARSA):
  def __init__(self) -> None:
    '''
    Init the SARSA algorithm with predefined parameters
    '''

    # Learning parameters
    epsilon=0.4
    alpha=0.85
    gamma=0.95

    # Define Actions
    actions = [
      Action('Happy', 'Bunny smiles and moves around slightly with ears up'),
      Action('Whining', "Bunny makes squeling noises, doesn't move and ears are pointing down"),
      Action('Exited', "Bunny wiggles around with enthusiasim"),
      Action('Sleep', "Bunny is sleeping. Dont disturb it"),
    ]

    # State space variables
    task_status = StateVar( 'Task', ['Undone', 'Done'], 0)
    engagement_lvl = StateVar( 'Engagement', ['Lo', 'Hi'], 0)
    valence_lvl = StateVar( 'Valence', ['Lo', 'Mid', 'Hi'], 2)

    # Create the state
    state = State([ task_status, engagement_lvl, valence_lvl ])
    
    #Create policy with initial values
    initialPolicy = np.matrix([
      # Happy, Whining, Exited, Sleep
      [ 0.0,   1.0,     0.0,    0.0 ], # Task: Undone, Engagement: Lo, Valence: Lo
      [ 0.0,   0.0,     0.0,    1.0 ], # Task: Done, Engagement: Lo, Valence: Lo
      [ 0.0,   0.0,     1.0,    0.0 ], # Task: Undone, Engagement: Hi, Valence: Lo
      [ 1.0,   0.0,     0.0,    0.0 ], # Task: Done, Engagement: Hi, Valence: Lo
      [ 0.0,   1.0,     0.0,    0.0 ], # Task: Undone, Engagement: Lo, Valence: Mid
      [ 0.0,   0.0,     0.0,    1.0 ], # Task: Done, Engagement: Lo, Valence: Mid
      [ 0.0,   0.0,     0.0,    0.0 ], # Task: Undone, Engagement: Hi, Valence: Mid
      [ 0.0,   0.0,     0.0,    1.0 ], # Task: Done, Engagement: Hi, Valence: Mid
      [ 0.0,   1.0,     0.0,    0.0 ], # Task: Undone, Engagement: Lo, Valence: Hi
      [ 0.0,   0.0,     0.0,    1.0 ], # Task: Done, Engagement: Lo, Valence: Hi
      [ 1.0,   0.0,     0.0,    0.0 ], # Task: Undone, Engagement: Hi, Valence: Hi
      [ 0.0,   0.0,     1.0,    0.0 ], # Task: Done, Engagement: Hi, Valence: Hi
    ])

    #Call the original init function
    super().__init__(state, actions, initialPolicy, epsilon=epsilon, alpha=alpha, gamma=gamma)

  def rewardFunction(self):
    '''
    Reward function gives rewards based on changes in states
    '''
    # Its not ideal that the reward functions doesnt feed to the update function atm. 
    # Buuut its acceptalbe for narrative purposes in the main loop

    curr_task = self.state.getVar('Task')
    prev_task = self.prevState.getVar('Task')

    curr_engagement = self.state.getVar('Engagement')
    prev_engagement = self.prevState.getVar('Engagement')

    if (curr_task.value == 'Done' and
        prev_task.value != 'Done'):
      return 5
    
    if (curr_engagement.value == 'Hi' and
        prev_engagement.value != 'Hi' and
        curr_task.value != 'Done'):
      return 1

    if (curr_engagement.value != 'Hi' and
        prev_engagement.value == 'Hi' and
        curr_task.value != 'Done'):
      return -1
    
    # 0 if nothing
    return 0


