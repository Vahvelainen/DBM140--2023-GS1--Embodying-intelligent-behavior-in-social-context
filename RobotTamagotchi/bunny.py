'''
  This would propably configure things like:
  - Actions
  - Initial policy
  - ???
'''

from SARSA import SARSA, State, StateVar, Action
import numpy as np

# Define Actions
# Would be great if these are designed further
actions = [
  Action('Happy', 'Bunny smiles and moves around slightly with ears up'),
  Action('Whining', "Bunny makes squeling noises, doesn't move and ears are pointing down"),
  Action('Exited', "Bunny wiggles around with enthusiasim"),
  Action('Sleep', "Bunny is sleeping. Dont disturb it"),
]

# State space variables
# Should be propably between 10-50 states in total
task_status = StateVar( 'Task', ['Undone', 'Done'], 0)
engagement_lvl = StateVar( 'Engagement', ['Lo', 'Hi'], 0)
valence_lvl = StateVar( 'Valence', ['Lo', 'Mid', 'Hi'], 2)

# Create the state
s = State([ task_status, engagement_lvl, valence_lvl ])

#Create policy with initial values
initial_policy = np.matrix([
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

#Create Sarsa with state, list of actions and initial policy
Bunny = SARSA(s, actions, initial_policy, epsilon=0.4, alpha=0.85, gamma=0.95)

def rewardFunction(currState: State, previousState: State):
  if (currState.getVar('Task').value == 'Done' and
      previousState.getVar('Task').value != 'Done'):
   return 5
  
  if (currState.getVar('Engagement').value == 'Hi' and
      previousState.getVar('Engagement').value != 'Hi' and
      currState.getVar('Task').value != 'Done'):
   return 1

  if (currState.getVar('Engagement').value != 'Hi' and
      previousState.getVar('Engagement').value == 'Hi' and
      currState.getVar('Task').value != 'Done'):
   return -1
  
  # 0 if nothing
  return 0


