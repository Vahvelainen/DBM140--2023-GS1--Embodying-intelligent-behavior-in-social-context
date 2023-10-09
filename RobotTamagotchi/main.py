
import numpy as np

'''
  Main loop for testing learnign algorithm with different user agents
  Needs import as a class:
    - Robotbunny
    - Agent representing the user
    - Scheduling maybe?

  Currently just testing ground to see that SARSA works
'''

from SARSA import SARSA, State, StateVar, Action

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
valence_lvl = StateVar( 'Valence', ['Lo', 'Mid', 'Hi'], 1)

# Create the state
s = State([ task_status, engagement_lvl, valence_lvl ])

# Variables of the state can be altered freely:
valence_lvl.setIndex(2) # Mid -> Hi

print('Starting state:\n' + str(s))

# Create full state space for policy
stateSpace = s.getStateSpace()
print('\nComplete StateSpace:')
for state in stateSpace:
  print(state)

#Create policy with initial values
initial_policy = np.matrix([
  # Happy, Whining, Exited, Sleep
  [ 0.0,   1.0,     0.5,    0.0 ], # Task: Undone, Engagement: Lo, Valence: Lo
  [ 0.5,   0.0,     0.5,    1.0 ], # Task: Done, Engagement: Lo, Valence: Lo
  [ 0.5,   0.5,     1.0,    0.0 ], # Task: Undone, Engagement: Hi, Valence: Lo
  [ 1.0,   0.0,     0.5,    0.5 ], # Task: Done, Engagement: Hi, Valence: Lo
  [ 0.5,   1.0,     0.5,    0.0 ], # Task: Undone, Engagement: Lo, Valence: Mid
  [ 0.5,   0.0,     0.5,    1.0 ], # Task: Done, Engagement: Lo, Valence: Mid
  [ 0.5,   0.5,     0.5,    0.0 ], # Task: Undone, Engagement: Hi, Valence: Mid
  [ 0.5,   0.0,     0.5,    1.0 ], # Task: Done, Engagement: Hi, Valence: Mid
  [ 0.5,   1.0,     0.5,    0.0 ], # Task: Undone, Engagement: Lo, Valence: Hi
  [ 0.5,   0.0,     0.5,    1.0 ], # Task: Done, Engagement: Lo, Valence: Hi
  [ 1.0,   0.5,     0.5,    0.0 ], # Task: Undone, Engagement: Hi, Valence: Hi
  [ 0.5,   0.0,     1.0,    0.5 ], # Task: Done, Engagement: Hi, Valence: Hi
])

print('\nInitial Policy:\n')
print(initial_policy)

#Create Sarsa with state, list of actions and initial policy
sarsa = SARSA(s, actions, initial_policy, epsilon=0.2, alpha=0.85, gamma=0.95)

#Simulating steps of the loop:
reward = 0

print('\nSTEP 1')
#calculate changes to Q table for last action taken and return new action
action = sarsa.update(reward)
print(action.description)

#use action selected in last update
#wait for the effect
#read the sensors etc. and update state space variables
print('Nothing happens')
print(s)

#calculate reward, use whatever variables needed
reward = 0 #No reward

print('\nSTEP 2')
#calculate changes to Q table for last action taken and return new action
action = sarsa.update(reward)
print(action.description)

#use action selected in last update
#wait for the effect
#read the sensors etc. and update state space variables
engagement_lvl.setIndex(1) # Low -> Hi
print('Child is now engaged')
print(s)

#calculate reward, use whatever variables needed
reward = 0.5 #Engaged

print('\nSTEP 3')
#calculate changes to Q table for last action taken and return new action
action = sarsa.update(reward)
print(action.description)

#use action selected in last update
#wait for the effect
#read the sensors etc. and update state space variables
task_status.setIndex(1) # Undone -> Done
print('Child did the task!!!')
print(s)

#calculate reward, use whatever variables needed
reward = 10 #Engaged

#STEP 4
print('\nSTEP 4')
action = sarsa.update(reward)
print(action.description)

print('\nFinal Policy:\n')
print(sarsa)
