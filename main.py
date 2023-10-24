import numpy as np
'''
Iterate learning algorithm with agent and write the actions to the console
After its done, show changes in policy Q table
'''

from bunny import Bunny
from Child1 import Child1

sarsa = Bunny()
agent = Child1()

initial_policy = sarsa.Q.copy()

def rewardFunction():
   #The reward function for bunny is defined in the same class but its not stricly necessary
  return sarsa.rewardFunction()

#Simulation parameters
steps_N = 40
task_reset_rate = 20

reward = 0
for step_index in range(steps_N):
  #Print the step in green
  print('\033[92m' + '\nSTEP ' + str(step_index) + '\033[0m')

  #Set Task Undone every N:th step
  if ( step_index%task_reset_rate == 0):
    task = sarsa.state.getVar('Task')
    #Print the new task and previous task status in yellow
    print('\033[93m' + 'It is time for a new task. Previous task: ' + task.value  + '\033[0m')
    task.set('Undone')

  #calculate changes to Q table for last action taken and return new action
  action = sarsa.update(reward)
  print( 'Robot is acting ' + action.name + ': ' + action.description )

  #use action selected in last update
  result = agent.react(action, sarsa.state)
  print( result )

  #wait for the effect
  # print( sarsa.state )

  #calculate reward
  reward = sarsa.rewardFunction()
  # print('Reward: ' + str(reward))

print('\nInitial Policy:\n')
print(initial_policy)

print('\nFinal Policy:\n')
rounded_matrix = np.round(sarsa.Q, decimals=2)
print(rounded_matrix)