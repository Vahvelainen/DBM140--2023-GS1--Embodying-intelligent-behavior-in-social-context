
from bunny import Bunny

from agent_template import Agent

sarsa = Bunny()
agent = Agent()

steps_N = 100

initial_policy = sarsa.Q.copy()

reward = 0
for step_index in range(steps_N):
  #Print the step in green
  print('\033[92m' + '\nSTEP ' + str(step_index) + '\033[0m')

  #Set Task Undone avery 10th step
  if ( step_index%10 == 0):
    task = sarsa.state.getVar('Task')
    #Print the new task and previous task status in yellow
    print('\033[93m' + 'It is time for a new task. Previous task: ' + task.value  + '\033[0m')
    task.set('Undone')

  #calculate changes to Q table for last action taken and return new action
  action = sarsa.update(reward)
  print( action.description )

  #use action selected in last update
  result = agent.react(action, sarsa.state)
  print( result )

  #wait for the effect
  print( sarsa.state )

  #calculate reward
  reward = sarsa.rewardFunction()
  print('Reward: ' + str(reward))

print('\nInitial Policy:\n')
print(initial_policy)

print('\nFinal Policy:\n')
print(sarsa)