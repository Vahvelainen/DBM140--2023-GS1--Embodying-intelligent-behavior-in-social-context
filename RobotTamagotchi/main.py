
from bunny import Bunny, initial_policy, rewardFunction

from agent_template import Agent

sarsa = Bunny
agent = Agent()

steps_N = 100

initial_policy = initial_policy.copy()

reward = 0
prevState = sarsa.state.copy()
for step_index in range(steps_N):
  print('\nSTEP ' + str(step_index))

  #calculate changes to Q table for last action taken and return new action
  action = sarsa.update(reward)
  print( action.description )

  #use action selected in last update
  result = agent.react(action, sarsa.state)
  print( result )

  #wait for the effect
  print( sarsa.state )

  #calculate reward, use whatever variables needed
  currState = sarsa.state.copy()
  reward = rewardFunction(currState, prevState)
  prevState = currState
  print('Reward: ' + str(reward))

print('\nInitial Policy:\n')
print(initial_policy)

print('\nFinal Policy:\n')
print(sarsa)