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
    epsilon=0.2
    alpha=0.4
    gamma=0.2

    # Define Actions
    actions = [
      Action('Needy', 'Ears to the back, eyes down, tail low, making sad noises'),
      Action('Demanding', 'Ears to the middle, eyes down, tail low, nagging noises'),
      Action('Content', 'all body parts in neutral position (ears side, eyes open)'),
      Action('Curious', 'body part neutral + looking at you'),
      Action('Focused', 'Body part neutral + looking at you + waggling (no sound)'),
      Action('Anticipating', 'Body part neutral + looking at you + waggling + making sound'),
      Action('Happy', 'tail waggling slowly, not making eye contact, ears side'),
      Action('Grateful', 'tail waggling slowly, making eye contact, ears side to front movement'),
      Action('Joyful', 'tail waggling faster, making eye contact, ears side to front movement, happy noises'),
      Action('Excited', 'tail waggling heavily, making eye contact, ears to front, happy noises, waggling around'),
    ]

    # State space variables
    valence_lvl = StateVar( 'Valence', ['Neg', 'Neutral', 'Pos'], 0)
    engagement_lvl = StateVar( 'Engagement', ['Unaware', 'Aware', 'Engaged'], 0)
    task_status = StateVar( 'Task', ['Undone', 'Done'], 0)

    # Create the state
    state = State([  valence_lvl, engagement_lvl, task_status ])
    
    # Create policy with initial values
    # This could be better if it would just be selecting action per state, idk why matrix was a good idea
    initialPolicy = np.matrix([
      #Simplified initial policy:
      # Needy if task undone and child unaware
      # Curious if task undone and child aware
      # Anticipating if task undone and chilg engaging
      # Gratefull if task done

      #Task undone
      #Needy, Deman, Cntnt, Cur, Foc, Antp, Hap, Grat, Joy, Exct
      [  1.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Neg, Engagement: Unaware, Task: Undone
      [  1.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Neutral, Engagement: Unaware, Task: Undone
      [  1.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Pos, Engagement: Unaware, Task: Undone
      [  0.0,   0.0,   0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Neg, Engagement: Aware, Task: Undone
      [  0.0,   0.0,   0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Neutral, Engagement: Aware, Task: Undone
      [  0.0,   0.0,   0.0, 1.0, 0.0,  0.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Pos, Engagement: Aware, Task: Undone
      [  0.0,   0.0,   0.0, 0.0, 0.0,  1.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Neg, Engagement: Engaged, Task: Undone
      [  0.0,   0.0,   0.0, 0.0, 0.0,  1.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Neutral, Engagement: Engaged, Task: Undone
      [  0.0,   0.0,   0.0, 0.0, 0.0,  1.0, 0.0,  0.0, 0.0, 0.0 ], # Valence: Pos, Engagement: Engaged, Task: Undone
      #Task done
      #Needy, Deman, Cntnt, Cur, Foc, Antp, Hap, Grat, Joy, Exct
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Neg, Engagement: Unaware, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Neutral, Engagement: Unaware, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Pos, Engagement: Unaware, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Neg, Engagement: Aware, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Neutral, Engagement: Aware, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Pos, Engagement: Aware, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Neg, Engagement: Engaged, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Neutral, Engagement: Engaged, Task: Done
      [  0.0,   0.0,   0.0, 0.0, 0.0,  0.0, 0.0,  1.0, 0.0, 0.0 ], # Valence: Pos, Engagement: Engaged, Task: Done
    ])

    #Call the original init function
    super().__init__(state, actions, initialPolicy, epsilon=epsilon, alpha=alpha, gamma=gamma)

  def rewardFunction(self):
    '''
    Reward function gives rewards based on changes in states
    '''
    # In this case reward is only based on state variables but in reality they don't need to be connected at all

    # Its not ideal that the reward functions doesnt feed to the update function regardless being in same class
    # Buuut its acceptalbe for narrative purposes in the main loop
    # Reward function is part of the bunny bc it defines the "implementation" as a whole and its convinient.

    curr_task = self.state.getVar('Task')
    prev_task = self.prevState.getVar('Task')

    curr_engagement = self.state.getVar('Engagement')
    prev_engagement = self.prevState.getVar('Engagement')

    curr_valence = self.state.getVar('Valence')
    prev_valence = self.prevState.getVar('Valence')

    reward = 0

    # Big reward if task changed to Done
    if (curr_task.value == 'Done' and
        prev_task.value != 'Done'):
      reward += 5
    
    if curr_task.value == 'Undone':
      #Reward based on engagement change (higher the better)
      engagementDiff = curr_engagement.getIndex() - prev_engagement.getIndex()
      reward += engagementDiff * 1.5 #Could be used with multiplier if wanted to

    valenceDiff = curr_valence.getIndex() - prev_valence.getIndex()
    reward += valenceDiff #Could be used with multiplier if wanted to


    # 0 if nothing
    return reward


