from cs236203.infrastructure.utils import convert_listofrollouts
import numpy as np

class ReplayBuffer(object):
    '''This class is used to store and manage experiences collected from rollouts (sequences of states, actions, rewards, etc.)'''

    def __init__(self, max_size=1000000):

        self.max_size = max_size

        # store each rollout
        self.paths = []

        # store (concatenated) component arrays from each rollout
        self.obs = None
        self.actions = None
        self.rewards = None
        self.next_obs = None
        self.terminals = None

    def __len__(self):
        if self.obs.any():
            return self.obs.shape[0]
        else:
            return 0

    def add_rollouts(self, paths, concat_rew=True):

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays, and append them onto
        # our arrays
        observations, actions, rewards, next_observations, terminals = (
            convert_listofrollouts(paths, concat_rew))

        if self.obs is None: # Initializes with the last element
            self.obs = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_obs = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else: # If the arrays are already initialized
            # Concatenates the new element to the existing ones and truncates to max_size.
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]
            self.actions = np.concatenate([self.actions, actions])[-self.max_size:]
            if concat_rew:
                self.rewards = np.concatenate(
                    [self.rewards, rewards]
                )[-self.max_size:]
            else:
                if isinstance(rewards, list):
                    self.rewards += rewards
                else:
                    self.rewards.append(rewards)
                self.rewards = self.rewards[-self.max_size:]
            self.next_obs = np.concatenate(
                [self.next_obs, next_observations]
            )[-self.max_size:]
            self.terminals = np.concatenate(
                [self.terminals, terminals]
            )[-self.max_size:]

