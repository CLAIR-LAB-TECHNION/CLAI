
from collections import OrderedDict
import cv2
import numpy as np
import sys
from tqdm.auto import tqdm

from cs236203.infrastructure import pytorch_util as ptu


def sample_trajectory(env, policy, max_path_length, render=False):
    """Sample a rollout in the environment from a policy."""
    
    # initialize env for the beginning of a new rollout
    ob =  env.reset() 

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, 'sim'):
                img = env.sim.render(camera_name='track', height=500, width=500)[::-1]
            else:
                img = env.render(mode='single_rgb_array')
            image_obs.append(cv2.resize(img, dsize=(250, 250), interpolation=cv2.INTER_CUBIC))
    
        ac = policy.get_action(ob) # this is a numpy array
        if ac.ndim > 1:
            ac = ac[0]

        
        next_ob, rew, done, _ = env.step(ac)
        
        steps += 1
        rollout_done = (done or steps > max_path_length)
        
        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """Collect rollouts until we have collected min_timesteps_per_batch steps."""

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        #collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

        #count steps
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """Collect ntraj rollouts."""

    paths = []
    for i in range(ntraj):
        # collect rollout
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
    return paths


########################################
########################################


def convert_listofrollouts(paths, concat_rew=True):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    return observations, actions, rewards, next_observations, terminals


########################################
########################################
            

def compute_metrics(paths, eval_paths):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [path["reward"].sum() for path in paths]
    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    # episode lengths, for logging
    train_ep_lens = [len(path["reward"]) for path in paths]
    eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    # logs["Eval_MaxReturn"] = np.max(eval_returns)
    # logs["Eval_MinReturn"] = np.min(eval_returns)
    # logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    # logs["Train_MaxReturn"] = np.max(train_returns)
    # logs["Train_MinReturn"] = np.min(train_returns)
    # logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


def evaluate_policy(env, policy, num_episodes=100, max_actions_per_episode=25, seed=None):
    """
    evaluates a given policy on an initialized taxi environment.
    :param env: a gym taxi environment (v3)
    :param policy: a function that, given a taxi environment state, returns a valid action.
    :param num_episodes: The number of episodes to run during evaluation
    :param max_actions_per_episode: The number of time steps before the episode is ended environment is reset.
    :param seed: a random seed for the environment to enable reproducible results.
    :return: a tuple (total_reward, mean_reward), where `total_reward` is the sum of all rewards achieved in all
             episodes and `mean_reward` is the mean reward per episode.
    """
    # set random seed if given
    if seed is not None:
        env.seed(seed)

    # iterate episodes and accumulate rewards
    all_episode_rewards = 0
    for _ in tqdm(range(num_episodes)):

        # reset env and get initial observation
        obs = env.reset()

        # iterate time steps and accumulate episode rewards
        total_rewards = 0
        for _ in range(max_actions_per_episode):

            # get policy action
            action = policy(obs)

            # perform policy step and accumulate rewards
            obs, reward, done, _ = env.step(action)
            total_rewards += reward

            if done:
                # if task completed, end episode early.
                break

        # accumulate rewards for all episodes
        all_episode_rewards += total_rewards

    # flush excess tqdm output
    sys.stderr.flush()

    return all_episode_rewards, all_episode_rewards / num_episodes
