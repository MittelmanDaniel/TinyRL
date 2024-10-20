import gymnasium as gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from gymnasium.wrappers import RecordVideo

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, hidden_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.network(x)

# Hyperparameters
NUM_SAMPLES = 256    # Number of episodes per epoch
NUM_EPOCHS = 50      # Total training epochs

EVAL_EVERY = 10


ENV_NAME = 'LunarLander-v2'
# Initialize the environment
env = gym.make(ENV_NAME, render_mode = "rgb_array")

eval_env =  RecordVideo(env, video_folder="lander-agent", name_prefix="eval", episode_trigger=lambda x: True)

# Retrieve dimensions
obs_dim = env.observation_space.shape[0]
action_dim = int(env.action_space.n)

# Initialize the policy network
model = PolicyNetwork(obs_dim, hidden_size=32, action_size=action_dim)

# Define helper functions
def get_probs(obs):
    """
    Given an observation tensor, return the Categorical distribution based on probabilities.
    """
    return Categorical(probs=torch.exp(model(obs)))

def get_action(obs):
    """
    Sample an action from the policy network's output distribution.
    """
    return get_probs(obs).sample().item()

def compute_loss(obs: torch.Tensor, act: torch.Tensor, rew: torch.Tensor):
    """
    Compute the policy gradient loss.
    
    Args:
        obs (torch.Tensor): Batch of observations. Shape: [batch_size, obs_dim]
        act (torch.Tensor): Batch of actions taken. Shape: [batch_size]
        rew (torch.Tensor): Batch of rewards (returns). Shape: [batch_size]
    
    Returns:
        torch.Tensor: Computed loss value.
    """
    logp = model(obs)  # Log probabilities: Shape [batch_size, action_dim]
    # Gather log probabilities of the taken actions
    act_logps = torch.gather(logp, 1, act.unsqueeze(1)).squeeze(1)  # Shape [batch_size]
    # Compute the loss (negative for gradient ascent)
    loss = -((act_logps * rew).sum()) / NUM_SAMPLES
    return loss

# Initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr = 1e-2)

def train_one_epoch():
    """
    Collect trajectories and perform a single policy gradient update.
    
    Returns:
        tuple: (loss value, list of episode returns, list of episode lengths)
    """
    batch_obs = []
    batch_acts = []
    batch_rewards = []
    batch_returns = []
    batch_lens = []
    
    for sample in range(NUM_SAMPLES):
        sample_rewards = []
        obs, info = env.reset()
        
        while True:
            # Store a copy of the current observation
            batch_obs.append(obs.copy())
            
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs)
            
            # Select and store action
            act = get_action(obs_tensor.unsqueeze(0))
            batch_acts.append(act)
            
            # Take action in the environment
            obs, rew, terminated, truncated, info = env.step(act)
            sample_rewards.append(rew)
            
            done = terminated or truncated
            
            if done:
                # Compute total return for the episode
                sample_ret = sum(sample_rewards)
                sample_len = len(sample_rewards)
                
                # Store return and episode length
                batch_returns.append(sample_ret)
                batch_lens.append(sample_len)
                
                # Assign return as reward for all timesteps in the episode
                batch_rewards += [sample_ret] * sample_len
                
                break
    
    # Convert lists to tensors
    batch_obs_tensor = torch.as_tensor(np.array(batch_obs))
    batch_act_tensor = torch.as_tensor(batch_acts)
    batch_rew_tensor = torch.as_tensor(batch_rewards)
    
    # Compute loss
    optimizer.zero_grad()
    batch_loss = compute_loss(batch_obs_tensor, batch_act_tensor, batch_rew_tensor)
    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item(), batch_returns, batch_lens

# Training loop
for epoch in range(1,NUM_EPOCHS+1):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    
    print(f'epoch: {epoch:3d} \t loss: {batch_loss:.3f} \t '
          f'return: {np.mean(batch_rets):.3f} \t '
          f'ep_len: {np.mean(batch_lens):.3f}')


    if(epoch%EVAL_EVERY == 0):
        with model.eval():
            eval_reward = 0
            obs, info = eval_env.reset()
            while True:
                obs_tensor = torch.as_tensor(obs)
                act = get_action(obs_tensor.unsqueeze(0))

                obs, rew, terminated, truncated, info = eval_env.step(act)

                eval_reward+=rew

                done = terminated or truncated
                
                if done:
                    break

            print(f'eval: {epoch//EVAL_EVERY} \t eval_reward: {eval_reward:.3f}')


