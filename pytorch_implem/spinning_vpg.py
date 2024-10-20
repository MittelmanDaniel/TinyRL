import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym  # Updated to use gymnasium
from gymnasium.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """
    Build a feedforward neural network.
    
    Args:
        sizes (list): Sizes of each layer.
        activation (nn.Module): Activation function for hidden layers.
        output_activation (nn.Module): Activation function for the output layer.
    
    Returns:
        nn.Sequential: The constructed neural network.
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, 
          epochs=50, batch_size=5000, render=False):
    """
    Train a policy network using the policy gradient method.
    
    Args:
        env_name (str): Name of the Gymnasium environment.
        hidden_sizes (list): Sizes of hidden layers in the policy network.
        lr (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Number of timesteps per batch.
        render (bool): Whether to render the environment.
    """
    
    # Create environment and verify spaces
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Initialize policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    def get_policy(obs):
        """
        Compute the action distribution given observations.
        
        Args:
            obs (torch.Tensor): Observations.
        
        Returns:
            Categorical: Categorical distribution over actions.
        """
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        """
        Sample an action from the policy.
        
        Args:
            obs (torch.Tensor): Observations.
        
        Returns:
            int: Selected action.
        """
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, weights):
        """
        Compute the policy gradient loss.
        
        Args:
            obs (torch.Tensor): Observations.
            act (torch.Tensor): Actions taken.
            weights (torch.Tensor): Returns to weight the log probabilities.
        
        Returns:
            torch.Tensor: The computed loss.
        """
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).sum()

    # Initialize optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    def train_one_epoch():
        """
        Collect a batch of experience and perform a policy update.
        
        Returns:
            tuple: Batch loss, list of episode returns, list of episode lengths.
        """
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_rets = []
        batch_lens = []

        # Reset environment
        obs, info = env.reset()
        terminated, truncated = False, False
        ep_rews = []
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())

            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            act = get_action(obs_tensor)
            batch_acts.append(act)

            # Step the environment
            obs, rew, terminated, truncated, info = env.step(act)
            ep_rews.append(rew)

            done = terminated or truncated

            if done:
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # Assign return as weight for all timesteps in the episode
                batch_weights += [ep_ret] * ep_len

                # Reset episode-specific variables
                obs, info = env.reset()
                terminated, truncated = False, False
                ep_rews = []
                finished_rendering_this_epoch = True

                if len(batch_obs) > batch_size:
                    break

        # Convert batch data to tensors
        batch_obs_tensor = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
        batch_acts_tensor = torch.as_tensor(batch_acts, dtype=torch.int64)
        batch_weights_tensor = torch.as_tensor(batch_weights, dtype=torch.float32)

        # Compute loss and perform backpropagation
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=batch_obs_tensor,
                                  act=batch_acts_tensor,
                                  weights=batch_weights_tensor)
        batch_loss.backward()
        optimizer.step()

        return batch_loss.item(), batch_rets, batch_lens

    # Training loop
    for i in range(1, epochs + 1):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print(f'epoch: {i:3d} \t loss: {batch_loss:.3f} \t '
              f'return: {np.mean(batch_rets):.3f} \t '
              f'ep_len: {np.mean(batch_lens):.3f}')

    # Close the environment after training
    env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')  # Updated default
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing the simplest formulation of policy gradient with Gymnasium.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
