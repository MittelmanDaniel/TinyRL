import gymnasium as gym
from tinygrad import Tensor, nn, TinyJit, Variable

import numpy as np

class PolicyNet():
    def __init__(self, obs_size, hidden, act_size):
        self.input_transform = nn.Linear(obs_size, hidden)
        self.out_transform = nn.Linear(hidden, act_size)


    def __call__(self, state):
        x = self.input_transform(state).tanh()

        return self.out_transform(x)



# Hyperparameters
NUM_SAMPLES = 256    # Number of episodes per epoch
NUM_EPOCHS = 50      # Total training epochs


# Initialize the environment
env = gym.make('CartPole-v1')

# Retrieve dimensions
obs_dim = env.observation_space.shape[0]
action_dim = int(env.action_space.n)


model = PolicyNet(obs_dim, 32, action_dim)

optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-2)

@TinyJit
def get_probs(obs):
    with Tensor.test():
        return model(obs).softmax(axis = 1)

@TinyJit
def get_action(obs):
    with Tensor.test():
        return model(obs).softmax(axis = 1).multinomial().realize()


def compute_loss(obs: Tensor, act: Tensor, rew: Tensor):
    logp = model(obs).log_softmax(axis = 1)

    act_logps = logp.gather(1, act.unsqueeze(1)).squeeze(1)

    loss = -(act_logps * rew).sum()/NUM_SAMPLES

    return loss




def train_one_epoch():
    batch_obs = []
    batch_acts = []
    batch_rewards = []
    batch_returns = []
    batch_lens = []

    for sample in range(NUM_SAMPLES):
        sample_rewards = []
        obs, info = env.reset()

        while True:
            batch_obs.append(obs.copy())

            obs_tensor = Tensor(obs).unsqueeze(0)

            act = get_action(obs_tensor).item()

            batch_acts.append(act)

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

    with Tensor.train():
        batch_obs_tensor = Tensor(batch_obs)

        batch_act_tensor = Tensor(batch_acts)

        batch_rew_tensor = Tensor(batch_rewards)


        # Compute loss
        optimizer.zero_grad()
        batch_loss = compute_loss(batch_obs_tensor, batch_act_tensor, batch_rew_tensor)
        batch_loss.backward()
        optimizer.step()
        
        return batch_loss.item(), batch_returns, batch_lens

# Training loop
for epoch in range(NUM_EPOCHS):
    get_action.reset()
    get_probs.reset()
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    
    print(f'epoch: {epoch:3d} \t loss: {batch_loss:.3f} \t '
          f'return: {np.mean(batch_rets):.3f} \t '
          f'ep_len: {np.mean(batch_lens):.3f}')


