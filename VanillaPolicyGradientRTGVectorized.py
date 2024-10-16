import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from tinygrad import Tensor, nn, TinyJit
from typing import List




class PolicyNet:
    def __init__(self, obs_size, action_size, hidden_sizes):
        self.input_transform = nn.Linear(obs_size, hidden_sizes[0])
        self.hiddens = [nn.Linear(hidden_sizes[i-1], hidden_sizes[i]) for i in range(1,len(hidden_sizes))]
        self.output_transform = nn.Linear(hidden_sizes[-1], action_size)

    def __call__(self, state: Tensor) -> Tensor:
        x = self.input_transform(state).relu()
        for hidden in self.hiddens:
            x = hidden(x).relu()
        return self.output_transform(x).log_softmax()
    
env = gym.make_vec("LunarLander-v3", num_envs=8, vectorization_mode="sync")

model = PolicyNet(env.single_observation_space.shape[0], int(env.single_action_space.n), [24])

opt = nn.optim.Adam(nn.state.get_parameters(model))

