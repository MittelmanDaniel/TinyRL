import gymnasium as gym
from tinygrad import Tensor, nn, TinyJit
from typing import Tuple

NUM_UPDATES = 1000

NUM_SAMPLES = 10

DISCOUNT_FACTOR = 0.99

LEARNING_RATE = 0.01


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
    



# Initialise the environment
eval_env = gym.make("LunarLander-v3", render_mode="human")

env = gym.make("LunarLander-v3")

model = PolicyNet(env.observation_space.shape[0], int(env.action_space.n), [24])

opt = nn.optim.Adam(nn.state.get_parameters(model), lr = LEARNING_RATE)


#Takes in a tensor of rewards
#returns a tensor where index i maps to the discounted reward to go
#@TinyJit
def rewards_to_go(rewards: Tensor) -> Tensor:
    # 1, gamma, gamma **2
    discounts = DISCOUNT_FACTOR ** Tensor.arange(rewards.shape[0])
    # r0, gamma*r1, gamma**2 * r2 
    discounted_rewards = discounts*rewards
    # r0 + gamma r1 + gamma**2 r2 + ... + gamma**n rn,  gamma * r1 + ..., gamma ** 2 r2 + ...
    discount_cum_sum = discounted_rewards.flip(0).cumsum(0).flip(0)

    discount_cum_sum = discount_cum_sum / discounts
    
    return discount_cum_sum

@TinyJit
def get_action(obs: Tensor) -> Tensor:
    with Tensor.test():
        act = model(obs).exp().multinomial().realize()
    return act

#@TinyJit
def calc_sample_loss(states: Tensor, actions: Tensor, rewards: Tensor):
    
    with Tensor.train():

        actions_mask = actions.one_hot(int(env.action_space.n))
        
        #print(actions_mask.shape)
        
        log_probs = model(states)

        #print(log_probs.shape)

        #print(rewards.shape)
        psuedo_loss =  (rewards * (log_probs * actions_mask).sum(axis = 1)).sum()

    return psuedo_loss

for update in range(NUM_UPDATES):

    returns = []
    sample_losses = Tensor(0.0)

    for sample in range(NUM_SAMPLES):
        rewards = []
        actions = []
        states  = []
        obs, info = env.reset()

        sample_over = False

        while not sample_over:
            states.append(obs)

            action = get_action(Tensor(obs))
            #print(action)
            actions.append(action.item())

            obs, reward, terminated, truncated, info =  env.step(action.item())
            
            rewards.append(reward)

            sample_over = terminated or truncated


        returns.append(sum(rewards))

        rewards = rewards_to_go(Tensor(rewards))

        #print(rewards.shape)

        states = Tensor(states)
        #print(states.shape)
        actions = Tensor(actions)
        
        #print(actions.shape)
        
        
        sample_losses =  sample_losses +  calc_sample_loss(states, actions, rewards)

    with Tensor.train():

        opt.zero_grad()

        total_loss = (-sample_losses/NUM_SAMPLES).backward()
        opt.step()

    print(f'Episode: {update + 1}, Average Return: {sum(returns)/len(returns)}')

        
with Tensor.test():

    obs, info = eval_env.reset()

    sample_over = False

    while not sample_over:
        

        action = get_action(Tensor(obs))
        #print(action)
        

        obs, reward, terminated, truncated, info =  eval_env.step(action.item())
        
        

        sample_over = terminated or truncated
