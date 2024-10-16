import gymnasium as gym
from tinygrad import Tensor, nn, TinyJit
from typing import Tuple

NUM_UPDATES = 1000

NUM_SAMPLES = 10

DISCOUNT_FACTOR = 0.99



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





if __name__ == "__main__":

    # Initialise the environment
    eval_env = gym.make("LunarLander-v3", render_mode="human")

    env = gym.make("LunarLander-v3")
    #print(env.observation_space)
    #print(env.action_space.n)

    model = PolicyNet(env.observation_space.shape[0], int(env.action_space.n), [24])

    opt = nn.optim.Adam(nn.state.get_parameters(model))

    #@TinyJit
    def get_action(obs: Tensor) -> Tensor:
        log_probs = model(obs)
        probs = log_probs.exp()
        ret = probs.multinomial().realize()
        num_actions = probs.shape[0]
        
        # Create an action mask
        action_indices = Tensor.arange(num_actions)
        action_mask = (action_indices == ret).cast('float32')
        
        # Compute the log probability of the selected action
        curr_log_prob = (log_probs * action_mask).sum()
        return ret, curr_log_prob



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
    

    def eval():
        obs, info = eval_env.reset()

        sample_over = False
        while not sample_over:
            action, curr_log_prob = get_action(Tensor(obs))
            actions.append(action)
            obs, reward, terminated, truncated, info =  eval_env.step(action.item())
            sample_over = terminated or truncated

            env.render()

for update in range(NUM_UPDATES):

    with Tensor.train():
        # Initialize loss as a Tensor
        loss = Tensor(0.0)
                
        epoch_rew = 0

        for sample in range(NUM_SAMPLES):

            obs, info = env.reset()
            states, actions, rewards = [], [], []
            log_probs = Tensor

            init_log_probs = False
                

            sample_over = False
            while not sample_over:

                states.append(obs)

                action, curr_log_prob = get_action(Tensor(obs))
                actions.append(action)
                        
                if not init_log_probs:
                    log_probs = curr_log_prob
                    init_log_probs = True
                else:
                    log_probs = log_probs.cat(curr_log_prob)

                obs, reward, terminated, truncated, info =  env.step(action.item())
                rewards.append(reward)

                sample_over = terminated or truncated

            epoch_rew += sum(rewards)
            reward_to_go = rewards_to_go(Tensor(rewards))

            # Use standard addition instead of in-place addition
            loss = loss + (-reward_to_go * log_probs).sum()

        print(f'Average Sample Reward: {epoch_rew/NUM_SAMPLES}')
                
        loss = loss / NUM_SAMPLES
        opt.zero_grad()
        loss.backward()
        opt.step()

