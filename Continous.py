import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math

class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, output_size)
        self.fc_log_std = nn.Parameter(torch.zeros(output_size))
        
        self.activation = nn.Tanh() # torch.tanh alternativt

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        mean = self.fc_mean(x)

        log_std = self.fc_log_std
        return mean, log_std


    

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


# Set hyperparameters
lr_actor = 3e-4
lr_critic = 1e-3
gamma = 0.99
K = 6
n = 6

log_interval = (1000// (K*n)) * K*n#1000
eval_interval = (20000// (K*n)) * K*n#5000
num_eval_episodes = 10
# Create enviroment
worker_envs = [gym.make('InvertedPendulum-v4') for _ in range(K)]
input_size = worker_envs[0].observation_space.shape[0]
output_size = 1#worker_envs[0].action_space.n

# Create actor and critic
actor = Actor(input_size, output_size)
critic = Critic(input_size)

# Optimizers for the actor and critic
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

#
episode_returns = [[] for _ in range(K)]
# [[r r r], [r,r,,r], [r,r,r]]
episode_count = 0

# Training loop
max_steps = 300000
step = 0
#start_state, _ = env.reset()
worker_state = [worker_env.reset()[0] for worker_env in worker_envs]
rewards = np.zeros(K)

train_return_history = []
eval_return_history = []

train_loss_actor_history = []
train_loss_critic_history = []
value_trajectories = []

prob_mask = 0.1
while step <= max_steps:
    advantages = []
    returns = [[] for _ in range(K)] #[[ 1 2 3], [1 2 3 4 5]]
    k_n_states = [[] for _ in range(K)]
    k_n_rewards = [[] for _ in range(K)]
    log_probs =  [[] for _ in range(K)]#torch.zeros(K)
    last_K_state = [None for _ in range(K)]
    dones = [False for _ in range(K)]
    for i in range(K):
        for j in range(n):
            state = worker_state[i]
            k_n_states[i].append(state) #[[.....]]
            mean, log_std = actor(torch.tensor(state, dtype=torch.float32))
           
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_probs[i].append(dist.log_prob(action))
            action = torch.clamp(action, -3, 3)
            next_state, reward, terminated, truncated, _ = worker_envs[i].step(action)
            done = terminated or truncated
            rewards[i] = rewards[i] + reward
            mask = np.random.binomial(1, prob_mask)
            k_n_rewards[i].append(reward*mask)
            worker_state[i] = next_state
            last_K_state[i] = next_state
            dones[i] = terminated
            if done:
                episode_returns[i].append(rewards[i])
                rewards[i] = 0
                state, _ = worker_envs[i].reset()
                worker_state[i] = state
                break # I think?

            
    for i in range(K):
        discounting = []
        N = len(k_n_states[i]) # [[s1, s2], [s1, s2, s3, s4, s5]]
        for j in range(N):
            discounting = [gamma** power for power in range(N - j)] # [gamma^0, gamma^1, gamma^2, gamma^3]
            returns[i].append(np.dot(discounting, k_n_rewards[i][j:]) + (1-dones[i])*gamma**(N-j) * critic(torch.tensor(last_K_state[i], dtype=torch.float32)).item())
            
            #discounting = [gamma** power for power in range(N - j)] # [gamma^0, gamma^1, gamma^2, gamma^3]
            #returns[i].append(np.dot(discounting, k_n_rewards[i][:N-j]) + gamma**(N-j) * critic(torch.tensor(last_K_state[i])).item())
            # if K=N=1, discounting = [1],
            # k_n_rewards[j:] = [r_t]
            
            # r_t +gamma *critic(s_t+1)
            # discounting = [gamma^0, gamma^1, gamma^2, ...gamma^(N-1)]	 ]
            # k_n_rewards[0][0:] = [r_0, r_1, r_2, ... r_(N-1)]
            
    # K_n states= [[s1, s2], [s1, s2, s3, s4, s5]]
    # Log_probs = [[log_prob_1, log_prob_2], [log_prob_1, log_prob_2, log_prob_3, log_prob_4, log_prob_5]]
    k_n_states_flat = [state for worker_states in k_n_states for state in worker_states]
    returns_flat = [ret for worker_returns in returns for ret in worker_returns]
    log_probs_flat = [prob for worker_probs in log_probs for prob in worker_probs]

    
    advantages = torch.tensor(returns_flat).float() - critic(torch.tensor(k_n_states_flat).float()).squeeze(-1)
    actor_loss = -torch.mean(torch.stack(log_probs_flat).float() * advantages.clone().detach())
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_loss = nn.MSELoss()(critic(torch.tensor(k_n_states_flat).float()).squeeze(-1), torch.tensor(returns_flat).float())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Logging
    if step % (log_interval) == 0:
        avg_returns = [np.mean(returns) for returns in episode_returns if len(returns) > 0]
        avg_return = np.mean(avg_returns)

        train_return_history.append(avg_return)
        train_loss_actor_history.append(actor_loss.item())
        train_loss_critic_history.append(critic_loss.item())
        
        print(f"Step {step}: Average episodic return = {avg_return:.2f}")
        print(f"Step {step}: Critic loss = {critic_loss.item():.4f}")
        print(f"Step {step}: Actor loss = {actor_loss.item():.4f}")
        episode_returns = [[] for _ in range(K)]
        


    # Evaluation
    if step % (eval_interval) == 0:
        eval_env = gym.make('InvertedPendulum-v4')  # Create a new environment for evaluation
        eval_returns = []
        
        trajectory_states = []
        trajectory_values = []
        for i in range(num_eval_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_return = 0
            
            
            while not done:
                mean, log_std = actor(torch.tensor(state, dtype=torch.float32))
                
                action = torch.clamp(mean.detach(), -3, 3) # instead of sampling from the distribution, we use the mean for the action 
                
                state, reward, terminated, truncated, _ = eval_env.step(action)
                
                episode_return += reward
                done = terminated or truncated
                
                if i == 0:
                    trajectory_states.append(state)
                    value = critic(torch.tensor(state, dtype=torch.float32)).item()
                    trajectory_values.append(value)

            eval_returns.append(episode_return)

        plt.figure(figsize=(10, 5))
        plt.plot(range(len(trajectory_states)), trajectory_values)
        plt.xlabel('Time Step')
        plt.ylabel('Value Function')
        plt.title('Value Function on Sampled Trajectory')
        plt.show()
        
        value_trajectories.append(trajectory_values)
        avg_eval_return = np.mean(eval_returns)
        eval_return_history.append(avg_eval_return)
        
        print(f"Step {step}: Average evaluation return = {avg_eval_return:.2f}")

    step += K*n


plt.figure(figsize=(10, 5))
plt.plot(range(len(train_return_history)), train_return_history)
plt.xlabel('Time Step')
plt.ylabel('Average Return')
plt.title('Return during Training')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(eval_return_history)), eval_return_history)
plt.xlabel('Time Step')
plt.ylabel('Average Return')
plt.title('Return during Evauluation')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(train_loss_critic_history)), train_loss_critic_history)
plt.xlabel('Time Step')
plt.ylabel('Loss')
plt.title('Loss of Critic during Training')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(train_loss_actor_history)), train_loss_actor_history)
plt.xlabel('Time Step')
plt.ylabel('Loss')
plt.title('Loss of Actor during Training')
plt.show()

max_length = max(len(lst) for lst in value_trajectories)

padded_trajectories = []
for lst in value_trajectories:
    padded_lst = lst + [lst[-1]] * (max_length - len(lst))
    padded_trajectories.append(padded_lst)

means = np.mean(padded_trajectories, axis=0)

for i, lst in enumerate(value_trajectories):
    plt.plot(range(len(lst)), lst, color='blue', alpha=0.1)

plt.plot(range(len(means)), means, color='red', label='Mean Value Function')
plt.xlabel('Time Step')
plt.ylabel('Value Function')
plt.title('Value Function on Sampled Trajectories')
plt.legend()
plt.show()

# Close the worker environments
for env in worker_envs:
    env.close()
    