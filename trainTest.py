import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh() # torch.tanh alternativt
        self.policy = nn.Softmax(dim=-1)  # Softmax for discrete action space

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return self.policy(x)

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
lr_actor = 1e-5
lr_critic = 1e-3
gamma = 0.99
K = 1
n = 1

log_interval = 1000
eval_interval = 20000
num_eval_episodes = 10
# Create enviroment
worker_envs = [gym.make('CartPole-v1') for _ in range(K)]
input_size = worker_envs[0].observation_space.shape[0]
output_size = worker_envs[0].action_space.n

# Create actor and critic
actor = Actor(input_size, output_size)
critic = Critic(input_size)

# Optimizers for the actor and critic
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

#
episode_returns = [[] for _ in range(K)]
episode_count = 0

# Training loop
max_steps = 500000
step = 0
#start_state, _ = env.reset()
worker_state = [worker_env.reset()[0] for worker_env in worker_envs]
rewards = np.zeros(K)

while step < max_steps:
    advantages = []
    states = []
    actions = []
    log_probs = torch.zeros(K)
    for i in range(K):
        state = worker_state[i]
        action_prob = actor(torch.tensor(state))
        action = torch.multinomial(action_prob, 1).item()
        log_probs[i] = torch.log(action_prob[action])
        next_state, reward, terminated, truncated, _ = worker_envs[i].step(action)
        done = terminated or truncated
        rewards[i] = rewards[i] + reward
        
        R = reward + gamma * (1-done)*critic(torch.tensor(next_state)).item()
        advantage = R - critic(torch.tensor(state)).item()
        worker_state[i] = next_state
        states.append(state)
        actions.append(action)
        advantages.append(advantage)

        if done:
            episode_returns[i].append(rewards[i])
            rewards[i] = 0
            state, _ = worker_envs[i].reset()
            worker_state[i] = state

    # Update the actor
    actor_loss = torch.sum(log_probs * torch.tensor(advantages))
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    #Update the critic
    critic_loss = torch.sum(torch.tensor(advantages, requires_grad=True)**2)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Logging
    if step % log_interval == 0:
        avg_returns = [np.mean(returns) for returns in episode_returns]
        avg_return = np.mean(avg_returns)
        print(f"Step {step}: Average episodic return = {avg_return:.2f}")
        print(f"Step {step}: Critic loss = {critic_loss.item():.4f}")
        print(f"Step {step}: Actor loss = {actor_loss.item():.4f}")
        # Log other metrics like entropy, grad norms, etc.
        #episode_returns = [[] for _ in range(K)]
        #episode_count = 0

    # Evaluation
    if step % eval_interval == 0:
        eval_env = gym.make('CartPole-v1')  # Create a new environment for evaluation
        eval_returns = []
        for _ in range(num_eval_episodes):
            state, _ = eval_env.reset()
            done = False
            episode_return = 0
            while not done:
                action_prob = actor(torch.tensor(state))
                action = torch.argmax(action_prob).item()
                state, reward, terminated, truncated, _ = eval_env.step(action)
                episode_return += reward
                done = terminated or truncated
            eval_returns.append(episode_return)
        avg_eval_return = np.mean(eval_returns)
        print(f"Step {step}: Average evaluation return = {avg_eval_return:.2f}")


    step += 1

# Close the worker environments
for env in worker_envs:
    env.close()