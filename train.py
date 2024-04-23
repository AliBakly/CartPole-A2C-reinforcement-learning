import torch
import torch.nn as nn
import gymnasium as gym

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

# Create enviroment
env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Create actor and critic
actor = Actor(input_size, output_size)
critic = Critic(input_size)

# Optimizers for the actor and critic
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

# Training loop
max_steps = 500000
step = 0
start_state, _ = env.reset()
worker_state = [start_state for _ in range(K)]
while step < max_steps:
    advantages = []
    states = []
    actions = []
    rewards = []
    done = False
    for i in range(K):
        state = worker_state[i]
        action_prob = actor(torch.tensor(state))
        action = torch.argmax(action_prob).item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        R = reward + gamma * critic(torch.tensor(next_state)).item()
        advantage = R - critic(torch.tensor(state)).item()
        worker_state[i] = next_state
        states.append(state)
        actions.append(action)
        advantages.append(advantage)

        done = terminated or truncated # kanske sen

    # Update the actor
    log_probs = torch.log(actor(torch.tensor(states)))
    actor_loss = -torch.sum(log_probs * torch.tensor(advantages))
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Update the critic
    critic_loss = torch.sum(torch.tensor(advantages, requires_grad=True)**2)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    step += 1


# yoooooo ali b

