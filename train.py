import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, continous = False):
        super(Actor, self).__init__()
        self.continous = continous
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh() # torch.tanh alternativt
        if continous:
            self.fc_log_std = nn.Parameter(torch.zeros(output_size))
        else:
            self.policy = nn.Softmax(dim=-1)  # Softmax for discrete action space
        

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        if self.continous:
            return x, self.fc_log_std
        
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

def env_step(k_n_states, k_n_rewards, log_probs, last_K_state, dones, actor, rewards, worker_state, worker_envs, i, prob_mask, episode_returns, continous=False):
    state = worker_state[i]
    k_n_states[i].append(state) 
    if continous:
        mean, log_std = actor(torch.tensor(state, dtype=torch.float32))

        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_probs[i].append(dist.log_prob(action))
        action = torch.clamp(action, -3, 3)
    else:
        action_prob = actor(torch.tensor(state))
        action = torch.multinomial(action_prob, 1).item()
        log_probs[i].append(torch.log(action_prob[action]))
        
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
    
    return done

def update_params(optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def plot_result(return_history, xlabel, ylabel, title):

    plt.figure(figsize=(10, 5))
    if type(return_history[0]) == list:
        returns_array = np.array(return_history)
        mean_returns = np.mean(returns_array, axis=0)
        min_returns = np.min(returns_array, axis=0)
        max_returns = np.max(returns_array, axis=0)
        
        plt.plot(range(len(mean_returns)), mean_returns, label='Mean')
        plt.fill_between(range(len(mean_returns)), min_returns, max_returns, alpha=0.2)
    else:
        plt.plot(range(len(return_history)), return_history)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Set hyperparameters
def train(lr_actor, lr_critic, gamma, K, n, env_name, continous, log_interval, eval_interval, num_eval_episodes, max_steps, prob_mask, seeds = [69], record_video = False):
    train_return_history_all = []
    eval_return_history_all = []
    train_loss_actor_history_all = []
    train_loss_critic_history_all = []
        
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        log_interval = (log_interval// (K*n)) * K*n#1000
        eval_interval = (eval_interval// (K*n)) * K*n#5000

        # Create enviroment
        worker_envs = [gym.make(env_name) for _ in range(K)]

        input_size = worker_envs[0].observation_space.shape[0]
        if continous:
            output_size = 1
        else:
            output_size = worker_envs[0].action_space.n

        # Create actor and critic
        actor = Actor(input_size, output_size, continous=continous)
        critic = Critic(input_size)

        # Optimizers for the actor and critic
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        #
        episode_returns = [[] for _ in range(K)]
        # [[r r r], [r,r,,r], [r,r,r]]
        episode_count = 0

        # Training loop
        step = 0
        #start_state, _ = env.reset()
        worker_state = [worker_env.reset(seed=seed)[0] for worker_env in worker_envs]
        rewards = np.zeros(K)

        train_return_history = []
        eval_return_history = []

        train_loss_actor_history = []
        train_loss_critic_history = []
        value_trajectories = []

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
                    done = env_step(k_n_states, k_n_rewards, log_probs, last_K_state, dones, actor, rewards, worker_state, worker_envs, i, prob_mask, continous = continous, episode_returns=episode_returns)
                    if done:
                        break
            
            for i in range(K):
                discounting = []
                N = len(k_n_states[i]) # [[s1, s2], [s1, s2, s3, s4, s5]]
                for j in range(N):
                    discounting = [gamma** power for power in range(N - j)] # [gamma^0, gamma^1, gamma^2, gamma^3]
                    returns[i].append(np.dot(discounting, k_n_rewards[i][j:]) + (1-dones[i])*gamma**(N-j) * critic(torch.tensor(last_K_state[i], dtype=torch.float32)).item())

            k_n_states_flat = [state for worker_states in k_n_states for state in worker_states]
            returns_flat = [ret for worker_returns in returns for ret in worker_returns]
            log_probs_flat = [prob for worker_probs in log_probs for prob in worker_probs]

    
            advantages = torch.tensor(returns_flat).float() - critic(torch.tensor(k_n_states_flat).float()).squeeze(-1)
            actor_loss = -torch.mean(torch.stack(log_probs_flat).float() * advantages.clone().detach())
            critic_loss = nn.MSELoss()(critic(torch.tensor(k_n_states_flat).float()).squeeze(-1), torch.tensor(returns_flat).float())
            
            update_params(actor_optimizer, actor_loss)
            update_params(critic_optimizer, critic_loss)

            # Logging
            if step % (log_interval) == 0 and step > 0:
                avg_returns = [np.mean(returns) for returns in episode_returns if len(returns) > 0]
                if avg_returns:
                    avg_return = np.mean(avg_returns)
                    
                    train_return_history.append(avg_return)
                    train_loss_actor_history.append(actor_loss.item())
                    train_loss_critic_history.append(critic_loss.item())
                    
                    print(f"Step {step}: Average episodic return = {avg_return:.2f}")
                    print(f"Step {step}: Critic loss = {critic_loss.item():.4f}")
                    print(f"Step {step}: Actor loss = {actor_loss.item():.4f}")
                    episode_returns = [[] for _ in range(K)]

            # Evaluation
            if step % (eval_interval) == 0 and step > 0:
                eval_env = gym.make(env_name, render_mode = "rgb_array")  # Create a new environment for evaluation
                eval_returns = []
                
                trajectory_states = []
                trajectory_values = []
                for i in range(num_eval_episodes):
                    state, _ = eval_env.reset(seed=seed)
                    
                    if i == 0 and record_video and (step + eval_interval > max_steps):
                        eval_env = RecordVideo(eval_env, "./mp4") 
                    
                    done = False
                    episode_return = 0
                    
                    while not done:
                        if continous:
                            mean, _ = actor(torch.tensor(state, dtype=torch.float32))
                            action = torch.clamp(mean.detach(), -3, 3)
                        else:
                            action_prob = actor(torch.tensor(state))
                            action = torch.argmax(action_prob).item()
                        
                        state, reward, terminated, truncated, _ = eval_env.step(action)
                        episode_return += reward
                        
                        if i == 0 and record_video and (step + eval_interval > max_steps):
                            eval_env.render()
                            
                        done = terminated or truncated
                        
                        if i == 0:
                            trajectory_states.append(state)
                            value = critic(torch.tensor(state, dtype=torch.float32)).item()
                            trajectory_values.append(value)
                            

                    eval_returns.append(episode_return)
                
                plot_result(trajectory_values, 'Time Step', 'Value Function', 'Value Function on Sampled Trajectory')
                
                value_trajectories.append(trajectory_values)
                avg_eval_return = np.mean(eval_returns)
                eval_return_history.append(avg_eval_return)
                
                print(f"Step {step}: Average evaluation return = {avg_eval_return:.2f}")

            step += K*n


        train_return_history_all.append(train_return_history)
        eval_return_history_all.append(eval_return_history)
        train_loss_actor_history_all.append(train_loss_actor_history)
        train_loss_critic_history_all.append(train_loss_critic_history)    

        # DO THIS OUTSSIDE OF THE SEED LOOP???
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
            
    plot_result(train_return_history_all, 'Time Step', 'Average Return', 'Return during Training')
    plot_result(eval_return_history_all, 'Time Step', 'Average Return', 'Return during Evaluation')
    plot_result(train_loss_critic_history_all, 'Time Step', 'Loss', 'Loss of Critic during Training')
    plot_result(train_loss_actor_history_all, 'Time Step', 'Loss', 'Loss of Actor during Training')

### Testing
        