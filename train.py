import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import time as time

class Actor(nn.Module):
    """
    Actor network for policy-based reinforcement learning.

    Args:
        input_size (int): Dimension of the input (state) space.
        output_size (int): Dimension of the output (action) space.
        hidden_size (int, optional): Dimension of hidden layers. Default is 64.
        continous (bool, optional): Flag to indicate if the action space is continuous. Default is False.
    """
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
        """
        Forward pass of the actor network.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Action probabilities or mean and log standard deviation for continuous action space.
        """ 
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        if self.continous:
            return x, self.fc_log_std
        
        return self.policy(x)

class Critic(nn.Module):
    """
    Critic network for value-based reinforcement learning.

    Args:
        input_size (int): Dimension of the input (state) space.
        hidden_size (int, optional): Dimension of hidden layers. Default is 64.
    """
    def __init__(self, input_size, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        """
        Forward pass of the critic network.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            torch.Tensor: State value.
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def env_step(k_n_states, k_n_rewards, log_probs, last_K_state, terminations, actor, rewards, worker_state,
             worker_envs, i, prob_mask, episode_returns, continous=False, device = "cpu"):
    """
    Perform an environment step for a single worker and updates all the incoming arguments.

    Args:
        k_n_states (list): List to store states for each worker.
        k_n_rewards (list): List to store rewards for each worker.
        log_probs (list): List to store log probabilities of actions.
        last_K_state (list): List to store the last state of each worker (for bootstrap).
        terminations (list): List to store termination flags for each worker.
        actor (Actor): Actor network.
        rewards (list): List to accumulate rewards for each worker.
        worker_state (list): Current state for each worker.
        worker_envs (list): List of environment instances for each worker.
        i (int): Index of the current worker.
        prob_mask (float): Probability mask for rewards.
        episode_returns (list): List to store episodic returns for each worker.
        continous (bool, optional): Flag to indicate if the action space is continuous. Default is False.
        device (str, optional): Device to perform computations on. Default is "cpu".

    Returns:
        bool: Flag indicating if the episode has ended.
    """
    state = worker_state[i]
    k_n_states[i][-1].append(state) # Append state to the last episode for worker i
    if continous:
        mean, log_std = actor(torch.tensor(state, dtype=torch.float32).to(device))

        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std) # Normal distribution for continous action space
        action = dist.sample()
        log_probs[i].append(dist.log_prob(action)) 
        action = torch.clamp(action, -3, 3)
    else:
        action_prob = actor(torch.tensor(state).to(device)) # Get action probabilities
        action = torch.multinomial(action_prob, 1).item() # Sample action from the action probability distribution
        log_probs[i].append(torch.log(action_prob[action]))
        
    next_state, reward, terminated, truncated, _ = worker_envs[i].step(action)
    done = terminated or truncated
    rewards[i] = rewards[i] + reward
    mask = np.random.binomial(1, prob_mask)
    k_n_rewards[i][-1].append(reward*mask) # Append reward to the last episode for worker i with probability prob_mask
    
    worker_state[i] = next_state
    last_K_state[i][-1] = next_state
    terminations[i][-1] = terminated # For bootstraping 
    if done: # If an episode ends, we append the episodic return to the episode_returns list and reset the environment
        episode_returns[i].append(rewards[i])
        rewards[i] = 0
        state, _ = worker_envs[i].reset()
        worker_state[i] = state
    
    return done

def update_params(optimizer, loss):
        """
        Update network parameters using the given optimizer and loss.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            loss (torch.Tensor): Computed loss.
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# Plotting function MIGHT REMOVE !!!
def plot_result(return_history, xlabel, ylabel, title, range_step = 1, min_returns_all = None, max_returns_all = None):

    plt.figure(figsize=(8, 6))
    if min_returns_all is not None and max_returns_all is not None:
        plt.plot(range(range_step,range_step*len(return_history) + range_step, range_step), return_history, label='Mean')
        plt.fill_between(range(range_step,range_step*len(return_history) + range_step, range_step), min_returns_all, max_returns_all, alpha=0.2, label='Min/Max')
        plt.legend()
    elif type(return_history[0]) == list:
        mean_returns = np.mean(np.array(return_history), axis=0)
        min_returns = np.min(np.array(return_history), axis=0)
        max_returns = np.max(np.array(return_history), axis=0)

        plt.plot(range(range_step,range_step*len(mean_returns) + range_step, range_step), mean_returns, label='Mean')

        plt.fill_between(range(range_step,range_step*len(mean_returns) + range_step, range_step), min_returns, max_returns, alpha=0.2, label='Min/Max')
        plt.legend()
    else:
        min_returns = min(return_history)
        max_returns = max(return_history)
        plt.plot(range(range_step,range_step*len(return_history) + range_step, range_step), return_history)
        plt.ylim(min_returns - 10, max_returns + 10)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# Set hyperparameters
def train(lr_actor, lr_critic, gamma, K, n, env_name, continous, log_interval, eval_interval,
          num_eval_episodes, max_steps, prob_mask, device = "cpu", seeds = [69], record_video = None):
    
    """
    Train the actor-critic model.

    Args:
        lr_actor (float): Learning rate for the actor network.
        lr_critic (float): Learning rate for the critic network.
        gamma (float): Discount factor.
        K (int): Number of workers.
        n (int): Number of steps per worker before updating the network.
        env_name (str): Name of the environment.
        continous (bool): Flag to indicate if the action space is continuous.
        log_interval (int): Interval for logging training information.
        eval_interval (int): Interval for evaluation.
        num_eval_episodes (int): Number of episodes for evaluation.
        max_steps (int): Maximum number of training steps.
        prob_mask (float): Probability mask for rewards.
        device (str, optional): Device to perform computations on. Default is "cpu".
        seeds (list, optional): List of random seeds. Default is [69].
        record_video (str, optional): Path to save evaluation videos. Default is None.

    Returns:
        dict: Dictionary containing training and evaluation results.
    """
    #train_return_history_all = [] # REMOVE
    eval_return_history_all = [] # Eval return history for all seeds: [[], [], []]
    train_loss_actor_history_all = [] # Actor loss history for all seeds: [[], [], []]
    train_loss_critic_history_all = [] # Critic loss history for all seeds: [[], [], []]
    value_trajectories_mean_all = []  # Value function trajectories for all seeds: [[], [], []]
    min_log_returns_all = [] # Min return at each log interval for all seeds: [[], [], []]
    mean_log_returns_all = [] # Mean return at each log interval for all seeds: [[], [], []]
    max_log_returns_all = []  # Max return at each log interval for all seeds: [[], [], []]
    log_interval = (log_interval // (K*n)) * K*n # Adjust log_interval to be divisible by K*n
    eval_interval = (eval_interval // (K*n)) * K*n # Adjust eval_interval to be divisible by K*n
    value_funcs_20_100_500 = [] # REMOVE
    
    for seed in seeds: # Loop over all seeds
        min_log_returns = [] # Min return at each log interval for current seed
        mean_log_returns = [] # Mean return at each log interval for current seed
        max_log_returns = [] # Max return at each log interval for current seed
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create enviroment. We do not use vectorized environments in this implementation.
        worker_envs = [gym.make(env_name) for _ in range(K)]

        input_size = worker_envs[0].observation_space.shape[0]
        if continous:
            output_size = 1 # Continous action only oututs the mean of the normal distribution
        else:
            output_size = worker_envs[0].action_space.n
        
        # Create actor and critic
        actor = Actor(input_size, output_size, continous=continous).to(device)
        critic = Critic(input_size).to(device)

        # Optimizers for the actor and critic
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

        #
        episode_returns = [[] for _ in range(K)] # List of lists to store episodic returns for each worker (only last 1k steps)

        step = 0 # Step counter
        worker_state = [worker_env.reset(seed=seed)[0] for worker_env in worker_envs] # Initial state for each worker
        rewards = np.zeros(K) # Reward accumulator for each worker, reset when an episode ends

        #train_return_history = [] # REMOVE
        eval_return_history = []

        train_loss_actor_history = [] # Actor loss history for current seed
        train_loss_critic_history = [] # Critic loss history for current seed
        value_trajectories_mean = [] # Value function trajectories for current seed
        
        t = 0
        t1 = time.time()
        while step <= max_steps: # Loop over steps
            
            # advantages = [] REMOVE
            returns = [[] for _ in range(K)] # R-values for each worker
            
            # Two lists below are of form [ [[s1, s2], [s1, s2, s3, s4]],  [[s1, s2, s3, s4, s5, s6]]] for K = 2, n = 6
            # Here [[s1, s2], [s1, s2, s3, s4]] are the states for worker 1, where [s1 s2] and [s1 s2 s3 s4] are the states for seperate episodes                               
            k_n_states = [[[]] for _ in range(K)] # K*n states in the form explained above
            k_n_rewards = [[[]] for _ in range(K)] # K*n rewards in the form explained above
            
            log_probs =  [[] for _ in range(K)] # Log probabilities for each worker
            
            # Two lists below are of form [ [s1, s2], [s1]] for K = 2. Here [s1, s2] is the states to bootsrap on 
            # for worker 1, where s1 and s2 are the states for seperate episodes.    
            last_K_state = [[None] for _ in range(K)] # The state we will bootstrap from for each worker [[], []]
            terminations = [[False] for _ in range(K)] # Termination flag for each worker [[], []]
            
                        
            for i in range(K): # Loop over workers
                for j in range(n): # Loop over steps for each worker
                    done = env_step(k_n_states, k_n_rewards, log_probs, last_K_state, terminations, actor, rewards, worker_state,
                                    worker_envs, i, prob_mask, continous = continous, episode_returns = episode_returns, device=device)
                    
                    if done and j < n-1: # If an episode ends before n steps, we append accordingly
                        k_n_states[i].append([])
                        k_n_rewards[i].append([])
                        last_K_state[i].append(None)
                        terminations[i].append(False)

            # Calculate returns (R-values)
            for i in range(K):
                n_episode_breaks = len(k_n_states[i]) # Number of episodes for worker i
                for episode in range(n_episode_breaks):
                    N = len(k_n_states[i][episode]) # Number of steps collected in episode x
                    for j in range(N): 
                        discounting = [gamma** power for power in range(N - j)] # Discounting factor for each step in episode x
                        returns[i].append(np.dot(discounting, k_n_rewards[i][episode][j:]) + (1-terminations[i][episode])*gamma**(N-j) * critic(torch.tensor(last_K_state[i][episode], dtype=torch.float32).to(device)).item())

            # Flatten lists for calculation of loss and advantage
            k_n_states_flat = [x for sublist1 in k_n_states for sublist2 in sublist1 for x in sublist2]
            returns_flat = [ret for worker_returns in returns for ret in worker_returns]
            log_probs_flat = [prob for worker_probs in log_probs for prob in worker_probs]
    
            value = critic(torch.tensor(k_n_states_flat).float().to(device)).squeeze(-1)
            with torch.no_grad(): # No gradient for the advantage calculation
                returns_flat = torch.tensor(returns_flat).float().to(device)
                advantages = returns_flat - value # Calculate advantage

            # Calculate actor and critic loss
            actor_loss = -torch.mean(torch.stack(log_probs_flat).float().to(device) * advantages)
            critic_loss = nn.MSELoss()(value, returns_flat)

            # Update actor and critic
            update_params(actor_optimizer, actor_loss)
            update_params(critic_optimizer, critic_loss)

            # Logging
            if step % (log_interval) == 0 and step > 0:
                #avg_returns = [np.mean(returns) for returns in episode_returns if len(returns) > 0]
                
                # Flatten list of episodic returns and calculate min, max and mean
                episode_returns_flat = [reward for worker in episode_returns for reward in worker]
                if not episode_returns_flat: # if no worker finished an episode we pick the last episode min, max and mean
                    min_log  = min_log_returns[-1]
                    max_log = max_log_returns[-1]
                    mean_log = mean_log_returns[-1]
                else:
                    min_log = min(episode_returns_flat)
                    max_log = max(episode_returns_flat)
                    mean_log = np.mean(episode_returns_flat) # The value here will be averaged over all seeds later on
                    episode_returns = [[] for _ in range(K)] # Reset episodic returns

                min_log_returns.append(min_log)
                max_log_returns.append(max_log)
                mean_log_returns.append(mean_log)
                train_loss_actor_history.append(actor_loss.item())
                train_loss_critic_history.append(critic_loss.item())
                    #if avg_returns:
                    #train_return_history.extend(avg_log(episode_returns))

                # Print logging information
                print(f"Step {step}: Average episodic return = {mean_log:.2f}")
                print(f"Step {step}: Critic loss = {critic_loss.item():.7f}")
                print(f"Step {step}: Actor loss = {actor_loss.item():.7f}")
            

            # Evaluation
            if step % (eval_interval) == 0 and step > 0:
                eval_env = gym.make(env_name, render_mode = "rgb_array")  # Create a new environment for evaluation
                eval_returns = [] # List to store evaluation returns
                trajectory_states = [] # List to store states for value function trajectory
                trajectory_values = [] # List to store values for value function trajectory
                
                first = step == eval_interval
                middle = step == eval_interval* (max_steps // (2*eval_interval))
                last = step + 2*eval_interval > max_steps # Save two just in case
                for i in range(num_eval_episodes): # Loop over evaluation episodes
                    state, _ = eval_env.reset(seed=seed)
                    
                    record_video_condition = (i == 0) and (seed == seeds[0]) and (record_video is not None) and (step + eval_interval > max_steps)
                    
                    # Record video for first eval loop (over num_eval_episodes), the first seed and time we run eval
                    if record_video_condition:
                        eval_env = RecordVideo(eval_env, "./" + record_video) 
                    
                    done = False
                    episode_return = 0
                    
                    while not done:
                        if continous: #
                            with torch.no_grad(): # No gradient for evaluation
                                mean, _ = actor(torch.tensor(state, dtype=torch.float32).to(device)) # Greedy action so we take the most likely action, the mean
                            action = torch.clamp(mean, -3, 3) 
                        else: # Discrete action space
                            with torch.no_grad(): # No gradient for evaluation
                                action_prob = actor(torch.tensor(state).to(device))
                            action = torch.argmax(action_prob).item() # Greedy action
                        
                        state, reward, terminated, truncated, _ = eval_env.step(action)
                        episode_return += reward
                        
                        # Render the env if conditions for recodring video are met
                        if record_video_condition:
                            eval_env.render()
                            
                        done = terminated or truncated
                        
                        if i == 0 : # Only store states and values for the first evaluation episode (arbitrary)
                            trajectory_states.append(state)
                            value = critic(torch.tensor(state, dtype=torch.float32).to(device)).item()
                            trajectory_values.append(value)
                            
                    eval_returns.append(episode_return)
                
                
                plot_result(trajectory_values, 'Time Step', 'Value Function', 'Value Function on Sampled Trajectory')
                if (first or middle or last) and seed == seeds[0]:
                    value_funcs_20_100_500.append(trajectory_values)
                    
                    
                value_trajectories_mean.append(np.mean(trajectory_values))
                avg_eval_return = np.mean(eval_returns)
                eval_return_history.append(avg_eval_return)
                
                print(f"Step {step}: Average evaluation return = {avg_eval_return:.2f}")

            
            step += K*n
        t2 = time.time()
        print(t2-t1)
        #value_funcs_20_100_500.append(value_trajectories) # REMOVE
        value_trajectories_mean_all.append(value_trajectories_mean)
        #train_return_history_all.append(train_return_history) # REMOVE
        eval_return_history_all.append(eval_return_history)
        train_loss_actor_history_all.append(train_loss_actor_history)
        train_loss_critic_history_all.append(train_loss_critic_history)    

        # Close the worker environments
        for env in worker_envs:
            env.close()
        
        min_log_returns_all.append(min_log_returns)
        mean_log_returns_all.append(mean_log_returns)
        max_log_returns_all.append(max_log_returns)
    
    min_log_returns_all = np.min(np.array(min_log_returns_all), axis=0).tolist()
    mean_log_returns_all = np.mean(np.array(mean_log_returns_all), axis=0).tolist()
    max_log_returns_all = np.max(np.array(max_log_returns_all), axis=0).tolist()
    
    plot_result(value_trajectories_mean_all, 'Time Step', 'Value Function', 'Mean Over Value Function Trajectories', range_step= eval_interval)
    #plot_result(train_return_history_all, 'Episode', 'Average Return', 'Return During Training')
    plot_result(mean_log_returns_all, 'Episode', 'Average Return', 'Return During Training', min_returns_all = min_log_returns_all, max_returns_all = max_log_returns_all, range_step = log_interval)
    plot_result(eval_return_history_all, 'Time Step', 'Average Return', 'Return During Evaluation', range_step =eval_interval)
    plot_result(train_loss_critic_history_all, 'Time Step', 'Loss', 'Loss of Critic During Training', range_step =log_interval)
    plot_result(train_loss_actor_history_all, 'Time Step', 'Loss', 'Loss of Actor During Training', range_step =log_interval)

    dict_list = {"value_funcs_20_100_500": value_funcs_20_100_500,
                "value_trajectories_mean_all": value_trajectories_mean_all,
                "mean_log_returns_all": mean_log_returns_all,
                "min_log_returns_all": min_log_returns_all,
                "max_log_returns_all": max_log_returns_all, 
                "eval_return_history_all": eval_return_history_all, 
                "train_loss_critic_history_all": train_loss_critic_history_all, 
                "train_loss_actor_history_all": train_loss_actor_history_all}

    return dict_list

### Testing

def avg_log(arr): # arr = [[x11 x12 x13 x14], [x21 x22], [x31 x32 x33 ]...]
    max_len = max(len(lst) for lst in arr)
    avg_list = []
    for i in range(max_len):
        col = [lst[i] for lst in arr if i < len(lst)]
        avg_list.append(sum(col) / len(col))
    return avg_list

# [m1 m2 m3 m4]    m