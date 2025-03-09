import numpy as np
import matplotlib.pyplot as plt

# Define Environment: Robot Arm's Movements in Ashok Leyland Factory
actions = ["Move Left", "Move Right", "Extend Arm", "Retract Arm", "Weld"]
num_actions = len(actions)
num_states = 10  # Robot has 10 different positions along the assembly line

# Initialize Q-Table (States x Actions)
Q_table = np.zeros((num_states, num_actions))

# RL Parameters
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0  # Start with full exploration
exploration_decay = 0.995  # Decay exploration over time
num_episodes = 1000

# Reward System (Based on Welding Accuracy, Energy Use, Time Taken)
reward_matrix = np.random.randint(-10, 10, (num_states, num_actions))
reward_matrix[:, 2] = 10  # Reward 'Extend Arm' for correct welding position
reward_matrix[:, 4] = 15  # Reward 'Weld' action
reward_matrix[:, 0] = -5  # Penalize unnecessary movements

# Train the RL Agent (Q-Learning)
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Start at a random position

    while True:
        # Exploration vs Exploitation
        if np.random.rand() < exploration_rate:
            action = np.random.randint(0, num_actions)  # Explore
        else:
            action = np.argmax(Q_table[state, :])  # Exploit best known action
        
        # Take action and get reward
        next_state = np.random.randint(0, num_states)  # Robot moves to a new state
        reward = reward_matrix[state, action]

        # Update Q-Table using Bellman Equation
        Q_table[state, action] = Q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state, :]) - Q_table[state, action]
        )

        state = next_state  # Move to next state

        if np.random.rand() < 0.1:  # End episode randomly (simulating task completion)
            break

    exploration_rate *= exploration_decay  # Reduce exploration over time

# Visualizing Learning Progress
plt.figure(figsize=(10, 5))
plt.plot(np.max(Q_table, axis=1), marker='o', linestyle='-')
plt.xlabel("Robot's Position (State)")
plt.ylabel("Best Q-Value (Expected Reward)")
plt.title("Reinforcement Learning: Optimizing Ashok Leyland's Robotic Arm")
plt.grid(True)
plt.show()

# Print Final Learned Q-Table
print("Final Q-Table (State-Action Values):\n", np.round(Q_table, 2))
