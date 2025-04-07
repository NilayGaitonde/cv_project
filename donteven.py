import numpy as np

# Gridworld parameters
GRID_SIZE = 4
GAMMA = 1  # Discount factor
THETA = 1e-4  # Convergence threshold

# Initialize value function (terminal states s_1 and s_16 are 0)
V = np.zeros((GRID_SIZE, GRID_SIZE))

# Define terminal states
terminal_states = [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]

# Iterative policy evaluation
while True:
    delta = 0  # Track changes for convergence
    V_new = np.copy(V)
    
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) in terminal_states:
                continue  # Skip terminal states
            
            # Get neighbors with equal probability (random policy)
            neighbors = []
            if i > 0: neighbors.append(V[i - 1, j])  # Up
            if i < GRID_SIZE - 1: neighbors.append(V[i + 1, j])  # Down
            if j > 0: neighbors.append(V[i, j - 1])  # Left
            if j < GRID_SIZE - 1: neighbors.append(V[i, j + 1])  # Right
            
            # Compute expected value
            V_new[i, j] = -1 + (GAMMA / len(neighbors)) * sum(neighbors)
            
            # Compute max difference for convergence
            delta = max(delta, abs(V_new[i, j] - V[i, j]))
    
    V = V_new  # Update value function
    
    if delta < THETA:
        break  # Stop when values converge

# Print final value function
print("Final Value Function:")
print(np.round(V, 2))
