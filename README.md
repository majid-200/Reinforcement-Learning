# Reinforcement Learning Explorations: From Fundamentals to Advanced Applications

Welcome to my Reinforcement Learning (RL) repository! This collection, featurs projects from the official [Gymnasium](https://gymnasium.farama.org/) documentation and coursework from the Higher School of Computer Science (ESI-SBA).

The goal of this repository is to explore and implement a wide range of RL concepts, from foundational algorithms like Q-Learning to more advanced techniques like Policy Gradients and parallel training.

## 🚀 Getting Started

To run these notebooks, it's recommended to set up a dedicated Python virtual environment.

### Prerequisites
- Python 3.11+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/majid-200/RL.git
    cd RL
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```
    gymnasium[all]
    torch
    numpy
    pandas
    matplotlib
    seaborn
    imageio
    imageio[ffmpeg]
    ale-py
    pygame
    copier
    
    ```
---

## 📚 Part 1: Exploring the Gymnasium Ecosystem

This section contains notebooks based on the official tutorials from the Gymnasium documentation. These projects explore specific features and concepts within the Gymnasium ecosystem, demonstrating proficiency with standard RL tools.

### 1. Custom Environments and Wrappers
*   **Notebooks:** `Gymnasium/Custom_Environment.ipynb` and `Gymnasium/Custom_wrappers.ipynb`

These notebooks introduce two fundamental concepts: creating a custom RL environment from scratch and modifying existing environments using **Wrappers**.

**Concepts Covered:**
*   **Custom Environments:** We implement a `GridWorld-v0` environment, defining its observation space, action space, `reset()` and `step()` methods. This demonstrates the core components required for any RL task.
*   **Wrappers:** Wrappers are a powerful feature for cleanly modifying an environment's behavior without altering its source code. We explore:
    *   `gymnasium.ObservationWrapper`: To transform the observations returned by the environment. The notebook implements a `RelativePosition` wrapper that calculates the vector from the agent to the target.
    *   `gymnasium.ActionWrapper`: To modify the actions passed to the environment. An example shows how to convert a discrete action space to a continuous one for `LunarLander`.
    *   `gymnasium.RewardWrapper`: To change the reward signal. An example demonstrates how to clip rewards to a specific range for numerical stability.

### 2. Customizing MuJoCo Environments
*   **Notebook:** `Gymnasium/Load_Custom_Quadruped_Robot_Environments`

This notebook dives into the powerful MuJoCo physics simulator. We learn how to load a custom robot model and fine-tune its physical and task-specific parameters within Gymnasium.

**Concepts Covered:**
*   **Loading Custom Models:** We load a custom `unitree_go1` quadruped robot model into the standard `Ant-v5` environment, replacing the default ant.
*   **Environment Parameter Tweaking:** The `gym.make()` function allows for deep customization. We systematically adjust:
    *   **Simulation Parameters:** `frame_skip` (to control the simulation timestep `dt`) and `reset_noise_scale` (to improve policy robustness).
    *   **Termination Conditions:** `healthy_z_range` to define the conditions under which an episode ends (e.g., the robot falls over).
    *   **Reward Function:** `forward_reward_weight` and `ctrl_cost_weight` are modified to shape the agent's behavior, encouraging forward movement while penalizing excessive motor torque.

### 3. Action Masking for Improved Performance
*   **Notebook:** `Gymnasium/Action_Masking_Taxi.ipynb`

Action masking is a technique to prevent an RL agent from selecting invalid actions in a given state. This often leads to faster and more stable learning.

**Concepts Covered:**
*   **The Problem of Invalid Actions:** In many environments like `Taxi-v3`, not all actions are valid in every state (e.g., you can't pick up a passenger if you're not at their location).
*   **Using the `action_mask`:** Gymnasium environments can provide an `action_mask` in the `info` dictionary returned by `step()` and `reset()`. This binary mask indicates which actions are currently valid.
*   **Implementation with Q-Learning:** We implement a tabular Q-learning agent and show how to use the action mask to:
    1.  **Constrain Exploration:** Select random actions only from the set of valid ones.
    2.  **Constrain Exploitation:** Select the greedy action (highest Q-value) only from the set of valid ones.
*   **Performance Comparison:** The notebook concludes with a compelling visualization that compares the training performance of agents with and without action masking, clearly demonstrating the benefits of this technique.

### 4. Policy Gradients with REINFORCE for MuJoCo
*   **Notebook:** `Gymnasium/Mujoco_REINFORCE.ipynb` 

This notebook implements the classic REINFORCE algorithm, a foundational policy gradient method, to solve the continuous control task `InvertedPendulum-v4`.

**Concepts Covered:**
*   **Policy Gradient Methods:** Unlike value-based methods (like Q-Learning), policy gradient algorithms directly learn a parameterized policy that maps states to action probabilities.
*   **REINFORCE Algorithm:** We implement the REINFORCE agent from scratch using PyTorch. The key steps include:
    1.  Building a neural network policy that outputs a probability distribution (a Normal distribution for continuous actions).
    2.  Sampling actions from this policy and collecting trajectories (sequences of states, actions, and rewards).
    3.  Calculating Monte-Carlo returns (discounted rewards) for each step in an episode.
    4.  Updating the policy network's weights by performing gradient ascent on the expected return.
*   **Training and Evaluation:** The agent is trained on `InvertedPendulum-v4`, and the importance of running experiments with multiple random seeds is highlighted to ensure robust results. The final learning curve shows the agent successfully learning to balance the pole.

### 5. Speeding Up Training with Vectorized Environments & A2C
*   **Notebook:** `Gymnasium/Vector_A2C.ipynb`

Training RL agents can be slow. This notebook demonstrates how to significantly accelerate the process using **Vectorized Environments** to train an **Advantage Actor-Critic (A2C)** agent.

**Concepts Covered:**
*   **Vectorized Environments:** We use `gym.make_vec` to run multiple instances of an environment in parallel. This allows the agent to collect batches of experience far more efficiently, reducing training time and variance.
*   **Advantage Actor-Critic (A2C):** A synchronous actor-critic algorithm that works naturally with vectorized environments. We implement A2C from scratch, including separate actor and critic networks and the use of **Generalized Advantage Estimation (GAE)** for stable advantage calculation.
*   **Domain Randomization:** A technique for training more robust agents. We create a vectorized environment where each parallel instance has slightly different physical parameters (e.g., gravity, wind power), forcing the agent to learn a policy that generalizes well.
*   **Parallel Training Loop:** The implementation showcases how to process batches of states, actions, and rewards from the parallel environments to perform efficient updates.

### 6. Tabular Q-Learning for Classic RL Problems
*   **Notebooks:** `Gymnasium/Blackjack_Q_Learning.ipynb` and `Gymnasium/Frozenlake_Q_Learning.ipynb`

These notebooks provide a deep dive into **Tabular Q-Learning**, one of the most fundamental RL algorithms, by applying it to two classic environments.

**Concepts Covered:**
*   **Q-Learning:** An off-policy, value-based algorithm that learns a Q-table representing the expected future reward for taking an action in a state.
*   **Epsilon-Greedy Policy:** The core strategy for balancing exploration (trying random actions) and exploitation (choosing the best-known action).
*   **Blackjack (`Blackjack-v1`):**
    *   We build a Q-learning agent to learn the optimal strategy for playing Blackjack.
    *   The notebook includes excellent visualizations of the final learned policy and state-value function, showing when the agent decides to "hit" or "stick."
*   **FrozenLake (`FrozenLake-v1`):**
    *   The agent learns to navigate a frozen lake to reach a goal without falling into holes.
    *   We run experiments across procedurally generated maps of increasing size (`4x4` to `11x11`) to analyze how the algorithm's performance scales with the complexity of the state space.
    *   The learned policy is visualized as a heatmap of arrows, providing an intuitive map of the agent's navigation strategy.

---
## 🏫 Part 2: ESI-SBA Coursework - From-Scratch Implementations

This section contains notebooks developed as part of my coursework at the Higher School of Computer Science (ESI-SBA). These projects focus on implementing core RL algorithms from scratch to build a deep, foundational understanding of their mechanics.

### 1. Tabular Methods: Q-Learning and SARSA
*   **Notebook:** `School Courses/Q_Learning_and_SARSA_for_taxi_v3.ipynb`

This notebook introduces and compares two fundamental tabular RL algorithms: **Q-Learning** (off-policy) and **SARSA** (on-policy).

**Concepts Covered:**
*   **Random Agent Baseline:** Establishes a performance benchmark in the `Taxi-v3` environment.
*   **Q-Learning:** An agent is built from scratch to learn an optimal policy by updating its Q-table based on the maximum possible value of the next state.
*   **SARSA (State-Action-Reward-State-Action):** An on-policy agent that updates its Q-table based on the action it *actually* takes, making its learning dependent on its current policy.
*   **Evaluation and Visualization:** Both agents are trained and evaluated, with visualizations of rewards, steps per episode, and penalties. The final policy is used to generate a video of the agent's performance.

### 2. Deep Q-Networks (DQN and DDQN) for CartPole
*   **Notebook:** `School Courses/PyTorch_DQN_DDQN_Implementations.ipynb`

This project moves from tabular methods to deep reinforcement learning. We implement **Deep Q-Networks (DQN)** and its popular improvement, **Double DQN (DDQN)**, to solve the `CartPole-v1` environment.

**Concepts Covered:**
*   **Deep Q-Network (DQN):** A neural network is built to approximate the Q-function, incorporating key components:
    *   **Experience Replay:** A memory buffer to de-correlate experiences.
    *   **Target Network:** A separate, periodically updated network to stabilize training.
*   **Double DQN (DDQN):** To address DQN's overestimation bias, we implement DDQN, which decouples action selection from action evaluation.
*   **Performance Analysis:** Both agents are trained on `CartPole-v1`, and their learning curves are plotted to compare performance and stability.

### 3. DDQN for High-Dimensional Inputs: Space Invaders
*   **Notebook:** `School Courses/03-ddqn-for-atari.ipynb` (example filename)

This notebook scales up the DDQN algorithm to play the Atari game **Space Invaders** directly from pixel inputs, demonstrating a complete vision-based RL pipeline.

**Concepts Covered:**
*   **Convolutional Neural Networks (CNNs):** A CNN is used as the function approximator to automatically learn relevant spatial features from game screens.
*   **Image Preprocessing & Frame Stacking:** Raw frames are converted to grayscale, resized, and stacked to provide the network with temporal information (e.g., motion).
*   **Dueling DQN Architecture:** This advanced architecture separates the estimation of the state value function (`V(s)`) and the advantage function (`A(s, a)`), leading to better policy evaluation.

### 4. Policy Gradient Methods: REINFORCE and Actor-Critic
*   **Notebooks:** `school-coursework/04-reinforce.ipynb` and `school-coursework/05-actor-critic.ipynb` (example filenames)

These notebooks explore policy gradient methods, which directly optimize the agent's policy instead of first learning a value function. Both are implemented from scratch in PyTorch and tested on `CartPole-v1`.

**Concepts Covered:**
*   **REINFORCE (Monte Carlo Policy Gradient):** This method updates the policy based on the complete return of an entire episode, pushing the agent to take actions that led to high rewards.
*   **Actor-Critic (A2C):** This hybrid approach reduces the high variance of REINFORCE. A **Critic** (value network) estimates the value of states, and the **Actor** (policy network) uses this feedback to make more stable updates.
























## 🏫 Part 2: ESI-SBA Coursework - From-Scratch Implementations

This section contains notebooks developed as part of my coursework at the Higher School of Computer Science (ESI-SBA). These projects focus on implementing core RL algorithms from scratch to build a deep, foundational understanding of their mechanics.

### 1. Tabular Methods: Q-Learning and SARSA
*   **Notebook:** `School Courses/Q_Learning_and_SARSA_for_taxi_v3.ipynb`

This notebook introduces and compares two fundamental tabular RL algorithms: **Q-Learning** (off-policy) and **SARSA** (on-policy).

**Concepts Covered:**
*   **Random Agent Baseline:** Establishes a performance benchmark in the `Taxi-v3` environment.
*   **Q-Learning:** An agent is built from scratch to learn an optimal policy by updating its Q-table based on the maximum possible value of the next state.
*   **SARSA (State-Action-Reward-State-Action):** An on-policy agent that updates its Q-table based on the action it *actually* takes, making its learning dependent on its current policy.
*   **Evaluation and Visualization:** Both agents are trained and evaluated, with visualizations of rewards, steps per episode, and penalties. The final policy is used to generate a video of the agent's performance.

### 2. Deep Q-Networks: From CartPole to Atari
*   **Notebook:** `School Courses/PyTorch_DQN_DDQN_Implementations.ipynb`

This project moves from tabular methods to deep reinforcement learning. We implement **Deep Q-Networks (DQN)** and its improvements to solve both a classic control problem and a complex vision-based task.

**Concepts Covered:**
*   **Deep Q-Network (DQN) for `CartPole-v1`:** A neural network is built to approximate the Q-function, incorporating key components like **Experience Replay** and a **Target Network**.
*   **Double DQN (DDQN) for `CartPole-v1`:** To address DQN's overestimation bias, we implement DDQN, which decouples action selection from action evaluation.
*   **DDQN for `SpaceInvaders-v5`:** The DDQN algorithm is scaled up to play the Atari game directly from pixel inputs. This involves:
    *   **Convolutional Neural Networks (CNNs)** to learn spatial features.
    *   **Image Preprocessing & Frame Stacking** to handle visual and temporal information.
    *   A **Dueling DQN Architecture** to improve policy evaluation.

### 3. Policy Gradient Method: REINFORCE
*   **Notebook:** `School Courses/REINFORCE_from_Scratch_on_CartPole.ipynb`

This notebook explores the REINFORCE algorithm, a foundational policy gradient method that directly optimizes the agent's policy.

**Concepts Covered:**
*   **Policy Parameterization:** A neural network is designed to output a probability distribution over actions.
*   **Monte Carlo Policy Gradient:** The REINFORCE algorithm is implemented from scratch. This method updates the policy based on the complete, discounted return of an entire episode, pushing the agent to take actions that led to high rewards.
*   **Training and Evaluation:** The agent is trained on `CartPole-v1`, and its learning curve demonstrates the effectiveness of the policy gradient approach.

### 4. Policy Gradient Method: Actor-Critic
*   **Notebook:** `School Courses/PyTorch_Actor_Critic_for_CartPole.ipynb`

This project implements an **Actor-Critic (A2C)** algorithm, a more advanced policy gradient method that combines the best of both policy-based and value-based approaches.

**Concepts Covered:**
*   **The Actor-Critic Framework:** This hybrid model uses two networks:
    *   An **Actor** (the policy network) decides which action to take.
    *   A **Critic** (the value network) evaluates the action by estimating the state's value.
*   **Advantage Function:** The critic's feedback is used to calculate the *advantage*, which tells the actor how much better an action was than the average action for that state. This leads to lower variance and more stable updates compared to REINFORCE.
*   **Implementation:** The A2C agent is implemented from scratch in PyTorch and trained on `CartPole-v1` to demonstrate its improved stability and learning speed.
