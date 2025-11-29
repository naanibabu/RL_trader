# DQN Stock Trading Agent

## Project Overview
This project implements a **Deep Q-Network (DQN) reinforcement learning agent** that learns to trade multiple stocks simultaneously. The agent makes buy/sell/hold decisions to maximize portfolio returns using deep neural networks and experience replay.

---

## Executive Summary

### Problem Statement
Develop an intelligent trading agent that can learn optimal trading strategies across multiple stocks (AAPL, MSI, SBUX) to maximize portfolio returns.

### Solution
A DQN-based reinforcement learning agent that:
- Observes normalized stock price states
- Takes discrete actions (buy/sell/hold for each stock)
- Learns from experience replay with neural network function approximation
- Achieves better-than-random performance through Q-learning

---

## Technical Architecture

### Core Components

#### 1. **Data Pipeline** (`get_data()`)
- Loads historical stock price data from CSV
- Returns T × 3 matrix where:
  - T = number of timesteps
  - 3 = number of stocks (AAPL, MSI, SBUX)
- Data format: (timesteps, stocks)

#### 2. **State Normalization** (`get_scaler()`)
- Initializes MultiStockEnv with training data
- Executes random actions to generate baseline states
- Fits scikit-learn `StandardScaler` on collected states
- **Purpose**: Ensures consistent feature scaling across train/test phases
- Saved as `scaler.pkl` for reproducible inference

#### 3. **Episode Execution** (`play_one_episode()`)
Executes one complete trading episode:

**Process Flow:**
Reset Environment → Get Initial State
                ↓
     Transform State with Scaler
                ↓
          While Not Done:
                ├─ Agent selects action
                ├─ Environment executes action
                ├─ Receive: next_state, reward, done, info
                ├─ Transform next_state with scaler
                ├─ If Training:
                │     ├─ Store transition in replay buffer
                │     └─ Sample batch & perform gradient updates
                └─ Update state
                ↓
     Return Final Portfolio Value

#### 4. **Training/Testing Pipeline**

**Training Mode** (`-m train`):
- Episodes: 2,000
- Uses training data (first 50% of historical data)
- Agent explores environment (epsilon-greedy policy)
- Updates neural network weights via experience replay
- Saves: model weights (`dqn.weights.h5`) + scaler (`scaler.pkl`)

**Testing Mode** (`-m test`):
- Loads pre-trained weights and scaler
- Uses test data (last 50% of historical data)
- Sets epsilon=0.01 (minimal exploration, mostly exploitation)
- No weight updates (inference only)
- Evaluates learned strategy performance

---

## **Training and Testing the agent**
-  `python main.py -m train`

- `python plot_rl_rewards.py -m train`

- `python main.py -m test`

- `python plot_rl_rewards.py -m test`
---
