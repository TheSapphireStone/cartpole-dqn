import torch
import numpy as np
import gymnasium as gym
import random
from models.dqn import DQN
from replay.replay_buffer import ReplayBuffer
from utils import plot_rewards
from config import CONFIG
import torch.optim as optim
import torch.nn as nn

def train():
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=CONFIG["lr"])
    buffer = ReplayBuffer(CONFIG["buffer_size"])

    epsilon = CONFIG["epsilon_start"]
    rewards_log = []

    for ep in range(CONFIG["episodes"]):
        state, _ = env.reset()
        ep_reward = 0

        for step in range(500):

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = q_net(torch.tensor(state).float())
                    action = torch.argmax(q_vals).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(buffer) >= CONFIG["batch_size"]:
                states, actions, rewards, next_states, dones = buffer.sample(CONFIG["batch_size"])

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                targets = rewards + CONFIG["gamma"] * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(CONFIG["epsilon_min"], epsilon * CONFIG["epsilon_decay"])

        if ep % CONFIG["target_update_freq"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {ep}: Reward = {ep_reward}")
        rewards_log.append(ep_reward)

    plot_rewards(rewards_log)

if __name__ == "__main__":
    train()
