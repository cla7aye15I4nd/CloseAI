import gc 
import time
import math
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from itertools import count

from memory import ReplayMemory
from utils import device, Transition, make_env, get_arguments
from utils import load_model, store_model, test_model
from torch.nn.utils import clip_grad_norm_

from models import DQN

# hyper-parammater
BATCH_SIZE = 32

lr = 6.25e-5

GAMMA = 0.99
CHANNEL_NUM = 4
TARGET_UPDATE = 10000

EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 1e6

replay_start_size = 10000

skip_frame = 1

# Pong: PongNoFrameskip-v4
# Krull: KrullNoFrameskip-v4
# Tutankham: TutankhamNoFrameskip-v4
# Atlantis: AtlantisNoFrameskip-v4
# Freeway: FreewayNoFrameskip-v4
# Beam Rider: BeamRiderNoFrameskip-v4

env_id = 'KrullNoFrameskip-v4'
env = make_env(env_id)
args = get_arguments()

action_num = env.action_space.n
_, height, width = env.reset().shape


if args.load_model:
    print('Loading model...')
    steps, memory, policy_net, target_net = load_model()
else:
    policy_net = DQN(CHANNEL_NUM, height, width, action_num).to(device)
    target_net = DQN(CHANNEL_NUM, height, width, action_num).to(device)
    memory = ReplayMemory(50000)
    steps = 0
        
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

def initial_state(screen):
    state = torch.cat(tuple(torch.tensor(screen) for _ in range(CHANNEL_NUM)))
    return state.reshape(1, CHANNEL_NUM, height, width)

def get_next_state(state, screen):
    next_state = torch.zeros(1, CHANNEL_NUM, height, width, dtype=torch.uint8)
    next_state[:, :CHANNEL_NUM-1, :, :] = state[:, 1:, :, :]
    next_state[0][CHANNEL_NUM-1] = torch.from_numpy(screen.reshape(height, width))
    return next_state

def eps_threshold(steps):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps / EPS_DECAY)

def select_action(state, steps):
    if random.random() <= eps_threshold(steps):
        return env.action_space.sample()
    
    with torch.no_grad():
        return policy_net(state.to(device)).max(1)[1].item()

def compute_loss(transitions):
    batch = Transition(*zip(*transitions))

    state = torch.cat(batch.state).to(device)
    next_state = torch.cat(batch.next_state).to(device)
    action = torch.cat(tuple(map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    reward = torch.cat(tuple(map(lambda r: torch.tensor([[r]], device=device), batch.reward)))
    done = torch.cat(tuple(map(lambda d: torch.tensor([[1 if d else 0]], device=device), batch.done)))
    
    curr_q_value = policy_net(state).gather(1, action)
    target_next_q_values = target_net(next_state)
    
    # double
    # next_q_values = policy_net(next_state)
    # next_actions = next_q_values.max(1)[1].reshape(-1, 1)
    # next_q_value = target_next_q_values.gather(1, next_actions)
    
    # no double
    next_q_value = target_next_q_values.max(dim=1, keepdim=True)[0].detach().float()
    
    mask = 1 - done
    target = (reward + GAMMA * next_q_value * mask).float().to(device)

    loss = F.smooth_l1_loss(curr_q_value, target)
    return loss

def optimize_model():
    global memory
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    
    loss = compute_loss(transitions)
    
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy_net, 10)
    optimizer.step()

    return loss.item()

def main():
    global memory, policy_net, target_net, steps, env_id

    loss = -1
    action = 0
    
    while steps < args.frame_num:
        screen = env.reset()
        state = initial_state(screen)

        score = 0
        for t in count():
            # env.render()

            if t % skip_frame == 0:
                action = select_action(state, steps)
            
            next_screen, reward, done, info = env.step(action)  

            score += reward
            next_state = get_next_state(state, next_screen)

            memory.push(state, action, next_state, reward, done)

            state = next_state

            steps += 1
            if len(memory) > replay_start_size:
                loss = optimize_model()
                if steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if steps % 10000 == 0:
                    store_model(steps, memory, policy_net, target_net)
                    test_model(env_id, policy_net)

            if done:
                gc.collect()
                break
            
        print('[{}] {}/{} steps, {} scores, {} loss : {}'.format(time.strftime("%H:%M:%S", time.localtime()) , steps, args.frame_num, score, eps_threshold(steps), loss))

    store_model(steps, memory, policy_net, target_net)

if __name__ == '__main__':
    main()

