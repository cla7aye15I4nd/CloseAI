import torch
import random
import pickle
import argparse
from collections import namedtuple
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

def make_env(env_id='PongNoFrameskip-v4'):
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    return env

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame-num', type=int, default=10000000)
    parser.add_argument('--load-model', action='store_true')
    return parser.parse_args()

def load_model():
    model = torch.load('model.h5')
    return model['steps'], model['memory'], model['policy_net'], model['target_net']

def store_model(steps, memory, policy_net, target_net):
    torch.save({
        'steps' : steps,
        'memory' : memory,
        'policy_net' : policy_net,
        'target_net' : target_net
    }, 'model.h5')

def test_model(env_id, policy_net):
    env = make_atari(env_id)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)
    env = wrap_pytorch(env)

    CHANNEL_NUM = 4
    _, height, width = env.reset().shape

    def initial_state(screen):
        state = torch.cat(tuple(torch.tensor(screen) for _ in range(CHANNEL_NUM)))
        return state.reshape(1, CHANNEL_NUM, height, width)

    def get_next_state(state, screen):
        next_state = torch.zeros(1, CHANNEL_NUM, height, width, dtype=torch.uint8)
        next_state[:, :CHANNEL_NUM-1, :, :] = state[:, 1:, :, :]
        next_state[0][CHANNEL_NUM-1] = torch.from_numpy(screen.reshape(height, width))
        return next_state

    class Agent(object):
        def __init__(self, policy_net):
            self.state = None
            self.policy_net = policy_net
    
        def act(self, state):
            if self.state is None:
                self.state = initial_state(state)
            else:
                self.state = get_next_state(self.state, state)
            return self.policy_net(self.state.to(device)).max(1)[1].item()

    state = env.reset()
    episode_reward = 0
    current_model = Agent(policy_net)

    same = set()
    for t in range(100000):
        action = current_model.act(state)

        same.add(action)
        env.render()
        next_state, reward, done, _ = env.step(action)
    
        state = next_state
        episode_reward += reward

        if t and t % 10000 == 0:
            print('t : {} reward : {}'.format(t, episode_reward))
        if done:
            print('EXIT NORMAL')
            break

    print('reward:', episode_reward)
    if len(same) == 1:
        print('SAME!!!!!')

    torch.save(policy_net, f'checkpoint/model_{episode_reward}.pth')
    env.close()
