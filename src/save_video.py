from gym import wrappers

from agent import make_agent
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

def save_video(env_id):
    # create model
    current_model, num_frames = make_agent(env_id)
    print('Start test {} {} frames'.format(env_id, num_frames))
    
    env = make_atari(env_id)
    env = wrap_deepmind(env, episode_life=False, clip_rewards=False)
    env = wrap_pytorch(env)
    env = wrappers.Monitor(env, './videos/' + env_id + '/', force = True)
    
    all_rewards = []
    episode_reward = 0

    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        action = current_model.act(state)

        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            break
            
    env.close()
    
    # free agent memory
    del current_model

save_video('PongNoFrameskip-v4')
save_video('KrullNoFrameskip-v4')
save_video('TutankhamNoFrameskip-v4')
save_video('AtlantisNoFrameskip-v4')
save_video('FreewayNoFrameskip-v4')
