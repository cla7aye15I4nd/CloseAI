import torch

config = {
    'PongNoFrameskip-v4'      : ('models/pong.pth'     , 10 ** 5),
    'FreewayNoFrameskip-v4'   : ('models/freeway.pth'  , 10 ** 5),
    'KrullNoFrameskip-v4'     : ('models/krull.pth'    , 4 * 10 ** 5),
    'AtlantisNoFrameskip-v4'  : ('models/atlantis.pth' , 4 * 10 ** 5),
    'TutankhamNoFrameskip-v4' : ('models/tutankham.pth', 10 ** 5)
}

def initial_state(screen):
    CHANNEL_NUM, height, width = 4, 84, 84
    state = torch.cat(tuple(torch.tensor(screen) for _ in range(CHANNEL_NUM)))
    return state.reshape(1, CHANNEL_NUM, height, width)

def get_next_state(state, screen):
    CHANNEL_NUM, height, width = 4, 84, 84
    next_state = torch.zeros(1, CHANNEL_NUM, height, width, dtype=torch.uint8)
    next_state[:, :CHANNEL_NUM-1, :, :] = state[:, 1:, :, :]
    next_state[0][CHANNEL_NUM-1] = torch.from_numpy(screen.reshape(height, width))
    return next_state

class Agent(object):
    def __init__(self, model_file):
        self.state = None
        self.policy_net = torch.load(model_file).to('cuda')
    
    def act(self, state):
        if self.state is None:
            self.state = initial_state(state)
        else:
            self.state = get_next_state(self.state, state)
            
        return self.policy_net(self.state.to('cuda')).max(1)[1].item()

def make_agent(env_id):
    model_file, frame_num = config[env_id]
    return Agent(model_file), frame_num
