import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, in_channels, h, w, n):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        
        # advantage
        self.linear4 = nn.Linear(linear_input_size, 512)
        self.linear5 = nn.Linear(512, n)

        # value
        self.linear6 = nn.Linear(linear_input_size, 512)
        self.linear7 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.float() / 255        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        advantage = F.relu(self.linear4(x.view(x.size(0), -1)))
        advantage = self.linear5(advantage)

        #return advantage
        value = F.relu(self.linear6(x.view(x.size(0), -1)))
        value = self.linear7(value)
        
        return value + advantage - advantage.mean(1, keepdim=True)
