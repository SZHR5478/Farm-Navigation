import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init, weights_init_mlp, normal
import perception
import numpy as np
import math


def build_model(obs_space, action_space, args, device):
    model = A3CModel(obs_space.shape, action_space.shape, args.rnn_out, args.network, device)
    model.train()
    return model


'''将模型action映射到真正的控制指令范围内'''
def wrap_action(self, action):
    #输入action应该是一个介于[-1,1]之间的值
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low)/2.0 + (self.action_high + self.action_low)/2.0
    return out


def sample_action(mu_multi, sigma_multi, device):
    mu = torch.clamp(mu_multi, -1.0, 1.0)
    sigma = F.softplus(sigma_multi) + 1e-5
    eps = torch.randn(mu.size())
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    eps = Variable(eps).to(device)
    pi = Variable(pi).to(device)
    action = (mu + sigma.sqrt() * eps).data
    act = Variable(action)
    prob = normal(act, mu, sigma, device)
    action = torch.clamp(action, -1.0, 1.0)
    entropy = 0.5 * ((sigma * 2 * pi.expand_as(sigma)).log() + 1)  # 0.5 * (log(2*pi*sigma) + 1
    log_prob = (prob + 1e-6).log()
    action_env = action.cpu().numpy()
    return action_env, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.critic_linear = nn.Linear(input_dim, 1)
        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value


class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        if 'continuous' in head_name:
            num_outputs = action_space.shape[0]
            self.continuous = True
        else:
            num_outputs = action_space.n
            self.continuous = False

        self.actor_linear = nn.Linear(input_dim, num_outputs)
        if self.continuous:
            self.actor_linear2 = nn.Linear(input_dim, num_outputs)

        # init layers
        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        if self.continuous:
            self.actor_linear2.weight.data = norm_col_init(self.actor_linear2.weight.data, 0.01)
            self.actor_linear2.bias.data.fill_(0)

    def forward(self, x, test=False):
        if self.continuous:
            mu = F.softsign(self.actor_linear(x))
            sigma = self.actor_linear2(x)                #sigma 是方差不是标准差
        else:
            mu = self.actor_linear(x)
            sigma = torch.ones_like(mu)

        action, entropy, log_prob = sample_action(self.continuous, mu, sigma, self.device, test)
        return action, entropy, log_prob

class A3CModel(torch.nn.Module):
    def __init__(self, obs_space, action_space, rnn_out=128, head_name='cnn_lstm', device=None):
        super(A3CModel, self).__init__()

        if 'cnn' in head_name:
            self.encoder = perception.CNN_simple(obs_space)
        feature_dim = self.encoder.outdim

        if 'lstm' in head_name:
            self.lstm = nn.LSTM(feature_dim, rnn_out,batch_first =True)
            feature_dim = rnn_out

        #  create actor
        self.actor = PolicyNet(feature_dim, action_space, head_name, device)
        self.critic = ValueNet(feature_dim)

        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        feature = self.encoder(inputs)
        if 'lstm' in self.head_name:
            hx, cx = self.lstm(feature)
            feature = hx
        if 'gru' in self.head_name:
            hx = self.lstm(feature, hx)
            feature = hx
        value = self.critic(feature)
        action, entropy, log_prob = self.actor(feature, test)

        return value, action, entropy, log_prob, (hx, cx)