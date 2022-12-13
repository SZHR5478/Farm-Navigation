import math
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_,zeros_
import json
import logging


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, device, device_share):
    diff_device = device!=device_share
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None and not diff_device:
            return
        elif not diff_device:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.to(device_share)


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        xavier_uniform_(module.weight.data)  #不包含两边界值
        zeros_(module.bias.data)

def weights_init_mlp(module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1:
        module.weight.data.normal_(0, 1)            #标准正太分布
        module.weight.data *= 1 / \
            torch.sqrt(module.weight.data.pow(2).sum(1, keepdim=True))
        if module.bias is not None:
            zeros_(module.bias.data)

def normal(x, mu, sigma, device):
    pi = np.array([math.pi])
    pi = torch.from_numpy(pi).float()
    pi = Variable(pi).to(device)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b

def check_path(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)
