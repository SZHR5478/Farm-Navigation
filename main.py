import argparse
import torch
from environment import create_env
from model import build_model

#构造参数解析器
parser = argparse.ArgumentParser(description='VPG')

#添加参数
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')        #学习率
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor for rewards (default: 0.9)')  #折扣因子
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')      #随机种子
parser.add_argument('--workers', type=int, default=1, metavar='W', help='how many training processes to use')      #运行进程数
parser.add_argument('--num-steps', type=int, default=20, metavar='NS', help='number of forward steps in VPG')
parser.add_argument('--test-eps', type=int, default=100, metavar='TE', help='maximum length of an episode')
parser.add_argument('--env', default='FarmNavigationWorld-v0', metavar='ENV', help='environment to train on')        #训练环境
parser.add_argument('--env-test', default='FarmNavigationWorld-v0', metavar='ENVB', help='environment to test on ')  #测试环境
parser.add_argument('--optimizer', default='Adam', metavar='OPT', help='shares optimizer choice of Adam or RMSprop')
parser.add_argument('--amsgrad', default=True, metavar='AM', help='Adam optimizer amsgrad parameter')
parser.add_argument('--load-model-dir', default=None, metavar='LMD', help='folder to load trained models from')
parser.add_argument('--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument('--network', default='cnn-lstm-continuous', metavar='M', help='config Network Architecture')     #骨干网络
parser.add_argument('--obs', default='img', metavar='O', help='img or vector')
parser.add_argument('--single', dest='single', action='store_true', help='run on single agent env') #dest 属性名


parser.add_argument('--gray', dest='gray', action='store_true', help='gray image')
parser.add_argument('--crop', dest='crop', action='store_true', help='crop image')
parser.add_argument('--inv', dest='inv', action='store_true', help='inverse image')
parser.add_argument('--rescale', dest='rescale', action='store_true', help='rescale image to [-1, 1]')   #
parser.add_argument('--input-size', type=int, default=80, metavar='IS', help='input image size')

parser.add_argument('--shared-optimizer', dest='shared_optimizer', action='store_true', help='use a shared optimizer')   #只要运行时有shared-optimizer变量，就将该变量设置为true
parser.add_argument('--stack-frames', type=int, default=1, metavar='SF', help='Choose number of observations to stack')
parser.add_argument('--rnn-out', type=int, default=128, metavar='LO', help='rnn output size')         #骨干网络输出
parser.add_argument('--sleep-time', type=int, default=0, metavar='ST', help='seconds to sleep after a process launched')    #
parser.add_argument('--max-step', type=int, default=150000, metavar='MS', help='max learning steps')

if __name__ == '__main__':

    #解析参数
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cpu')


    env = create_env(args)
    build_model(env.observation_space, env.action_space, args, device)






