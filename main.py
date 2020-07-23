import os
import time
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

MODE = 'train' # or 'test'
# hyperparameter
TAU = 0.01 # soft replacement
LR_A = 1e-3 # learning rate for actor
LR_C = 2e-3 # learning rate for critic
GAMMA = 0.9 # reward discount
MEMORY_CAPACITY = 2000 # the capacity of replay buffer (~1000)
BATCH_SIZE = 32 # sample batch size (~32)
EXPLORATION_NOISE = 0.1 # noise amplitude for exploration
UPDATE_ITERATION = 10 # iteration to update model parameter
# normal distribution
SIGMA = np.array(((1.0, 0.0), (0.0, 1.0)), dtype=float)
SIGMA_INV = np.linalg.inv(SIGMA)
SIGMA_DET = np.linalg.det(SIGMA)
# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, dropout=0):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def resnet18(num_classes, dropout=0):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, dropout)

class Replay_buffer(object):

    def __init__(self, max_size=MEMORY_CAPACITY):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr+1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(X)
            y.append(Y)
            u.append(U)
            r.append(R)
            d.append(D)
        return np.array(x), np.array(y), np.array(u), np.array(r), np.array(d)

class Actor(nn.Module):

    def __init__(self, action_dim, action_range, action_offset):
        super(Actor, self).__init__()
        self.resnet = resnet18(num_classes=action_dim)
        self.action_range = action_range
        self.action_offset = action_offset

    def forward(self, x):
        out = torch.tanh(self.resnet(x))
        out = self.action_offset + self.action_range * out
        return out

class Critic(nn.Module):

    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.resnet = resnet18(num_classes=1)

    def forward(self, x, u):
        out = self.resnet(x)
        return out

class DDPG(object):

    def __init__(self, action_dim, action_range, action_offset, dir_name, order):
        super(DDPG, self).__init__()
        self.dir_name = dir_name
        self.order = order
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        self.actor = Actor(action_dim, action_range, action_offset).to(DEVICE)
        self.actor_target = Actor(action_dim, action_range, action_offset).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LR_A)

        self.critic = Critic(action_dim).to(DEVICE)
        self.critic_target = Critic(action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LR_C)

        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(dir_name)
        self.num_episode = 0
        self.num_training = 0
        self.num_actor_update_iteration = 0
        self.num_critic_update_iteration = 0

    def turn(self):
        self.order = 'sente' if self.order == 'gote' else 'gote'

    def getMatrix(self, s_state):
        matrix = np.zeros((1, 32, 32), dtype=float)
        sign = 1 if self.order == 'sente' else -1
        for i in range(16):
            px, py = s_state[0][2*i], s_state[0][2*i+1]
            if px == 0 and py == 0:
                continue
            mu = np.array((px, py))
            for x in range(32):
                for y in range(32):
                    tx, ty = x / 31 * 4.75, y / 31 * 7.6 + 2.9
                    cor = np.array((tx, ty))
                    matrix[0][x][y] += sign*np.exp((-1/2)*(cor-mu).T.dot(SIGMA_INV).dot(cor-mu))/np.sqrt(np.power(2*np.pi, 2)*SIGMA_DET)
            sign = -1 * sign
        return matrix

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        self.num_training += 1
        for it in range(UPDATE_ITERATION):
            # sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(BATCH_SIZE)
            state = torch.FloatTensor(x).to(DEVICE)
            next_state = torch.FloatTensor(y).to(DEVICE)
            action = torch.FloatTensor(u).to(DEVICE)
            reward = torch.FloatTensor(r).to(DEVICE)
            done = torch.FloatTensor(d).to(DEVICE)

            # compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * GAMMA * target_Q).detach()

            # get current Q estimate
            current_Q = self.critic(state, action)

            # compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            print('critic loss: {:.4f}'.format(critic_loss.item()))
            self.writer.add_scalar('Loss/{}/critic_loss'.format(self.order), critic_loss, global_step=self.num_critic_update_iteration)

            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute actor loss
            actor_loss = -1 * self.critic(state, self.actor(state)).mean()
            print('actor loss: {:.4f}'.format(actor_loss.item()))
            self.writer.add_scalar('Loss/{}/actor_loss'.format(self.order), actor_loss, global_step=self.num_actor_update_iteration)

            # optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def initiate(self):
        if not os.path.exists(os.path.join(self.dir_name, '{}.pth'.format(self.order))):
            self.save()
        self.turn()
        if not os.path.exists(os.path.join(self.dir_name, '{}.pth'.format(self.order))):
            self.save()
        self.turn()

    def save(self):
        state_dict = {
            'replay_buffer': self.replay_buffer,
            'num_episode': self.num_episode,
            'num_training': self.num_training,
            'num_actor_update_iteration': self.num_actor_update_iteration,
            'num_critic_update_iteration': self.num_critic_update_iteration,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict()
        }
        fpath = os.path.join(self.dir_name, '{}.pth'.format(self.order))
        torch.save(state_dict, fpath)
        print('model has been saved as {}'.format(fpath))

    def load(self):
        fpath = os.path.join(self.dir_name, '{}.pth'.format(self.order))
        state_dict = torch.load(fpath, map_location=torch.device(DEVICE))
        self.replay_buffer = state_dict['replay_buffer']
        self.num_episode = state_dict['num_episode']
        self.num_training = state_dict['num_training']
        self.num_actor_update_iteration = state_dict['num_actor_update_iteration']
        self.num_critic_update_iteration = state_dict['num_critic_update_iteration']
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_opt'])
        self.critic_optimizer.load_state_dict(state_dict['critic_opt'])
        print('model has been loaded from {}'.format(fpath))

def main():
    print('current device: {}'.format(DEVICE))
    # connection
    host = '127.0.0.1'
    port = 7788
    obj = socket.socket()
    obj.connect((host, port))
    # initialization
    s_state = None
    state = None
    order = None
    action = None
    dir_name = None
    ep_r = 0
    score = 0
    # phase
    is_start = False
    is_recv = False
    is_go = False
    has_action = False
    done = False

    while True:
        ret = str(obj.recv(1024), encoding='utf-8')
        if ret != '':
            print('recv:{}'.format(ret))
        messageList = ret.split(' ')

        if messageList[0] == 'NAME':
            name = messageList[1]
            if name == 'Player1':
                dir_name = 'player1'
                order = 'sente'
                is_start = True
            elif name == 'Player2':
                dir_name = 'player2'
                order = 'gote'
                is_start = True
            else:
                raise Exception

        if is_start:
            is_start = False
            action_dim = 3
            action_range = torch.FloatTensor(np.array([2.0, 2.2, 10.0]).reshape(1, -1)).to(DEVICE)
            action_offset = torch.FloatTensor(np.array([3.0, 0.0, 0.0]).reshape(1, -1)).to(DEVICE)
            agent = DDPG(action_dim, action_range, action_offset, dir_name, order)
            agent.initiate()
            agent.load()

        if messageList[0] == 'ISREADY':
            time.sleep(0.5)
            obj.send(bytes('READYOK', encoding='utf-8'))
            print('send:READYOK')
            obj.send(bytes('NAME hulvbobing', encoding='utf-8'))
            print('send:NAME hulvbobing')

        if messageList[0] == 'POSITION':
            s_state = [list(map(float, messageList[1:33]))]
        else:
            for i in range(1, len(messageList)):
                if messageList[i] == '\x00POSITION':
                    s_state = [list(map(float, messageList[i+1:i+33]))]

        if messageList[0] == 'SETSTATE':
            s_state.append(list(map(int, messageList[1:5])))
            is_recv = True
        else:
            for i in range(1, len(messageList)):
                if messageList[i] == '\x00SETSTATE':
                    s_state.append(list(map(int, messageList[i+1:i+5])))
                    is_recv = True

        if is_recv:
            is_recv = False
            if has_action or done:
                next_state = agent.getMatrix(s_state)

                if has_action:
                    has_action = False
                    px, py = s_state[0][s_state[1][0]*2-2], s_state[0][s_state[1][0]*2-1]
                    if px == 0 and py == 0:
                        reward = -100.0
                    else:
                        reward = -1.0 * ((px-2.375)**2 + (py-4.88)**2)
                    ep_r += reward
                    agent.replay_buffer.push((state, next_state, action, reward, 0))

                if done:
                    reward = 1000.0 * score
                    ep_r += reward
                    agent.replay_buffer.push((state, next_state, action, reward, 1))

                if len(agent.replay_buffer.storage) >= BATCH_SIZE:
                    agent.update()

                if done:
                    done = False
                    print('episode {:d}, reward: {:.4f}'.format(agent.num_episode, ep_r))
                    agent.writer.add_scalar('Reward/{}'.format(agent.order), ep_r, global_step=agent.num_episode)
                    ep_r = 0
                    agent.num_episode += 1
                    agent.save()
                    if score != 0:
                        next_order = 'sente' if score > 0 else 'gote'
                        if agent.order != next_order:
                            agent.turn()
                            agent.load()

        if messageList[0] == 'GO':
            is_go = True
        else:
            for i in range(1, len(messageList)):
                if messageList[i] == '\x00GO':
                    is_go = True

        if is_go:
            is_go = False
            has_action = True
            state = agent.getMatrix(s_state)

            action = agent.select_action(state)
            action = (action + np.array([2.0, 2.0, 2.0]) * np.random.normal(0, EXPLORATION_NOISE, size=3))
            action = np.clip(action, np.array([2.4, -2.2, -10]), np.array([8.0, 2.2, 10]))
            shot = 'BESTSHOT {}'.format(str(list(action))[1:-1].replace(',', ''))
            print('send:{}'.format(shot))
            obj.send(bytes(shot, encoding='utf-8'))

        if messageList[0] == 'SCORE':
            score = int(messageList[1])
            done = True
        else:
            for i in range(1, len(messageList)):
                if messageList[i] == '\x00SCORE':
                    score = int(messageList[i+1])
                    done = True

if __name__ == '__main__':
    main()
