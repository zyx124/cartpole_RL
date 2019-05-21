#!/usr/bin/env python
import numpy as np
import os
import time
import rospy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import matplotlib.pyplot as plt

from collections import deque
from collections import namedtuple

from robot_sim.srv import RobotPolicy
from robot_sim.srv import RobotPolicyRequest
from robot_sim.srv import RobotPolicyResponse
from robot_sim.srv import RobotAction
from robot_sim.srv import RobotActionRequest
from robot_sim.srv import RobotActionResponse

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # saves a transitions [s_t, a_t, s_{t+1}, r_{t+1}]
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):

    def __init__(self, input_dim, action_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 84)
        self.fc2 = nn.Linear(84, 84)
        #self.fc3 = nn.Linear(48, 32)
        self.fc4 = nn.Linear(84, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
       # x = F.tanh(self.fc3(x))

        x = self.fc4(x)
        return x

    def predict(self, features):
        self.eval()
        features = torch.from_numpy(features).float()
        return self.forward(features).detach().numpy()


class MyDataset(Dataset):
    def __init__(self, labels, features):
        super(MyDataset, self).__init__()
        self.labels = labels
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self,
                    idx):  # This tells torch how to extract a single datapoint from a dataset, Torch randomized and needs a way to get the nth-datapoint
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}


class FakeRobot(object):

    def __init__(self):
        self.service = rospy.Service('cartpole_policy', RobotPolicy, self.callback)
        self.probe_service = rospy.ServiceProxy('cartpole_robot', RobotAction, persistent=True)

        # define the exploring and exploiting rate
        self.epsilon = 1.0
        self.epsilon_dacay = 0.995
        self.epsilon_min = 0.01

        # define gamma and learning rate
        self.gamma = 0.95
        self.learning_rate = 0.001

        # define pole range and cart range
        self.q_max = 3 * np.pi / 180
        self.x_max = 1.2

        # define the replay memory
        self.memory = deque(maxlen=1000)

        # define the network model
        self.input_dim = 4
        self.action_dim = 2
        self.model = Net(self.input_dim, self.action_dim)
        self.target_net = Net(self.input_dim, self.action_dim)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1080)
        global episode_durations
        episode_durations = []

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    def get_random_sign(self):
        return 1.0 if random.random() < 0.5 else -1.0

    def state_out_of_bounds(self, state):
        return abs(state[0]) > self.x_max or abs(state[1]) > self.q_max

    def remember(self, state, action, act_idx, reward, next_state, done):
        self.memory.append((state, action, act_idx, reward, next_state, done))

    def act(self, state):
        # exploration mode
        action = [-10, 10]
        if np.random.rand() < self.epsilon:
            act_idx = random.randrange(2)
            return action[act_idx], act_idx
        # exploiting mode
        state = np.asarray(state)
        act_values = self.model.predict(state)
        # print(act_values)
        idx = int(np.argmax(act_values))
        return action[idx], idx

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        features = []
        labels = []

        for state, action, act_idx, reward, next_state, done in minibatch:
            next_state = np.asarray(next_state)
            state = np.asarray(state)
            # print('state', state)
            # print('action', action)
            # print('reward', reward)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_net.predict(next_state))

            target_f = self.model.predict(state)

            # print(target_f)
            target_f[act_idx] = target
            labels.append(target_f)
            features.append(state)

        labels = np.asarray(labels)
        features = np.asarray(features)

        self.model.train()
        dataset = MyDataset(labels, features)
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        self.train_epoch(loader)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_dacay

    def train_epoch(self, loader):
        total_loss = 0.0
        for i, data in enumerate(loader):
            features = data['feature'].float()
            labels = data['label'].float()
            # print(features)
            # print(labels)
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)

            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        print('loss', total_loss)

    def callback(self, req):
        """
        RobotPolicy.srv:
            float64[] robot_state
            ---
            float64[] action

        :param req: req is the RobotPolicyRequest, 1D tuple or list
        :return: Response 2D list or tuple.
        """
        path = './m.pth'
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))


        action = [-10, 10]
        req = np.asarray(req.robot_state)
        idx = int(np.argmax(self.model.predict(req)))
        # print(action[idx])
        # idx = 0 if req[1]>0 else 1
        return RobotPolicyResponse([action[idx]])

    def server(self):
        s = rospy.Service('cartpole_policy', RobotPolicy, self.callback)

    def train(self):
        print('start training')
        start = time.time()
        done = False
        n_episodes = 800
        scores = deque(maxlen=20)
        batch_size = 64

        for e in range(n_episodes):
            # initial state
            # self.memory.clear()
            reward = 0

            # reset the robot for each episode
            req = RobotActionRequest()
            req.reset_robot = True
            # np.random.seed(0)
            req.reset_pole_angle = np.random.uniform(np.deg2rad(0.0), np.deg2rad(3.0))*self.get_random_sign()
            # req.reset_pole_angle = np.deg2rad(2.0) * self.get_random_sign()
            state = self.probe_service(req).robot_state
            # print(state)

            i = 0
            done = False
            for j in range(1000):
                action, act_inx = self.act(state)
                # print('action', action, act_inx)
                req = RobotActionRequest()
                req.reset_robot = False
                req.action = [action]
                next_state = self.probe_service(req).robot_state

                done = self.state_out_of_bounds(next_state)
                reward = 10 if not done else -10
                # print(state, action, act_inx, reward, next_state, done)
                self.remember(state, action, act_inx, reward, next_state, done)

                # print(state, action, reward, next_state, done)
                state = next_state
                i = i + 1

                # scores.append(i)
                # mean = np.mean(scores)
                if done:
                    episode_durations.append(i)
                    #self.plot_durations()
                    #print(('episode: {}/{}, score:{}, e:{:.2}').format(e, n_episodes, i, self.epsilon))
                    break

            if e>100:
                if np.mean(episode_durations[-30:])>250 or (time.time()-start)>=290:
                    self.plot_durations()
                    # torch.save(self.model.state_dict(), './x.pth')
                    return

            self.replay(batch_size)

            if e % 10 == 0:
                self.target_net.load_state_dict(self.model.state_dict())
        self.plot_durations()



if __name__ == '__main__':
    rospy.init_node('cartpole_policy', anonymous=True)
    p = FakeRobot()

    start = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    p.train()
    end = time.time()
    print('training_time=', end - start)

    rospy.spin()
