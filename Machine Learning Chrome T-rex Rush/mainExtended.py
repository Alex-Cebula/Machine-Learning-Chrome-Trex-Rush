import os
import pygame
import argparse
from pygame.version import PygameVersion
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
import datetime
import distutils.util
import random
from random import randint
import numpy as np
import pandas as pd
from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

def define_parameters():
    """
    Define various settings of neural network and game
    """
    params = dict()
    # Neural Network
    params['epsilon_decay_linear'] = 1/11
    params['learning_rate'] = 0.00013629
    params['first_layer_size'] = 3 #200    # neurons in the first layer
    params['second_layer_size'] = 20   # neurons in the second layer
    params['third_layer_size'] = 50    # neurons in the third layer
    params['memory_size'] = 2500
    params['batch_size'] = 1000
    params['input#'] = 2
    params['output#'] = 2
    # Settings
    params['train'] = True
    params["game_speed"] = 4
    params['weights_path'] = 'weights/weights.h5'
    params['log_path'] = 'logs/scores_' + str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")) +'.txt'
    return params

def execute_move(playerDino, move, height, jump_sound):
    """
    Execute move based on neural net's action
    """                
    if np.array_equal(move, [0, 1]): #Jump
        if playerDino.rect.bottom == int(0.98*height):
            playerDino.isJumping = True
            if pygame.mixer.get_init() != None:
                jump_sound.play()
            if playerDino.isInAir == False:
                playerDino.initiatedJump = True
            else:
                playerDino.initiatedJump = False
            playerDino.movement[1] = -1*playerDino.jumpSpeed  

class Agent(torch.nn.Module):
    def __init__(self, params):
        """
        Constructor
        """
        super().__init__()
        self.reward = 0
        self.gamma = 0.9
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']        
        self.epsilon = 1
        self.epsilon_decay = params['epsilon_decay_linear']
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.input_count = params['input#']
        self.output_count = params['output#']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.optimizer = None
        self.network()

    def network(self):
        """
        Define neural network
        """
        # Layers
        self.f1 = nn.Linear(self.input_count, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.output_count)
        #self.f2 = nn.Linear(self.first_layer, self.second_layer)
        #self.f3 = nn.Linear(self.second_layer, self.third_layer)
        #self.f4 = nn.Linear(self.third_layer, self.output_count)
        # weights
        #self.model = self.load_state_dict(torch.load(self.weights))
        #print("weights loaded")

    def get_state(self, playerDino, cacti, gameSpeed):
        """
        Returns the current envrionment state in an bool bit array
        """
        jumpDuration = 28
        jumpDistance = gameSpeed*jumpDuration
        jumpZone = jumpDistance - playerDino.rect.width

        jumpDanger = False
        horizDanger = False 
        #isInAir = playerDino.isInAir
        
        for c in cacti:
            cXOffset1 = c.rect.right - playerDino.rect.right #X distance between cactus and player
            cXOffset2 = c.rect.left - playerDino.rect.left
            # Booleans
            cR_gte_pR = c.rect.right - jumpDistance > playerDino.rect.right
            cL_lse_pR = c.rect.left - jumpDistance < playerDino.rect.right
            cR_gte_pL = c.rect.right - jumpDistance > playerDino.rect.left
            cL_lse_pL = c.rect.left - jumpDistance < playerDino.rect.left
            cR_lse_pR = c.rect.right - jumpDistance < playerDino.rect.right
            cL_gte_pL = c.rect.left - jumpDistance > playerDino.rect.left

            if (cR_gte_pR and cL_lse_pR) or (cR_gte_pL and cL_lse_pL) or (cR_lse_pR and cL_gte_pL):
                jumpDanger = True
            if cXOffset1 < jumpZone + playerDino.rect.width/2 and not jumpDanger and cXOffset1 > 0:
                horizDanger = True
        
        state = [
            horizDanger, #danger infront
            jumpDanger#, #landing danger if we jump
            #isInAir
        ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, gameOver, old_move, old_state, airBorn, count):
        """
        Defines and returns reward
        """
        self.reward = 0 # base reward
        if gameOver:
            if not airBorn:
                self.reward = -10
        elif np.array_equal(old_move, [0, 1]): #jump
            if old_state[0] == 1: #horiz danger
                self.reward = 10
            else:
                if old_state[1] == 1: #jump danger
                    self.reward = -10
                #else:
                #    self.reward -= 1
        else:
            if old_state[0] == 1:
                self.reward = -1
            elif old_state[1] == 1:
                self.reward = 10
            
           

        return self.reward

    def set_epsilon(self, counter):
        self.epsilon = 1 - (counter * self.epsilon_decay)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, self.input_count)), dtype=torch.float32).to('cpu')
        state_tensor = torch.tensor(state.reshape((1, self.input_count)), dtype=torch.float32, requires_grad=True).to('cpu')
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to('cpu')
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to('cpu')
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()
    
    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.softmax(self.f2(x), dim=-1)
        #x = F.relu(self.f2(x))
        #x = F.relu(self.f3(x))
        #x = F.softmax(self.f4(x), dim=-1)
        return x