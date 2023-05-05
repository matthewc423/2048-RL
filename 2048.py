#%%
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import itertools
import logging
from six import StringIO
import sys

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import time

import matplotlib.pyplot as plt
from matplotlib import colors

onWeb = False
direction = {0: Keys.UP, 1: Keys.RIGHT, 2: Keys.DOWN, 3: Keys.LEFT}

if onWeb:
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(options=options)
    driver.get("https://play2048.co/")


    grid = driver.find_element(By.TAG_NAME, "body")
    driver.find_element(By.CLASS_NAME, "grid-container").click()

highesttiles = {}
highesttiles[0] = 0
for i in range(1, 17):
    highesttiles[2 ** i] = 0
scores = []
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class IllegalMove(Exception):
    pass

def stack(flat, layers=16):
  """Convert an [4, 4] representation into [4, 4, layers] with one layers for each value."""
  # representation is what each layer represents
  representation = 2 ** (np.arange(layers, dtype=int) + 1)

  # layered is the flat board repeated layers times
  layered = np.repeat(flat[:,:,np.newaxis], layers, axis=-1)

  # Now set the values in the board to 1 or zero depending whether they match representation.
  # Representation is broadcast across a number of axes
  layered = np.where(layered == representation, 1, 0)

  return layered.transpose(2,0,1)

class Game2048Env(gym.Env):
    metadata = {'render.modes': ['ansi', 'human', 'rgb_array']}

    def __init__(self):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (layers, self.w, self.h), dtype=int)
        self.set_illegal_move_reward(-2.)
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70

        # Initialise seed
        self.seed()

        # Reset ready for a game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    def calculate_reward(self, next_state):
        def monotonicity(grid):
            def score_line(line):
                return sum([1 if line[i] >= line[i + 1] else 0 for i in range(len(line) - 1)])
            return sum([score_line(grid[i]) for i in range(4)]) + sum([score_line(grid[:, i]) for i in range(4)])

        def smoothness(grid):
            return -sum([abs(grid[i, j] - grid[i, j + 1]) for i in range(4) for j in range(3)] +
                        [abs(grid[i, j] - grid[i + 1, j]) for i in range(3) for j in range(4)])

        max_tile_reward = np.max(next_state)/2
        empty_cells_reward = np.sum(next_state == 0) * 2
        monotonicity_reward = monotonicity(next_state)/2
        smoothness_reward = smoothness(next_state)/4

        total_reward = max_tile_reward + empty_cells_reward + monotonicity_reward + smoothness_reward
        return total_reward

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        done = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            if onWeb:
                grid.send_keys(direction[action])
                time.sleep(0.15)
                self.add_new_tiles()
            else:
                self.add_tile()
            done = self.isend()
            reward = float(score)
            #reward = float(score) + float(self.calculate_reward(self.get_board()))

        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            done = True
            reward = self.illegal_move_reward

        


        #print("Am I done? {}".format(done))
        info['highest'] = self.highest()

        # Return observation (board state), reward, done and info dict
        state = stack(self.Matrix)
        if done:
            highesttiles[self.highest()] += 1  
            scores.append(self.score)
            if onWeb:
                time.sleep(1)
                retry = driver.find_element(By.CLASS_NAME, "retry-button")
                while not retry.is_displayed():
                    time.sleep(0.5)
                retry.click()
        return state, reward, done, info

    def reset(self):
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0
        if onWeb:
            self.add_new_tiles()

        else:
            logging.debug("Adding tiles")
            self.add_tile()
            self.add_tile() 
        state = stack(self.Matrix)    
        
        return state

    def render(self, mode='human'):
        if mode == 'rgb_array':
            black = (0, 0, 0)
            grey = (128, 128, 128)
            white = (255, 255, 255)
            tile_colour_map = {
                2: (255, 0, 0),
                4: (224, 32, 0),
                8: (192, 64, 0),
                16: (160, 96, 0),
                32: (128, 128, 0),
                64: (96, 160, 0),
                128: (64, 192, 0),
                256: (32, 224, 0),
                512: (0, 255, 0),
                1024: (0, 224, 32),
                2048: (0, 192, 64),
                4096: (0, 160, 96),
            }
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
            fnt = ImageFont.truetype('Arial.ttf', 30)

            for y in range(4):
              for x in range(4):
                 o = self.get(y, x)
                 if o:
                     draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size], tile_colour_map[o])
                     (text_x_size, text_y_size) = draw.textsize(str(o), font=fnt)
                     draw.text((x * grid_size + (grid_size - text_x_size) // 2, y * grid_size + (grid_size - text_y_size) // 2), str(o), font=fnt, fill=white)
                     assert text_x_size < grid_size
                     assert text_y_size < grid_size

            return np.asarray(pil_board).swapaxes(0, 1)

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_new_tiles(self):
        tiles = driver.find_elements(By.CLASS_NAME, "tile-new")
        for tile in tiles:
            classes = tile.get_attribute("class").split()
            tile_col = int(classes[2][14:15]) - 1
            tile_row = int(classes[2][16:17]) - 1
            val = int(classes[1][5:])
            self.set(tile_row, tile_col, val)

    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        possible_tiles = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = self.np_random.choice(possible_tiles, 1, p=tile_probabilities)[0]
        empties = self.empties()
        assert empties.shape[0]
        empty_idx = self.np_random.choice(empties.shape[0])
        empty = empties[empty_idx]
        logging.debug("Adding %s at %s", val, (empty[0], empty[1]))
        self.set(empty[0], empty[1], val)

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        # Construct a range for extracting row/column into a list
        rx = list(range(self.w))
        ry = list(range(self.h))

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = [self.get(x, y) for x in rx]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for x in rx:
                            self.set(x, y, new[x])
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = [self.get(x, y) for y in ry]
                (new, ms) = self.shift(old, shift_direction)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        for y in ry:
                            self.set(x, y, new[y])
        if changed != True:
            raise IllegalMove

        return move_score

    def combine(self, shifted_row):
        """Combine same tiles when moving to one side. This function always
           shifts towards the left. Also count the score of combined tiles."""
        move_score = 0
        combined_row = [0] * self.size
        skip = False
        output_index = 0
        for p in pairwise(shifted_row):
            if skip:
                skip = False
                continue
            combined_row[output_index] = p[0]
            if p[0] == p[1]:
                combined_row[output_index] += p[1]
                move_score += p[0] + p[1]
                # Skip the next thing in the list.
                skip = True
            output_index += 1
        if shifted_row and not skip:
            combined_row[output_index] = shifted_row[-1]

        return (combined_row, move_score)

    def shift(self, row, direction):
        """Shift one row left (direction == 0) or right (direction == 1), combining if required."""
        length = len(row)
        assert length == self.size
        assert direction == 0 or direction == 1

        # Shift all non-zero digits up
        shifted_row = [i for i in row if i != 0]

        # Reverse list to handle shifting to the right
        if direction:
            shifted_row.reverse()

        (combined_row, move_score) = self.combine(shifted_row)

        # Reverse list to handle shifting to the right
        if direction:
            combined_row.reverse()

        assert len(combined_row) == self.size
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        # if self.max_tile is not None and self.highest() == self.max_tile:
        #     return True

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board

env = Game2048Env()


#%%
model = DQN("MlpPolicy", env, verbose=1,tensorboard_log="runs")
model.learn(1E7)
print(highesttiles)

#%%
model.save("dqn_mlp")
#%%
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),

        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float() 
             ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
# Batch 4 4  16 
# Batch channel H W 
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

class CNN3D2D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=(2, 3, 3), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=(2, 2, 2), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),

        )
        self.cnn2d = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None, None]).float() 
             ).shape[1]
            n_flatten2d = self.cnn2d(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten + n_flatten2d, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
         cnn3d = self.cnn(observations.unsqueeze(1))
         cnn2d = self.cnn2d(observations)
         return self.linear(th.concat((cnn3d, cnn2d), dim=1))
    
policy_kwargs3d2d = dict(
    features_extractor_class=CNN3D2D,
    features_extractor_kwargs=dict(features_dim=256),
)

class CNN3D(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size=(2, 2, 2), stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),

        )
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None, None]).float() 
             ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
         cnn3d = self.cnn(observations.unsqueeze(1))
         return self.linear(cnn3d)
    
policy_kwargs3d = dict(
    features_extractor_class=CNN3D,
    features_extractor_kwargs=dict(features_dim=256),
)
#%%
#import os
#venv = make_vec_env(Game2048Env)
# venv = VecTransposeImage(venv)
# env = Game2048Env()
model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs3d2d, verbose=1, tensorboard_log="runs", device="cuda")
model.learn(3E7, tb_log_name="3D2DCNN_mono")
model.policy.save("3D2DCNN_mono")
model.q_net.save("3D2DCNN_mono")
print(highesttiles)
# model.save("2048_dqn")
#%%
del model
venv = make_vec_env(Game2048Env)

model = DQN("CnnPolicy", venv, policy_kwargs=policy_kwargs, verbose=0, tensorboard_log="runs")

#%%
model.policy.load("2048_policy")
#%%
env = Game2048Env()
model = DQN.load("dqn_mlp", env=env)
#%%
from stable_baselines3.common.evaluation import evaluate_policy

evaluate_policy(model, env, n_eval_episodes=100)
print(highesttiles)
#%%
scores = []
highesttiles = {}
highesttiles[0] = 0
for i in range(1, 17):
    highesttiles[2 ** i] = 0
#%%

model.learn(2E7)
model.policy.save("2048_policy")
model.q_net.save("2048_qnet")
print(highesttiles)
#%%
import seaborn as sns
scores = [s for s in scores if s > 500]
#sns.kdeplot(scores)
sns.histplot(scores)