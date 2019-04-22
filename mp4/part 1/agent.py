import numpy as np
import utils
import random
import math

SEGMENT_SIZE = 40
BOARD_SIZE = 12

# One if matches one_key, two if two_key, zero else
def map_idx(val, one_key, two_key):
    if val == one_key:
        return 1
    elif val == two_key:
        return 2
    else:
        return 0

# One if val greater that decision, two if val less than decision, zero else
def map_range_idx(val, decision):
    if val < decision:
        return 1
    elif val > decision:
        return 2
    else:
        return 0


class Agent:

    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        self.reset()

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None

    def convert_state(self, state):
        snake_head_x = state[0]
        snake_head_y = state[1]
        snake_body = state[2]
        food_x = state[3]
        food_y = state[4]

        # Checking adjoining walls
        adjoining_wall_x = self.map_idx(snake_head_x, SEGMENT_SIZE, BOARD_SIZE * SEGMENT_SIZE)
        adjoining_wall_y = self.map_idx(snake_head_y, SEGMENT_SIZE, BOARD_SIZE * SEGMENT_SIZE)

        # Checking food direction
        food_dir_x = map_range_idx(food_x, snake_head_x)
        food_dir_y = map_range_idx(food_y, snake_head_y)

        # Checking adjoining body segments
        adjoining_body_top = 0
        adjoining_body_bottom = 0
        adjoining_body_left = 0
        adjoining_body_right = 0
        for i in range(len(snake_body)):
            body_x = snake_body[i][0]
            body_y = snake_body[i][1]

            if snake_head_x == body_x:
                if body_y == snake_head_y - SEGMENT_SIZE:
                    adjoining_body_top = 1
                elif body_y == snake_head_y + SEGMENT_SIZE:
                    adjoining_body_bottom = 1
            elif snake_head_y == body_y:
                if body_x == snake_head_x - SEGMENT_SIZE:
                    adjoining_body_left = 1
                elif body_x = snake_head_x + SEGMENT_SIZE:
                    adjoining_body_right = 1

        # Concatenating all values and returning tuple
        new_state = (adjoining_wall_x, adjoining_wall_y)
        new_state += (food_dir_x, food_dir_y)
        new_state += (adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)
        return new_state

    # Access and modification functions for our Q table
    def Q(self, s, a):
        return self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a]

    def update_Q(self, state, a, new_q):
        self.Q[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a] = new_q

    # Returns best action and corresponding q value in current state
    def best_move(self, s):
        max_a = 0
        max_q = -math.inf
        for i in range(len(self.actions)):
            curr_q = self.Q(s, self.actions[i])

            if curr_q >= max_q:
                max_q = curr_q
                max_a = self.actions[i]

        return max_a, max_q

    # Access and modification functions for our N function
    def N(self, s, a):
        return self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a]

    def increment_N(self, state, a):
        self.N[s[0]][s[1]][s[2]][s[3]][s[4]][s[5]][s[6]][s[7]][a] += 1

    # Alpha value in update Q function
    def get_alpha():
        return self.C / (self.C + self.N(self.s, self.a))

    # Our rewards function
    def R(self, s, points, dead):
        if dead:
            return -1
        elif self.points < points:
            return 1
        else:
            return -0.1

    def R_plus():
        return 1

    # Our exploration function
    def f(self, u, n):
        if n < self.Ne:
            return R_plus()
        else:
            return u


    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''

        if self._train:

            if self.s != None and self.a != None
                # Updating our Q value
                alpha = get_alpha()
                R_s = self.R(self.s, points, dead)
                max_curr_q, max_curr_a = self.best_move(state)
                new_q = (1 - alpha) * self.Q(self.s, self.a) + alpha * (R_s + self.gamma * max_curr_q)
                update_Q(self.s, self.a, new_q)

            if dead:
                self.reset()
                return
            else:
                # Calculating argmax of exploration policy
                max_a = 0
                max_f = -math.inf
                for i in range(len(self.actions)):
                    curr_a = self.actions[i]
                    curr_f = self.f(self.Q(state, a), self.N(state, curr_a))

                    if curr_f >= max_f:
                        max_f = curr_f
                        max_a = curr_a

                # Updating variables to keep track of last iter data
                self.s = state
                self.points = points
                self.a = max_a

        else:
            max_a = self.best_move(state)

        return max_a
