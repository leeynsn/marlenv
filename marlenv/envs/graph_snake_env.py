from .snake_env import SnakeEnv
import numpy as np
import math


"""
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)
"""

class GraphSnakeEnv(SnakeEnv):
    def __init__(self, lattice=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lattice = lattice

    def _init_obs(self):

        obs = SnakeEnv._init_obs(self)
        proc_obs = self._process_obs(obs)

        return proc_obs

    def _get_obs(self):

        obs = SnakeEnv._get_obs(self)
        proc_obs = self._process_obs(obs)
        
        return proc_obs

    def _process_obs(self, obs):
        
        if not self.observer=='snake':
            raise ValueError("This is not yet implemented for 'human' observers.")
        if self.image_obs:
            raise ValueError("This is not yet implemented for 'image' observation.")

        if not self.lattice:
            vision_range = 5 # range of vision in default five
            if self.vision_range:
                vision_range = self.vision_range
            proc_obs = []
            sqrt2 = math.sqrt(2)
            snake_idx = 0
            for snake in self.snakes: # for each snake
                proc_ob = []
                angle = math.atan2(snake.direction.value[1], snake.direction.value[0])
                head = snake.head_coord
                if self.vision_range: # if so, the head is at the center
                    head = (self.vision_range, self.vision_range)
                for l in range(3): # for each of three directions except backward
                    dx = (int(math.cos(angle + self.action_dict[l])), int(math.sin(angle + self.action_dict[l])))
                    proc_ob.append(np.zeros((self.obs_ch, )))
                    for i in range(vision_range):
                        temp_ob = obs[snake_idx][head[0] + dx[0]*(i+1)][head[1] + dx[1]*(i+1)]
                        if temp_ob[0] == 1: # up to the wall
                            proc_ob[-1] += temp_ob / (i+1)
                            break
                        proc_ob[-1] += temp_ob / (i+1)
                for l in [(0, 1), (0, 2)]: # each of two diagonal directions
                    dx = [(int(math.cos(angle + self.action_dict[l[q]])), int(math.sin(angle + self.action_dict[l[q]]))) for q in range(2)]
                    proc_ob.append(np.zeros((self.obs_ch, )))
                    for i in range(vision_range):
                        temp_ob = obs[snake_idx][head[0] + (dx[0][0] + dx[1][0])*(i+1)][head[1] + (dx[0][1] + dx[1][1])*(i+1)]
                        if temp_ob[0] == 1:
                            proc_ob[-1] += temp_ob / ((i+1)*sqrt2)
                            break
                        proc_ob[-1] += temp_ob / ((i+1)*sqrt2)
                proc_obs.append(np.array(proc_ob))
                snake_idx += 1

            return np.array(proc_obs)

        else:
            vision_range = 5 # range of vision in default five
            if self.vision_range:
                vision_range = self.vision_range
            proc_obs = []
            snake_idx = 0
            for snake in self.snakes: # for each snake
                angle_number = int(math.atan2(snake.direction.value[1], snake.direction.value[0]) / (math.pi/2))
                proc_ob = np.rot90(obs[snake_idx], -angle_number, (0, 1))
                proc_obs.append(proc_ob)
                snake_idx += 1

            return np.array(proc_obs)