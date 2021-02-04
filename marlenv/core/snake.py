from collections import deque
from enum import Enum


class Cell(Enum):
    EMPTY = 0
    FRUIT = 1
    WALL = 2
    TAIL = 3
    BODY = 4
    HEAD = 5


class Direction(Enum):
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)

    """
    little magic for convenience, coord + direction
    """

    def __radd__(self, other):
        dx, dy = self.value
        return other[0] + dx, other[1] + dy

    def __rsub__(self, other):
        dx, dy = self.value
        return other[0] - dx, other[1] - dy

    # def __add__(self, other):
    #     dx, dy = self.value
    #     return other[0] + dx, other[1] + dy


# class Segment:
#     def __init__(self, coord, direction):
#         self.coord = coord
#         self.direction = direction

class Snake:
    # def __init__(self, idx, head_coord, direction: Direction = Direction.RIGHT):
    #     self.idx: int = idx
    #     self.head_coord: tuple = head_coord
    #     # left coord of head for now
    #     self.tail_coord: tuple = head_coord - direction
    #     self.direction: Direction = direction
    #     self.directions = deque([direction])

    #     self.alive = True
    #     self.fruit = False
    #     self.death = False
    #     self.reward = 0.

    def __init__(self, idx, coords):
        assert len(coords) > 1
        self.idx: int = idx
        self.head_coord: tuple = coords[0]
        self.tail_coord: tuple = coords[-1]
        self.direction: Direction = Direction((coords[0][0] - coords[1][0],
                                              coords[0][1] - coords[1][1]))
        prev_coord = self.head_coord
        direction_list = []
        for next_coord in coords[1:]:
            direction_list.append(Direction((prev_coord[0] - next_coord[0],
                                             prev_coord[1] - next_coord[1])))
            prev_coord = next_coord

        self.directions = deque(direction_list)

        self.alive = True
        self._reset_reward_state()

    def __len__(self):
        return len(self.directions + 1)

    def _reset_reward_state(self):
        self.fruit = False
        self.death = False
        self.kills = 0
        self.win = False
        self.reward = 0.

    @property
    def coords(self):
        coord = self.head_coord
        coords = [coord]
        for direction in self.directions:
            coord -= direction
            coords.append(coord)

        return coords

    def move(self):
        self.head_coord += self.direction
        self.directions.appendleft(self.direction)

        prev_tail_coord = None
        if not self.fruit:
            prev_tail_coord = self.tail_coord
            tail_direction = self.directions.pop()
            self.tail_coord += tail_direction
        self._reset_reward_state()

        return prev_tail_coord
