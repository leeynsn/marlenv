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

    def __add__(self, other):
        dx, dy = self.value
        return other[0] + dx, other[1] + dy

    def __rsub__(self, other):
        dx, dy = self.value
        return other[0] - dx, other[1] - dy


# class Segment:
#     def __init__(self, coord, direction):
#         self.coord = coord
#         self.direction = direction

class Snake:
    def __init__(self, idx, head_coord, direction: Direction = Direction.RIGHT):
        self.idx: int = idx
        self.head_coord: tuple = head_coord
        # left coord of head for now
        self.tail_coord: tuple = head_coord - direction
        self.direction: Direction = direction
        self.directions = deque([direction])

        self.alive = True
        self.fruit = False
        self.death = False
        self.reward = 0.

    def __len__(self):
        return len(self.directions + 1)

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
        self.fruit = False
        self.death = False

        return prev_tail_coord