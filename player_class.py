import pygame
import random
from settings import *


vec = pygame.math.Vector2


class Player():
    def __init__(self, app, pos):
        self.app = app
        self.starting_pos = [pos.x, pos.y]
        self.grid_pos = pos
        self.pix_pos = self.get_pix_pos()
        self.direction = vec(0, 0)
        self.speed = 4
        self.current_score = 0
        rid = random.randint(0, len(self.app.coins) - 1)
        self.target = self.app.coins[rid]
        target1 = (int(self.target.x), int(self.target.y))

        self.pathBFS = self.BFS([int(self.grid_pos.x), int(self.grid_pos.y)], [
            int(self.target[0]), int(self.target[1])])
        self.path = self.pathBFS.copy()
        diagram4 = GridWithWeights(COLS + 1, ROWS + 1)
        diagram4.walls = self.app.walls
        self.pathAStar = a_star_search(diagram4, (pos.x, pos.y), target1)
        self.pathGreed = greeeed(diagram4, (pos.x, pos.y), target1)

    def update(self):
        if self.app.coins != []:

            if self.target == self.grid_pos:
                self.current_score += 1
                self.app.coins.remove(self.target)
                if len(self.app.coins) > 0:
                    rid = random.randint(0, len(self.app.coins) - 1)
                    self.target = self.app.coins[rid]

            self.pathBFS = self.BFS([int(self.grid_pos.x), int(self.grid_pos.y)], [
                int(self.target[0]), int(self.target[1])])
            self.path = self.pathBFS.copy()
            self.pix_pos += self.direction * self.speed
            if self.time_to_move():
                self.move()
                if len(self.path) > 1:
                    self.path.remove(self.path[0])

            self.grid_pos[0] = (self.pix_pos[0] - TOP_BOTTOM_BUFFER +
                                self.app.cell_width // 2) // self.app.cell_width + 1
            self.grid_pos[1] = (self.pix_pos[1] - TOP_BOTTOM_BUFFER +
                                self.app.cell_height // 2) // self.app.cell_height + 1

            if self.on_coin():
                self.app.coins.remove(self.grid_pos)
                self.current_score += 1

    def on_coin(self):
        if self.grid_pos in self.app.coins:
            if int(self.pix_pos.x+TOP_BOTTOM_BUFFER//2) % self.app.cell_width == 0:
                if self.direction == vec(1, 0) or self.direction == vec(-1, 0):
                    return True
            if int(self.pix_pos.y+TOP_BOTTOM_BUFFER//2) % self.app.cell_height == 0:
                if self.direction == vec(0, 1) or self.direction == vec(0, -1):
                    return True
        return False

    def time_to_move(self):
        if int(self.pix_pos.x + TOP_BOTTOM_BUFFER // 2) % self.app.cell_width == 0:
            if self.direction == vec(1, 0) or self.direction == vec(-1, 0) or self.direction == vec(0, 0):
                return True
        if int(self.pix_pos.y + TOP_BOTTOM_BUFFER // 2) % self.app.cell_height == 0:
            if self.direction == vec(0, 1) or self.direction == vec(0, -1) or self.direction == vec(0, 0):
                return True
        return False

    def get_pix_pos(self):
        return vec((self.grid_pos.x * self.app.cell_width) + TOP_BOTTOM_BUFFER // 2 + self.app.cell_width // 2,
                   (self.grid_pos.y * self.app.cell_height) + TOP_BOTTOM_BUFFER // 2 +
                   self.app.cell_height // 2)

    def draw(self):
        pygame.draw.circle(self.app.screen, PLAYER_COLOUR, (int(self.pix_pos.x),
                                                            int(self.pix_pos.y)), self.app.cell_width // 2 - 2)

    def move(self):
        self.direction = self.get_path_direction(self.target)

    def get_path_direction(self, target):
        xdir = 0
        ydir = 0
        if len(self.path) > 1:
            next_cell = self.path[1]
            xdir = next_cell[0] - self.path[0][0]
            ydir = next_cell[1] - self.path[0][1]
        return vec(xdir, ydir)

    def BFS(self, start, target):
        grid = [[0 for x in range(COLS)] for x in range(ROWS)]
        for cell in self.app.walls:
            if cell.x < COLS and cell.y < ROWS:
                grid[int(cell.y)][int(cell.x)] = 1
        queue = [start]
        path = []
        visited = []
        while queue:
            current = queue[0]
            queue.remove(queue[0])
            visited.append(current)
            if current == target:
                break
            else:
                neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                for neighbour in neighbours:
                    if neighbour[0] + current[0] >= 0 and neighbour[0] + current[0] < len(grid[0]):
                        if neighbour[1] + current[1] >= 0 and neighbour[1] + current[1] < len(grid):
                            next_cell = [neighbour[0] + current[0], neighbour[1] + current[1]]
                            if next_cell not in visited:
                                if grid[next_cell[1]][next_cell[0]] != 1:
                                    queue.append(next_cell)
                                    path.append({"Current": current, "Next": next_cell})
        shortest = [target]
        while target != start:
            for step in path:
                if step["Next"] == target:
                    target = step["Current"]
                    shortest.insert(0, step["Current"])
        return shortest

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]

class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current

    return reconstruct_path(came_from, start, goal)

class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path

def greeeed(graph, start, goal):
    typik = False
    i = 1
    n = 0
    frontier = PriorityQueue()
    frontier.put(start, 0)
    path = [start]
    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            priority = heuristic(goal, next)
            frontier.put(next, priority)

        for k in range(0, i):
            if frontier.empty():
                typik = True
                break
            else:
                nextstep = frontier.get()

        if nextstep == goal:
            path.append(nextstep)
            break

        if nextstep in path and typik is True:
            n += 2
            nextstep = path[-n]
            path.append(nextstep)
            frontier = PriorityQueue()
            frontier.put(nextstep, 0)
            i = 1
            typik = False
        elif nextstep in path and typik is False:
            frontier = PriorityQueue()
            frontier.put(current, 0)
            i += 1
        else:
            n = 0
            path.append(nextstep)
            frontier = PriorityQueue()
            frontier.put(nextstep, 0)
            i = 1

    return path