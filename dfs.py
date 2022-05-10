import numpy as np
from abc import ABC, abstractmethod
from collections import deque

class Solver(ABC):
    solution = None
    frontier = None
    max_depth = 0
    explored_nodes = set()
    initial_state = None

    def __init__(self, initial_state):
        self.initial_state = initial_state

    def ancestral_chain(self):
        current = self.solution
        chain = [current]
        while current.parent is not None:
            chain.append(current.parent)
            current = current.parent
        return chain

    @property
    def path(self):
        path = " ".join(node.operator for node in self.ancestral_chain()[-2::-1])
        count = 0
        for node in self.ancestral_chain()[-2::-1]:
            print(f"step {count}")
            print(node)
            count += 1
        return path

    @abstractmethod
    def solve(self):
        pass

    def set_solution(self, board):
        self.solution = board

class DFS(Solver):
    def __init__(self, initial_state, goal):
        super(DFS, self).__init__(initial_state)
        self.goal = goal
        self.frontier = []

    def solve(self):
        time_count = 0
        count = 0
        self.frontier.append(self.initial_state)
        while self.frontier:
            time_count += 1
            board = self.frontier.pop()
            self.explored_nodes.add(tuple(board.state))
            if board.goal_test(self.goal):
                self.set_solution(board)
                break
            for neighbor in board.neighbors()[::-1]:
                if tuple(neighbor.state) not in self.explored_nodes:
                    count += 1
                    self.frontier.append(neighbor)
                    self.explored_nodes.add(tuple(neighbor.state))
                    self.max_depth = max(self.max_depth, neighbor.depth)
        return [f"Time performance: {time_count} nodes popped off the queue", f"Space performance: {count} nodes in the queue at its max"]

class Board:
    parent = None
    state = None
    operator = None
    depth = 0
    zero = None
    cost = 0

    def __init__(self, state, parent=None, operator=None, depth=0):
        self.parent = parent
        self.state = np.array(state)
        self.operator = operator
        self.depth = depth
        self.zero = self.find_0()
        self.cost = self.depth + self.manhattan()

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        else:
            op_pr = {'Up': 0, 'Down': 1, 'Left': 2, 'Right': 3}
            return op_pr[self.operator] < op_pr[other.operator]

    def __str__(self):
        return str(self.state[:3]) + '\n' \
               + str(self.state[3:6]) + '\n' \
               + str(self.state[6:]) + str(self.operator) + '\n'

    def goal_test(self, goal):
        if np.array_equal(self.state, goal):
            return True
        else:
            return False

    def find_0(self):
        for i in range(9):
            if self.state[i] == 0:
                return i

    def manhattan(self):
        state = self.index(self.state)
        goal = self.index(np.arange(9))
        return sum((abs(state // 3 - goal // 3) + abs(state % 3 - goal % 3))[1:])

    @staticmethod
    def index(state):
        index = np.array(range(9))
        for x, y in enumerate(state):
            index[y] = x
        return index

    def swap(self, i, j):
        new_state = np.array(self.state)
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def up(self):
        if self.zero > 2:
            return Board(self.swap(self.zero, self.zero - 3), self, f'\naction= up, depth= {self.depth}, total_cost= {self.cost}', self.depth + 1)
        else:
            return None

    def down(self):
        if self.zero < 6:
            return Board(self.swap(self.zero, self.zero + 3), self, f'\naction= down, depth= {self.depth}, total_cost= {self.cost}', self.depth + 1)
        else:
            return None

    def left(self):
        if self.zero % 3 != 0:
            return Board(self.swap(self.zero, self.zero - 1), self, f'\naction= left, depth= {self.depth}, total_cost= {self.cost}', self.depth + 1)
        else:
            return None

    def right(self):
        if (self.zero + 1) % 3 != 0:
            return Board(self.swap(self.zero, self.zero + 1), self, f'\naction= right, depth= {self.depth}, total_cost= {self.cost}', self.depth + 1)
        else:
            return None

    def neighbors(self):
        neighbors = [self.up(), self.down(), self.left(), self.right()]
        return list(filter(None, neighbors))

    __repr__ = __str__

def main():
    start = np.array([1,3,4,8,6,2,0,7,5])
    goal = np.array([1,2,3,8,0,4,7,6,5])
    p1 = Board(start)
    s1 = DFS(p1, goal)
    result = s1.solve()
    print("Start")
    print(start.reshape(3,3))
    s1.path
    print("End")
    print(goal.reshape(3,3))

    print(result[0])
    print(result[1])
    print("Finished")

if __name__ == "__main__":
    main()
