


class GringottsController:

    def __init__(self, map_shape, harry_loc, initial_observations):
        self.map_shape = map_shape
        self.harry_loc = harry_loc
        self.turn = 0
        self.visited = [[-1 for _ in range(map_shape[1])] for _ in range(map_shape[0])]
        self.visited[harry_loc[0]][harry_loc[1]] = 0
        self.vaults = set()
        self.checked_vaults = set()
        self.dragons = set()
        self.dangerous_tiles = set()
        self.safe_tiles = set()
        self.prev_action = ('',)
        for obs in initial_observations:
            if obs[0] == 'vault':
                self.vaults.add(obs[1])
            elif obs[0] == 'dragon':
                self.dragons.add(obs[1])
            else:
                dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                for d in dirs:
                    if self.legal(harry_loc + d):
                        self.dangerous_tiles.add(harry_loc + d)

    def get_next_action(self, observations):
        self.turn += 1
        #print(self.prev_action)
        dirs = [(self.harry_loc[0] + d[0], self.harry_loc[1] + d[1]) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
        is_sulfur = False
        # update observations
        for obs in observations:
            if obs[0] == 'vault':
                if obs[1] not in self.checked_vaults:
                    self.vaults.add(obs[1])
            elif obs[0] == 'dragon':
                self.dragons.add(obs[1])
            elif obs[0] == 'sulfur':
                is_sulfur = True
                if self.prev_action[0] != 'destroy':
                    for d in dirs:
                        if self.legal(d) and d not in self.safe_tiles:
                            self.dangerous_tiles.add(d)
                else:
                    self.dangerous_tiles.discard(self.prev_action[1])
                    self.safe_tiles.add(self.prev_action[1])

        # updating safe tiles
        if not is_sulfur:
            for d in dirs:
                if self.legal(d):
                    self.dangerous_tiles.discard(d)
                    self.safe_tiles.add(d)

        # deciding next action
        # collect action
        if self.harry_loc in self.vaults:
            action = ('collect',)
            self.vaults.discard(self.harry_loc)
            self.checked_vaults.add(self.harry_loc)
            self.prev_action = action
            return action

        # move actions
        possible_dirs = set(dirs)
        illegal_dirs = set()
        for d in possible_dirs:
            if not self.legal(d) or d in self.dragons or self.visited[d[0]][d[1]] != -1:
                illegal_dirs.add(d)
        if len(illegal_dirs) == len(dirs):
            illegal_dirs.clear()
            for d in possible_dirs:
                if not self.legal(d) or d in self.dragons:
                    illegal_dirs.add(d)
        for d in illegal_dirs:
            possible_dirs.discard(d)

        for d in possible_dirs:
            if d not in self.dangerous_tiles and d in self.vaults:
                action = ('move', d)
                self.prev_action = action
                self.harry_loc = d
                self.visited[d[0]][d[1]] = self.turn
                return action
            elif d in self.vaults:
                action = ('destroy', d)
                self.prev_action = action
                self.dangerous_tiles.discard(d)
                self.safe_tiles.add(d)
                return action

        def unexplored_score(tile):
            return sum(1 for neighbor in [tuple(map(sum, zip(tile, d))) for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]]
                       if self.legal(neighbor) and self.visited[neighbor[0]][neighbor[1]] == -1
                       and neighbor not in self.dragons and neighbor not in self.safe_tiles)

        best_move = None
        max_score = -1
        for d in possible_dirs:
            if d not in self.dangerous_tiles:
                score = unexplored_score(d)
                if score > max_score:
                    max_score = score
                    best_move = d
                elif score == max_score:
                    if self.visited[d[0]][d[1]] < self.visited[best_move[0]][best_move[1]]:
                        best_move = d


        if best_move:
            action = ('move', best_move)
            self.prev_action = action
            self.visited[best_move[0]][best_move[1]] = self.turn
            self.harry_loc = best_move
            return action

        # Destroy to clear path if no safe move
        if is_sulfur:
            best_move = None
            max_score = -1
            for d in possible_dirs:
                score = unexplored_score(d)
                if score > max_score:
                    max_score = score
                    best_move = d


            action = ('destroy', best_move)
            self.prev_action = action
            self.dangerous_tiles.discard(best_move)
            self.safe_tiles.add(best_move)
            return action

        #new addition:
        action = ('wait',)
        self.prev_action = action
        return action

        # consequences of chosen action

    def legal(self, loc):
        if 0 <= loc[0] < self.map_shape[0] and 0 <= loc[1] < self.map_shape[1]:
            return True
        return False

