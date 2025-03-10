import search
import random
import math
import itertools
from heapq import heappush, heappop



class HarryPotterProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""
    def __init__(self, initial):
        self.map = initial['map']
        self.initial_wizards = len(initial['wizards'])
        self.max_dist = len(self.map) * len(self.map[0]) + 2000
        self.de = initial['death_eaters']
        self.initialHorcruxes = initial['horcruxes']
        self.voldemort_pos = next((r, c) for r, row in enumerate(self.map)
                                  for c, cell in enumerate(row) if cell == 'V')

        wizards = tuple(
            (name, position[0], position[1], lives)
            for name, (position, lives) in initial['wizards'].items()
        )
        turn = 0
        horcruxes = tuple([True] * len(initial['horcruxes']))
        initial_state = (turn, wizards, horcruxes, False)

        self.wizard_to_horcrux = self._assign_wizards_to_horcruxes(wizards, horcruxes)

        search.Problem.__init__(self, initial_state)
    def _assign_wizards_to_horcruxes(self, wizards, horcruxes):
        """
        Assign each wizard to the nearest horcrux.
        """
        assignments = {}
        remaining_horcruxes = [self.initialHorcruxes[i] for i, intact in enumerate(horcruxes) if intact]

        for wizard in wizards:
            wizard_pos = (wizard[1], wizard[2])
            if remaining_horcruxes:
                closest_horcrux = min(
                    remaining_horcruxes,
                    key=lambda h: abs(wizard_pos[0] - h[0]) + abs(wizard_pos[1] - h[1])
                )
                assignments[wizard[0]] = closest_horcrux

        return assignments

    def _update_horcrux_assignments(self, state):
        """
        Recalculate the assignments when the state changes.
        """
        self.wizard_to_horcrux = self._assign_wizards_to_horcruxes(state[1], state[2])
    def actions(self, state):
        """Return the valid actions that can be executed in the given state."""
        res = []
        if not state[1]:
            return ()
        for wizard in state[1]:
            destroyed_hor = False
            atomic_actions = []
            x,y = wizard[1], wizard[2]
            # adding kill action
            if wizard[0] == 'Harry Potter' and sum(state[2]) == 0 and self.map[x][y] == 'V':
                atomic_actions.append(('kill', 'Harry Potter'))
                res.append(atomic_actions)
                continue
            # adding move actions
            dirs = [(1,0),(0,1),(-1,0),(0,-1)]
            for d in dirs:
                cur = (x+d[0], y+d[1])
                if 0<=cur[0]<len(self.map) and 0<=cur[1]<len(self.map[0]) and self.map[cur[0]][cur[1]] != 'I':
                    atomic_actions.append(('move', wizard[0], cur))

            #adding destroy actions
            for i, h in enumerate(self.initialHorcruxes):
                if h == (x, y) and state[2][i] == True:
                    atomic_actions.append(('destroy', wizard[0], i))
                    destroyed_hor = True
            #adding wait actions
            if not destroyed_hor:
                atomic_actions.append(('wait', wizard[0]))
            res.append(atomic_actions)

        #getting all possible actions by calculating all permutations of atomic actions
        action_combinations = itertools.product(*res)
        # Convert the result to a tuple of tuples
        return tuple(action_combinations)

    def result(self, state, action):
        """Return the state that results from executing the given action in the given state."""

        def calc_lives_lost(x, y):
            cnt = 0
            for place in de_placement:
                if (x, y) == place:
                    cnt += 1
            return cnt

        newWizards = []
        voldemort_is_dead = False
        new_h = list(state[2])
        de_placement = []
        result_turn = state[0] + 1
        game_over_state = (result_turn, (), (), False)

        # calculating death eaters placement based on the turn
        for path in self.de.values():
            # Determine the oscillation pattern:
            path_length = len(path)
            oscillation_position = result_turn % (2 * path_length - 2)  # Oscillation cycle length
            # If oscillation_position is less than the path length, we move forward
            if oscillation_position < path_length:
                current_position = path[oscillation_position]
            else:  # Otherwise, we move backward
                current_position = path[2 * path_length - 2 - oscillation_position]
            de_placement.append(current_position)

        for i, a in enumerate(action):
            wizard = state[1][i]
            # stepped on a death eater
            lives = wizard[3] - calc_lives_lost(wizard[1], wizard[2]) if a[0] != 'move' else wizard[
                                                                                                 3] - calc_lives_lost(
                a[2][0], a[2][1])
            if lives <= 0:
                return game_over_state
            if a[0] == 'move':
                # stepped on Voldemort
                if self.map[a[2][0]][a[2][1]] == 'V':
                    if sum(state[2]) != 0 or wizard[0] != 'Harry Potter':
                        return game_over_state
                newWizards.append((wizard[0], a[2][0], a[2][1], lives))

            elif a[0] == 'wait':
                newWizards.append((wizard[0], wizard[1], wizard[2], lives))

            elif a[0] == 'kill':
                voldemort_is_dead = True
                newWizards.append((wizard[0], wizard[1], wizard[2], lives))

            elif a[0] == 'destroy':
                new_h[a[2]] = False  # the horcrux in the a[2] index is destroyed
                newWizards.append((wizard[0], wizard[1], wizard[2], lives))

        next_state = (result_turn, tuple(newWizards), tuple(new_h), voldemort_is_dead)

        # Update wizard-to-horcrux assignments if necessary
        if sum(state[2]) - sum(next_state[2]) >= self.initial_wizards:
            self._update_horcrux_assignments(next_state)

        return next_state

    def goal_test(self, state):
        """Return True if the state is a goal state."""
        return state[3]

    def h(self, node):
        """
        Enhanced heuristic using precomputed wizard-to-horcrux assignments for efficiency.
        """
        state = node.state
        wizards = state[1]
        if state[3] and all(wizard[3] > 0 for wizard in wizards) and sum(node.state[2]) == 0:
            return 0

        if sum(state[2]) == 0:
            harry_pos = next(((wizard[1], wizard[2]) for wizard in wizards if wizard[0] == 'Harry Potter'), None)
            if harry_pos:
                status = 1 if state[3] else 0
                return abs(harry_pos[0] - self.voldemort_pos[0]) + abs(harry_pos[1] - self.voldemort_pos[1]) + status
            return self.max_dist

        total_distance = 0
        for wizard in wizards:
            wizard_name = wizard[0]
            if wizard_name in self.wizard_to_horcrux:
                horcrux = self.wizard_to_horcrux[wizard_name]
                wizard_pos = (wizard[1], wizard[2])
                total_distance += 1 - 1/((abs(wizard_pos[0] - horcrux[0]) + abs(wizard_pos[1] - horcrux[1])) + 1)

        danger_penalty = 0
        for path in self.de.values():
            for wizard in wizards:
                wizard_pos = (wizard[1], wizard[2])
                death_eater_pos = path[state[0] % len(path)]  # מיקום נוכחי של אוכל המוות
                if abs(wizard_pos[0] - death_eater_pos[0]) + abs(wizard_pos[1] - death_eater_pos[1]) < 1:
                    danger_penalty += 1  # עונש קטן על קרבה לאוכלי מוות

        harry_pos = next(((wizard[1], wizard[2]) for wizard in wizards if wizard[0] == 'Harry Potter'), None)
        harry_dist = abs(harry_pos[0] - self.voldemort_pos[0]) + abs(harry_pos[1] - self.voldemort_pos[1])

        return total_distance + sum(state[2])*1.6 + harry_dist * (1 / (sum(state[2])+1))

def create_harrypotter_problem(game):
    return HarryPotterProblem(game)