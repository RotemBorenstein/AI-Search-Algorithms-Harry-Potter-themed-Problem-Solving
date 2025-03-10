import itertools
import copy
import math
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from itertools import product



DESTROY_HORCRUX_REWARD = 2
RESET_REWARD = -2
DEATH_EATER_CATCH_REWARD = -1

class OptimalWizardAgent:
    def __init__(self, initial: Dict):
        self.initial = initial
        self.map = initial['map']
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.turns = initial['turns_to_go']
        self.wizards = initial['wizards']
        self.horcruxes = initial['horcrux']
        self.death_eaters = initial['death_eaters']
        self.memo = {}  # Memoization dictionary to store computed values
        self.policy_dict = {}
        del initial['map']
        del initial['optimal']
        del initial['turns_to_go']
        self.value(initial, self.turns)
        #print("Keys in policy_dict:", self.policy_dict.keys())

    def value(self, state, turns_left) -> float:
        """
        Recursively compute the value of a state given the number of turns left.
        Args:
            state (Dict): The current state of the environment.
            turns_left (int): The number of turns remaining.
        Returns:
            float: The expected value of the state.
        """
        # Base case: No turns left
        if turns_left == 0:
            return 0

        # Check if the value is already computed
        state_key = self.hash_state(state)
        if (state_key, turns_left) in self.memo:
            return self.memo[(state_key, turns_left)]

        # Initialize the maximum value
        max_value = float('-inf')
        best_action = None
        cur_actions = self.actions(state)

        # Iterate over all possible actions
        for action in cur_actions:
            if isinstance(action, str) and action == "terminate":
                # Special case: Terminate action
                terminate_value = 0  # No next state
                if terminate_value > max_value:
                    max_value = terminate_value
                    best_action = action
            else:
                expected_value = 0
                # Compute the expected value of the action
                for next_state, prob in self.transition_model(state, action):
                    reward = self.reward_model(next_state, action)
                    expected_value += prob * (reward + self.value(next_state, turns_left - 1))

                # Update the maximum value
                if expected_value > max_value:
                    max_value = expected_value
                    best_action = action

        self.policy_dict[(state_key, turns_left)] = best_action

        # Store the computed value in the memo dictionary
        self.memo[(state_key, turns_left)] = max_value
        return max_value

    def hash_state(self, state: Dict) -> Tuple:
        """
        Create a hashable tuple representation of the state.
        Args:
            state (Dict): The state to hash.
        Returns:
            Tuple: A hashable tuple representation of the state.
        """
        # Convert the state into a tuple of sorted key-value pairs
        wizards_tuple = tuple((name, tuple(data["location"])) for name, data in sorted(state["wizards"].items()))
        horcrux_tuple = tuple(
            (name, (tuple(data["location"]), tuple(data["possible_locations"]), data["prob_change_location"]))
            for name, data in sorted(state["horcrux"].items())
        )
        death_eaters_tuple = tuple(
            (name, (data["index"], tuple(data["path"]))) for name, data in sorted(state["death_eaters"].items())
        )
        return (wizards_tuple, horcrux_tuple, death_eaters_tuple)



    def actions(self, state):
        """Return the valid actions that can be executed in the given state."""
        res = []
        destroyed = set()
        for wizard_name, loc in state['wizards'].items():
            atomic_actions = []
            x,y = loc['location'][0], loc['location'][1]
            # adding move actions
            dirs = [(1,0),(0,1),(-1,0),(0,-1)]
            for d in dirs:
                cur = (x+d[0], y+d[1])
                if 0<=cur[0]<len(self.map) and 0<=cur[1]<len(self.map[0]) and self.map[cur[0]][cur[1]] != 'I':
                    atomic_actions.append(('move', wizard_name, cur))

            #adding destroy actions
            for h_name, h_dict in state['horcrux'].items():
                if h_dict['location'] == (x, y) and (x, y) not in destroyed:
                    atomic_actions.append(('destroy', wizard_name, h_name))
                    destroyed.add((x, y))
            #adding wait actions
            atomic_actions.append(('wait', wizard_name))
            res.append(atomic_actions)

        #getting all possible actions by calculating all permutations of atomic actions
        action_combinations = itertools.product(*res)
        # Convert the result to a tuple of tuples
        return list(action_combinations) + ['terminate', 'reset']

    def act(self, state):
        state_key = self.hash_state(state)
        action = self.policy_dict[(state_key, self.turns)]

        self.turns = self.turns - 1
        return action






    def reward_model(self, state, action):
        """
        Compute the immediate reward for a state and action.
        """
        if action == 'terminate':
            return 0
        if action == 'reset':
            return -2
        reward = 0
        for a in action:
            if a[0] == 'destroy':
                reward += 2
        for wizard_name, loc in state['wizards'].items():
            for de_name, de_dict in state['death_eaters'].items():
                idx = de_dict['index']
                if loc['location'] == de_dict['path'][idx]:
                    reward -= 1
        return reward



    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighbors for a position on the map.
        Args:
            pos (Tuple[int, int]): Current position.
        Returns:
            List[Tuple[int, int]]: Valid neighboring positions.
        """
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols and self.map[nx][ny] == 'P':
                neighbors.append((nx, ny))
        return neighbors




    def transition_model(self, state, action):
        if isinstance(action, str) and action == 'reset':
            return [(self.initial, 1.0)]

        next_states = []

        # Start by copying the current state and decrementing turns
        new_state = copy.deepcopy(state)

        # Apply each atomic action in the global action tuple
        for atomic_action in action:
            action_type = atomic_action[0]
            wizard_name = atomic_action[1]

            if action_type == "move":
                destination = atomic_action[2]
                new_state["wizards"][wizard_name]["location"] = destination


        # Generate all possible combinations of horcrux and death eater positions
        horcrux_combinations = self.get_horcrux_combinations(new_state)
        death_eater_combinations = self.get_death_eater_combinations(new_state)

        # Combine horcrux and death eater possibilities
        for horcrux_state, horcrux_prob in horcrux_combinations:
            for death_eater_state, death_eater_prob in death_eater_combinations:
                combined_state = copy.deepcopy(new_state)
                combined_state["horcrux"] = horcrux_state
                combined_state["death_eaters"] = death_eater_state
                combined_prob = horcrux_prob * death_eater_prob

                # Add the combined state and its probability to the next_states dictionary
                #state_key = self.hash_state(combined_state)
                if combined_state not in next_states:
                    next_states.append((combined_state, combined_prob))
        return next_states


    def get_horcrux_combinations(self, state: Dict) -> List[Tuple[Dict, float]]:
        """
        Generate all possible configurations of horcrux positions and their probabilities.
        Args:
            state (Dict): The current state.
        Returns:
            List[Tuple[Dict, float]]: A list of tuples, where each tuple contains a
                                      horcrux configuration and its probability.
        """
        horcruxes = state["horcrux"]
        horcrux_combinations = []

        # Generate all combinations of horcrux positions
        horcrux_states = []
        horcrux_probs = []

        for horcrux_name, horcrux_data in horcruxes.items():
            possible_positions = horcrux_data["possible_locations"]
            prob_change = horcrux_data["prob_change_location"]

            # Calculate probabilities for each position
            current_pos = horcrux_data["location"]
            transitions = [(1 - prob_change + (prob_change / len(possible_positions)), current_pos)]
            for pos in possible_positions:
                if pos != current_pos:
                    transitions.append((prob_change / len(possible_positions), pos))

            horcrux_states.append([pos for _, pos in transitions])
            horcrux_probs.append([prob for prob, _ in transitions])

        # Compute all combinations of horcrux positions
        for positions, probs in zip(product(*horcrux_states), product(*horcrux_probs)):
            horcrux_config = {
                horcrux_name: {"location": pos, "possible_locations": horcruxes[horcrux_name]["possible_locations"],
                               "prob_change_location": horcruxes[horcrux_name]["prob_change_location"]}
                for horcrux_name, pos in zip(horcruxes.keys(), positions)
            }
            horcrux_combinations.append((horcrux_config, float(math.prod(probs))))

        return horcrux_combinations

    def get_death_eater_combinations(self, state: Dict) -> List[Tuple[Dict, float]]:
        """
        Generate all possible configurations of death eater positions and their probabilities.
        Args:
            state (Dict): The current state containing death eaters with their indices and paths.
        Returns:
            List[Tuple[Dict, float]]: A list of tuples, where each tuple contains
            a death eater configuration and its probability.
        """

        def get_possible_transitions(index: int, path_length: int) -> List[Tuple[float, int]]:
            """Helper function to get possible transitions and probabilities."""
            if index == 0:
                if path_length > 1:
                    return [(0.5, 0), (0.5, 1)]  # Stay or move forward
                else:
                    return [(1, 0)]
            elif index == path_length - 1:
                return [(0.5, index), (0.5, index - 1)]  # Stay or move backward
            else:
                return [(1 / 3, index), (1 / 3, index - 1), (1 / 3, index + 1)]  # Stay, back, or forward

        death_eaters = state["death_eaters"]
        death_eater_combinations = []
        death_eater_states = []  # Will store possible next indices for each death eater
        death_eater_probs = []  # Will store probabilities corresponding to those indices

        # Calculate possible transitions for each death eater
        for de_name, de_data in death_eaters.items():
            index = de_data["index"]
            path = de_data["path"]

            # Get possible transitions for this death eater
            transitions = get_possible_transitions(index, len(path))

            # Store indices and probabilities separately
            death_eater_states.append([idx for _, idx in transitions])
            death_eater_probs.append([prob for prob, _ in transitions])

        # Generate all possible combinations using itertools.product
        for indices, probs in zip(product(*death_eater_states), product(*death_eater_probs)):
            # Create new configuration for this combination
            death_eater_config = {
                de_name: {
                    "index": idx,  # Use index directly
                    "path": death_eaters[de_name]['path']
                }
                for de_name, idx in zip(death_eaters.keys(), indices)
            }

            # Calculate total probability for this combination
            total_prob = float(math.prod(probs))
            death_eater_combinations.append((death_eater_config, total_prob))

        return death_eater_combinations

    def tuple_to_dict_state(state_tuple: Tuple) -> Dict:
        wizards_tuple, horcrux_tuple, death_eater_tuple, turns_left = state_tuple

        # Convert wizards
        wizards = {
            name: {"location": location}
            for name, location in wizards_tuple
        }

        # Convert horcruxes
        horcrux = {
            name: {"location": location}
            for name, location in horcrux_tuple
        }

        # Convert death eaters
        death_eaters = {
            name: {"index": index}
            for name, index in death_eater_tuple
        }

        # Construct the full dictionary
        return {
            "wizards": wizards,
            "horcrux": horcrux,
            "death_eaters": death_eaters,
            "turns_to_go": turns_left
        }


    def get_states(self, initial):
        """
        Generate all possible states from the given initial input structure.

        Args:
            initial: A dictionary representing the initial state of the problem.

        Returns:
            List of all possible states in the same structure as the initial input.
        """
        # Extract the grid and components from the initial state
        grid = initial["map"]
        turns_to_go = initial["turns_to_go"]
        wizard_names = list(initial["wizards"].keys())
        horcrux_names = list(initial["horcrux"].keys())
        death_eater_names = list(initial["death_eaters"].keys())

        # Valid tiles for wizards (all 'P' tiles in the grid)
        valid_tiles = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row) if cell == 'P']
        wizard_positions = itertools.combinations_with_replacement(valid_tiles, len(wizard_names))

        # Possible positions for horcruxes
        horcrux_positions = itertools.product(
            *[initial["horcrux"][name]["possible_locations"] for name in horcrux_names]
        )

        # Possible positions for death eaters
        death_eater_positions = itertools.product(
            *[initial["death_eaters"][name]["path"] for name in death_eater_names]
        )

        # Combine all possibilities
        all_combinations = itertools.product(wizard_positions, horcrux_positions, death_eater_positions)

        # Create states
        all_states = []
        for wizards, horcruxes, death_eaters in all_combinations:
            # Build the new state dictionary
            state = {
                "optimal": initial["optimal"],
                "turns_to_go": turns_to_go,
                "wizards": {name: {"location": loc} for name, loc in zip(wizard_names, wizards)},
                "horcrux": {
                    name: {
                        "location": loc,
                        "possible_locations": initial["horcrux"][name]["possible_locations"],
                        "prob_change_location": initial["horcrux"][name]["prob_change_location"]
                    }
                    for name, loc in zip(horcrux_names, horcruxes)
                },
                "death_eaters": {
                    name: {
                        "index": initial["death_eaters"][name]["path"].index(loc),
                        "path": initial["death_eaters"][name]["path"]
                    }
                    for name, loc in zip(death_eater_names, death_eaters)
                }
            }
            all_states.append(state)

        return all_states



class WizardAgent:
    def __init__(self, initial: Dict):
        self.initial = copy.deepcopy(initial)
        self.map = initial['map']
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.turns = initial['turns_to_go']
        self.wizards = initial['wizards']
        self.horcruxes = initial['horcrux']
        self.death_eaters = initial['death_eaters']

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.rows and 0 <= new_y < self.cols and self.map[new_x][new_y] == 'P':
                neighbors.append((new_x, new_y))
        return neighbors

    def get_closest_horcrux(self, wizard_pos: Tuple[int, int], state: Dict, assigned_horcruxes: set) -> Tuple[
        str, Dict]:
        min_dist = float('inf')
        closest_horcrux = None, None

        for hor_name, hor_info in state['horcrux'].items():
            if hor_name in assigned_horcruxes:
                continue

            dist = self.manhattan_distance(wizard_pos, hor_info['location'])
            if dist < min_dist:
                min_dist = dist
                closest_horcrux = hor_name, hor_info

        return closest_horcrux

    def evaluate_position(self, state: Dict, pos: Tuple[int, int], is_current=False, target_horcrux=None) -> float:
        score = 0.0

        if target_horcrux:
            dist = self.manhattan_distance(pos, target_horcrux['location'])
            prob_stay = 1 - target_horcrux['prob_change_location']

            if dist == 0:
                score += 1000 * prob_stay
            else:
                score += 150 / (dist * (1 + dist))

            for possible_loc in target_horcrux['possible_locations']:
                possible_dist = self.manhattan_distance(pos, possible_loc)
                if possible_dist == 0:
                    score += 100 * target_horcrux['prob_change_location'] / len(target_horcrux['possible_locations'])
                else:
                    score += (90 / (possible_dist * (1 + possible_dist))) * (
                                target_horcrux['prob_change_location'] / len(target_horcrux['possible_locations']))
        else:
            for hor_info in state['horcrux'].values():
                dist = self.manhattan_distance(pos, hor_info['location'])
                prob_stay = 1 - hor_info['prob_change_location']

                if dist == 0:
                    score += 200 * prob_stay
                else:
                    score += 150 / (dist * (1 + dist))

                for possible_loc in hor_info['possible_locations']:
                    possible_dist = self.manhattan_distance(pos, possible_loc)
                    if possible_dist == 0:
                        score += 100 * hor_info['prob_change_location'] / len(hor_info['possible_locations'])
                    else:
                        score += (60 / (possible_dist * (1 + possible_dist))) * (
                                    hor_info['prob_change_location'] / len(hor_info['possible_locations']))

        danger_score = self.get_danger_score(pos, state)
        score -= danger_score * 0.3
        return score

    def get_danger_score(self, pos: Tuple[int, int], state: Dict) -> float:
        danger = 0

        for de_info in state['death_eaters'].values():
            path = de_info['path']
            index = de_info['index']

            if len(path) == 1:
                dist = self.manhattan_distance(pos, path[0])
                if dist == 0:
                    danger += 100.0
                continue

            curr_pos = path[index]
            curr_dist = self.manhattan_distance(pos, curr_pos)

            if curr_dist == 0:
                danger += 100.0 / (1 + curr_dist)

            if index > 0:
                prev_dist = self.manhattan_distance(pos, path[index - 1])
                if prev_dist == 0:
                    danger += 15.5 / (1 + prev_dist)

            if index < len(path) - 1:
                next_dist = self.manhattan_distance(pos, path[index + 1])
                if next_dist == 0:
                    danger += 15.5 / (1 + next_dist)

        return danger

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def act(self, state: Dict):
        actions = {}
        assigned_horcruxes = set()

        # First pass - prioritize destroying horcruxes and assign targets
        for wizard_name, wizard_info in state['wizards'].items():
            curr_pos = wizard_info['location']

            # Try to destroy
            for hor_name, hor_info in state['horcrux'].items():
                if curr_pos == hor_info['location']:
                    actions[wizard_name] = ("destroy", wizard_name, hor_name)
                    assigned_horcruxes.add(hor_name)
                    break

            if wizard_name in actions:
                continue

            # Get closest unassigned horcrux
            _, target_horcrux = self.get_closest_horcrux(curr_pos, state, assigned_horcruxes)
            if target_horcrux:
                curr_score = self.evaluate_position(state, curr_pos, True, target_horcrux)
                best_action = ("wait", wizard_name)
                best_score = curr_score

                neighbors = self.get_neighbors(curr_pos)
                neighbors.sort(key=lambda n: -self.evaluate_position(state, n, False, target_horcrux))

                for neighbor in neighbors:
                    neighbor_score = self.evaluate_position(state, neighbor, False, target_horcrux)
                    if neighbor_score > best_score:
                        best_score = neighbor_score
                        best_action = ("move", wizard_name, neighbor)

                actions[wizard_name] = best_action

            else:
                curr_score = self.evaluate_position(state, curr_pos, True)
                best_action = ("wait", wizard_name)
                best_score = curr_score

                neighbors = self.get_neighbors(curr_pos)
                for neighbor in neighbors:
                    neighbor_score = self.evaluate_position(state, neighbor)
                    if neighbor_score > best_score:
                        best_score = neighbor_score
                        best_action = ("move", wizard_name, neighbor)

                actions[wizard_name] = best_action
        return tuple(actions.values())
