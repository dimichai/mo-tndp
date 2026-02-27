import numpy as np
import gymnasium as gym
from gymnasium import spaces
from motndp.city import City
from motndp.constraints import Constraints
import pygame

# Maps actions from `self.action_space` to the direction the agent will walk in if that action is taken.
ACTION_TO_DIRECTION = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
)


class MOTNDP(gym.Env):
    """
    ## Description
    The Multi-Objective Transport Network Design Problem (MOTNDP) is a combinatorial optimization problem that involves designing the optimal transport line in a city to meet travel demand between various locations.

    The MOTNDP environment is a highly modular environment that allows for the creation of different types of transport network design problems.
    In this environemnt, an agent is tasked with placing stations within a city represented as a grid. Episodes start in a cell, and the agent can move between adjacent cells to place new stations.
    The environment is based on the City object, which contains the city grid and group definitions for each cell, and the Constraints object, which contains the constraints on agent's movement in the grid and can be used to mask actions, creating different types of transport lines, such as straight lines, curves, or loops.
    Each grid cell represents a location associated with a specific group. The reward reflects the total OD demand satisfied for each group, either in absolute or percentage terms.

    ## Observation Space
    The observation space is a concatenation of a multi-binary vector of covered cells and a one-hot vector of the agent's current location.
    It is a MultiBinary space with shape (2 * grid_size,).

    ## Action Space
    The actions is a discrete space where:
    - 0: walk up
    - 1: walk up-right
    - 2: walk right
    - 3: walk down-right
    - 4: walk down
    - 5: walk down-left
    - 6: walk left
    - 7: walk up-left
    At every step, the agent can move to one of the 8 adjacent cells, as long as the movement is allowed by the action mask.
    When an agent moves to a new cell, it places a station in the grid, connecting the previous cell with the new one.

    ## Reward Space
    The reward is a vector of length `nr_groups` (number of groups in the city). The reward reflects the total OD demand satisfied for each group, either in absolute or percentage terms.
    The type of reward can be set with the `od_type` argument: 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).

    ## Starting State
    The starting state is the initial location of the agent in the grid. The starting location can be set with the `starting_loc` argument. If not set, the starting location is chosen randomly.

    ## Episode Termination
    An episode terminates when the agent has placed all stations under the budget or when there are no more actions to take.

    ## Arguments
    - city (City): City object that contains the grid and the groups.
    - constraints (Constraints): Transport constraints object with the constraints on movement in the grid.
    - nr_stations (int): Episode length. Total number of stations to place (each station is an episode step).
    - od_type (str): Type of Origin Destination metric. Can be 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).
    - chained_reward (bool): If True, each new station will receive an additional reward based not only on the ODs covered between the immediate previous station, but also those before.
    - starting_loc (tuple): Set the default starting location of the agent in the grid. If None, the starting location is chosen randomly, or chosen in _reset().
    - render_mode (str): if 'human', the environment will render a pygame window with the agent's movement and the covered cells.

    ## Cite
    This environment is based on the following paper: TODO Add paper here
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        city: City,
        constraints: Constraints,
        nr_stations: int,
        od_type="pct",
        chained_reward=False,
        starting_loc=None,
        render_mode=None,
    ):
        """
        Args:
            city (City): City object that contains the grid and the groups.
            constraints (Constraints): Transport constraints object with the constraints on movement in the grid.
            nr_stations (int): Episode length. Total number of stations to place (each station is an episode step).
            od_type (str): Type of Origin Destination metric. Can be 'pct' (returns the percentage of satisfied OD pairs for each group) or 'abs' (returns the absolute number of satisfied OD pairs for each group).
            chained_reward (bool): If True, each new station will receive an additional reward based not only on the ODs covered between the immediate previous station, but also those before.
            starting_loc (tuple): Set the default starting location of the agent in the grid. If None, the starting location is chosen randomly, or chosen in _reset().
            render_mode (str): RENDERING IS NOT IMPLEMENTED YET.
        """

        assert od_type in ["pct", "abs"], "Invalid Origin-Destination Type. Choose one of: pct, abs"

        self.render_mode = render_mode

        self.city = city
        self.mask_actions = constraints.mask_actions
        self.nr_stations = nr_stations
        self.starting_loc = starting_loc
        self.od_type = od_type
        # An example of the chained reward: Episode with two actions leads to transport line 1 -> 2 -> 3.
        # If chained_reward is false, the reward received after the second action will be the OD between 2-3.
        # If chained_reward is true, the reward received after the second action will be the OD between 2-3 AND 1-3, because station 1 and 3 are now connected (and therefore their demand is deemed satisfied).
        self.chained_reward = chained_reward

        self.observation_space = spaces.MultiBinary(city.grid_size * 2)

        self.action_space = spaces.Discrete(8)
        # Allowed actions are updated at each step, based on the current location
        self.action_mask = np.ones(self.action_space.n, dtype=np.int8)

        # Nr of groups in the city (used for multi-objective reward)
        self.nr_groups = self.city.groups.shape[0]

        # Reward space is a vector of length `self.nr_groups`
        if self.od_type == "pct":
            low_reward = np.zeros(self.nr_groups)
            high_reward = np.ones(self.nr_groups)
        else:
            low_reward = np.zeros(self.nr_groups)
            high_reward = self.city.group_od_mx.sum(axis=(1, 2))

        self.reward_space = spaces.Box(
            low=np.float32(low_reward), high=np.float32(high_reward), dtype=np.float32
        )

        self.window = None
        self.clock = None

    def get_agent_location(self):
        multi_binary = np.zeros(self.city.grid_size, dtype=np.int8)
        indices = self.city.grid_to_index(np.array(self.covered_cells_coordinates))
        multi_binary[indices] = 1
        location = self.city.grid_to_one_hot(self._loc_grid_coordinates[None, :])
        return np.concatenate([multi_binary, location])

    def _get_info(self):
        return {
            "segments": self.covered_segments,
            "action_mask": self.action_mask,
            "covered_cells_coordinates": self.covered_cells_coordinates,
            "location_grid_coordinates": self._loc_grid_coordinates,
            "location_grid_index": self._loc_grid_index,
        }

    def _calculate_reward(self, segment):
        assert (
            self.city.group_od_mx is not None
        ), "Cannot use multi-objective reward without group definitions. Provide --groups_file argument"

        # Return zero rewards if segment is already covered
        if segment in self.covered_segments:
            return np.zeros(self.city.group_od_mx.shape[0]), np.empty((0, 2))

        segment = np.asarray(segment)
        if self.chained_reward:
            # Convert covered cells to numpy array (if not already)
            cells_to_chain = np.asarray(
                self.city.grid_to_index(np.array(self.covered_cells_coordinates))
            )

            # Get stations connected to the new segment
            connected_stations = self.city.connections_with_existing_lines(segment)

            # Extend only if connected stations exist
            if connected_stations:
                # self.connections_with_existing_lines = list(set(self.connections_with_existing_lines).union(connected_stations))
                self.connections_with_existing_lines.update(connected_stations)

            # Get the satisfied OD mask with connections and cells to chain
            sat_od_mask, sat_od_pairs = self.city.satisfied_od_mask(
                segment,
                cells_to_chain=cells_to_chain,
                connected_cells=self.connections_with_existing_lines,
                segments_to_ignore=self.covered_segments,
                return_od_pairs=True,
            )
        else:
            # Get the satisfied OD mask without chained reward
            sat_od_mask, sat_od_pairs = self.city.satisfied_od_mask(
                segment, segments_to_ignore=self.covered_segments, return_od_pairs=True
            )

        # Compute satisfied group ODs and their percentages
        sat_group_ods = (self.city.group_od_mx * sat_od_mask).sum(axis=(1, 2))
        sat_group_ods_pct = np.divide(
            sat_group_ods,
            self.city.group_od_sum,
            out=np.zeros_like(sat_group_ods),
            where=self.city.group_od_sum != 0,
        )

        # Determine reward type (percentage or absolute)
        group_rw = sat_group_ods_pct if self.od_type == "pct" else sat_group_ods

        return group_rw, sat_od_pairs

    def _update_agent_location(self, new_location):
        """Given the new location of the agent in grid coordinates (x, y), update a number of internal variables.
        _loc_grid_coordinates: the new location of the agent in grid coordinates (x, y)
        _loc_grid_index: the new location of the agent as a discrete index (0,0) -> 0, (0,1) -> 1, ....

        Args:
            new_location (_type_): _description_
        """
        self._loc_grid_coordinates = new_location
        self._loc_grid_index = self.city.grid_to_index(
            self._loc_grid_coordinates[None, :]
        ).item()

    def _update_action_mask(self, location):
        # Apply action mask based on the location of the agent (it should stay inside the grid) & previously visited cells (it should not visit the same cell twice)
        possible_locations = location + ACTION_TO_DIRECTION
        self.action_mask = self.mask_actions(
            location, possible_locations, self.covered_cells_coordinates
        )

    def is_action_allowed(self, location, action):
        possible_locations = location + ACTION_TO_DIRECTION
        action_mask = self.mask_actions(
            location, possible_locations, self.covered_cells_coordinates
        )
        return action_mask[action] == 1

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        loc = options.get("loc") if options else None

        # loc argument supersedes self.starting_loc
        starting_loc = loc if loc is not None else self.starting_loc
        if starting_loc:
            if type(starting_loc) == tuple:
                starting_loc = np.array([starting_loc[0], starting_loc[1]])
            self._update_agent_location(starting_loc)

        # Choose the agent's location uniformly at random, by sampling its x and y coordinates
        else:
            agent_x = self.np_random.integers(0, self.city.grid_x_size)
            agent_y = self.np_random.integers(0, self.city.grid_y_size)
            self._update_agent_location(np.array([agent_x, agent_y]))

        self.stations_placed = 1
        self.covered_cells_coordinates = [self._loc_grid_coordinates]
        # covered segments in the grid (pairs of cells)
        self.covered_segments = []
        # Stations from the existing lines that are connected to the line that the agent is currently building.
        self.connections_with_existing_lines = set()

        self._update_action_mask(self._loc_grid_coordinates)
        observation = self.get_agent_location()

        if self.render_mode == "human":
            self.render()

        return observation, self._get_info()

    def step(self, action):
        # Map the action to the direction we walk in
        direction = ACTION_TO_DIRECTION[action]

        if self.is_action_allowed(self._loc_grid_coordinates, action):
            new_location = np.array(
                [
                    self._loc_grid_coordinates[0] + direction[0],
                    self._loc_grid_coordinates[1] + direction[1],
                ]
            )
            self.stations_placed += 1

            # We add a new dimension to the agent's location to match grid_to_index's generalization
            from_idx = self.city.grid_to_index(
                self._loc_grid_coordinates[None, :]
            ).item()
            to_idx = self.city.grid_to_index(new_location[None, :]).item()
            reward, sat_od_pairs = self._calculate_reward([from_idx, to_idx])

            # Update the covered segments and cells, based on sat_od_pairs
            for pair in sat_od_pairs:
                self.covered_segments.append(pair.tolist())
                self.covered_segments.append(
                    pair[::-1].tolist()
                )  # add the reverse OD pair

            self.covered_cells_coordinates.append(new_location)

            # Update the agent's location
            self._update_agent_location(new_location)
            # Update the action mask
            self._update_action_mask(self._loc_grid_coordinates)
        else:
            raise Exception(
                "Not allowed action was taken. Make sure you apply the constraints to the action selection."
            )

        # An episode is done if the agent has placed all stations under the budget or if there's no more actions to take
        terminated = (
            self.stations_placed >= self.nr_stations or np.sum(self.action_mask) == 0
        )

        observation = self.get_agent_location()
        info = self._get_info()

        if self.render_mode == "human":
            self.render(reward)

        # We return the observation, the reward, whether the episode is done, truncated=False and no info
        return observation, reward, terminated, False, info

    def render(self, reward=None):
        screen_width, screen_height = 1200, 800
        legend_width = 200  # Width for the legend
        grid_width = screen_width - legend_width
        grid_height = screen_height
        cell_size = min(grid_width // self.city.grid_x_size, grid_height // self.city.grid_y_size)
        window_width = cell_size * self.city.grid_x_size + legend_width
        window_height = cell_size * self.city.grid_y_size

        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. mo_gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("MOTNDP")
                self.window = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
            else:
                self.window = pygame.Surface((window_width, window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))
        # Define colors for each group once
        group_colors = {}
        for group_id in self.city.groups:
            blue_shade = 255 - int((group_id / self.nr_groups) * 100)
            group_colors[group_id] = (0, 0, blue_shade)

        for i in range(self.city.grid_x_size):
            for j in range(self.city.grid_y_size):
                group_id = self.city.grid_groups[i, j]
                if np.isnan(group_id):
                    continue
                color = group_colors[group_id]
                surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                surface.fill(color)
                self.window.blit(surface, (j * cell_size, i * cell_size))

        legend_x = self.city.grid_x_size * cell_size + 10
        legend_y = 10
        font = pygame.font.SysFont(None, 24)
        for group_id, color in group_colors.items():
            pygame.draw.rect(self.window, color[:3], (legend_x, legend_y, cell_size, cell_size))
            text = font.render(f'Group {group_id}', True, (0, 0, 0))
            self.window.blit(text, (legend_x + cell_size + 5, legend_y))
            legend_y += cell_size + 5

        if self.city.existing_lines:
            pygame.draw.circle(self.window, (255, 0, 0), (legend_x + 10, legend_y + 10), 10)
            pygame.draw.line(self.window, (255, 0, 0), (legend_x + 10, legend_y + 10), (legend_x + 30, legend_y + 10), 5)
            text = font.render('Existing Lines', True, (0, 0, 0))
            self.window.blit(text, (legend_x + 40, legend_y))
            legend_y += cell_size + 5

        pygame.draw.polygon(self.window, (255, 165, 0), [(legend_x + 10, legend_y), (legend_x, legend_y + 10), (legend_x + 20, legend_y + 10)])
        text = font.render('Agent Location', True, (0, 0, 0))
        self.window.blit(text, (legend_x + 40, legend_y))
        legend_y += cell_size + 5

        font = pygame.font.SysFont(None, 21)
        if reward is not None:
            for group_id, group_reward in enumerate(reward):
                text = font.render(f'Reward Group {group_id}: {group_reward:.2f}', True, (0, 0, 0))
                self.window.blit(text, (legend_x, legend_y + 60 + group_id * 20))

        for line in self.city.existing_lines:
            for i, cell in enumerate(line):
                cell = self.city.index_to_grid(cell)[0]
                pygame.draw.circle(
                    self.window,
                    (255, 0, 0),
                    (cell[1] * cell_size + cell_size // 2, cell[0] * cell_size + cell_size // 2),
                    cell_size // 2,
                )
                if i > 0:
                    prev_cell = line[i - 1]
                    prev_cell = self.city.index_to_grid(prev_cell)[0]
                    pygame.draw.line(
                        self.window,
                        (255, 0, 0),
                        (prev_cell[1] * cell_size + cell_size // 2, prev_cell[0] * cell_size + cell_size // 2),
                        (cell[1] * cell_size + cell_size // 2, cell[0] * cell_size + cell_size // 2),
                        cell_size // 5,
                    )

        for i, cell in enumerate(self.covered_cells_coordinates):
            pygame.draw.circle(
                self.window,
                (0, 0, 0),
                (cell[1] * cell_size + cell_size // 2, cell[0] * cell_size + cell_size // 2),
                cell_size // 2,
            )
            if i > 0:
                prev_cell = self.covered_cells_coordinates[i - 1]
                pygame.draw.line(
                    self.window,
                    (0, 0, 0),
                    (prev_cell[1] * cell_size + cell_size // 2, prev_cell[0] * cell_size + cell_size // 2),
                    (cell[1] * cell_size + cell_size // 2, cell[0] * cell_size + cell_size // 2),
                    cell_size // 5,
                )

        agent_x = self._loc_grid_coordinates[1] * cell_size + cell_size // 2
        agent_y = self._loc_grid_coordinates[0] * cell_size + cell_size // 2
        triangle_points = [
            (agent_x, agent_y - cell_size // 2),
            (agent_x - cell_size // 2, agent_y + cell_size // 2),
            (agent_x + cell_size // 2, agent_y + cell_size // 2),
        ]
        pygame.draw.polygon(self.window, (255, 165, 0), triangle_points)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
