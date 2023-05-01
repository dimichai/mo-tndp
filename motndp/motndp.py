import numpy as np

import gymnasium as gym
from gymnasium import spaces
from motndp.city import City


class MOTNDP(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    metadata = {}

    def __init__(self, city: City, nr_stations: int, render_mode=None):
        # city environment
        self.city = city
        # total number of stations (steps) to place
        self.nr_stations = nr_stations
        self.stations_placed = 0
        # visited cells in the grid (by index)
        self.covered_cells_vid = []
        # visited cells in the grid (by coordinate)
        self.covered_cells_gid = [] 
        # covered segments in the grid (pairs of cells)
        self.covered_segments = []
        self.observation_space = spaces.Discrete(self.city.grid_size)

        # We have 8 actions, corresponding to
        # 0: walk up
        # 1: walk up-right
        # 2: walk right
        # 3: walk down-right
        # 4: walk down
        # 5: walk down-left
        # 6: walk left
        # 7: walk up-left
        self.action_space = spaces.Discrete(8)
        # Allowed actions are updated at each step, based on the current location
        self.action_mask = np.ones(self.action_space.n, dtype=np.int8)
        
        # Nr of groups in the city (used for multi-objective reward)
        self.nr_groups = self.city.groups.shape[0]
        # Reward space is a vector of length `self.nr_groups` with values between 0 and 1 -- used in morl-baselines
        self.reward_space = spaces.Box(low=np.array([0] * self.nr_groups), high=np.array([1] * self.nr_groups), dtype=np.float32)

        """
        The following array maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        """
        self._action_to_direction = np.array([
            [-1, 0],
            [-1, 1],
            [0 , 1],
            [1 , 1],
            [1 , 0],
            [1, -1],
            [0, -1],
            [-1, -1]
        ])


    def _get_obs(self):
        return {'location': self._agent_location, 
                'location_vid': self._agent_location_vid, 
                'location_vector': self._agent_location_vector}
    
    def _get_info(self):
        return {'segments': self.covered_segments, 'action_mask': self.action_mask}
    
    def _calculate_reward(self, segment, use_pct=True):
        # DO NOT SET use_pct to false, because it will render the reward_space useless
        assert self.city.group_od_mx, 'Cannot use multi-objective reward without group definitions. Provide --groups_file argument'

        if segment in self.covered_segments:
            return np.zeros(len(self.city.group_od_mx))

        segment = np.array(segment)
        sat_od_mask = self.city.satisfied_od_mask(segment)
        sat_group_ods = np.zeros(len(self.city.group_od_mx))
        sat_group_ods_pct = np.zeros(len(self.city.group_od_mx))
        for i, g_od in enumerate(self.city.group_od_mx):
            sat_group_ods[i] = (g_od * sat_od_mask).sum().item()
            sat_group_ods_pct[i] = sat_group_ods[i] / g_od.sum()

        if use_pct:
            group_rw = sat_group_ods_pct
        else:
            group_rw = sat_group_ods
        
        return group_rw
    
    def _update_agent_location(self, new_location):
        """Given the new location of the agent in grid coordinates (x, y), update a number of internal variables.
        _agent_location: the new location of the agent in grid coordinates (x, y)
        _agent_location_vid: the new location of the agent as a discrete discrete index (0,0) -> 0, (0,1) -> 1, ....
        _agent_location_vector: the new location of the agent as a one-hot vector.

        Args:
            new_location (_type_): _description_
        """
        self._agent_location = new_location
        self._agent_location_vid = self.city.grid_to_vector(self._agent_location[None, :]).item()
        self._agent_location_vector = np.zeros(self.observation_space.n)
        self._agent_location_vector[self._agent_location_vid] = 1
    
    def _update_action_mask(self, location, prev_action=None):
        # Apply action mask based on the location of the agent (it should stay inside the grid) & previously visited cells (it should not visit the same cell twice)
        possible_locations = location + self._action_to_direction 
        self.action_mask = np.all(possible_locations >= 0, axis=1) &  \
                            (possible_locations[:, 0] < self.city.grid_x_size) & \
                            (possible_locations[:, 1] < self.city.grid_y_size) & \
                            (~np.isin(self.city.grid_to_vector(possible_locations), self.covered_cells_vid)) # mask out visited cells
        self.action_mask = self.action_mask.astype(np.int8)

    def reset(self, seed=None, loc=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        if loc:
            if type(loc) == tuple:
                loc = np.array([loc[0], loc[1]])
            self._update_agent_location(loc)
        # Choose the agent's location uniformly at random, by sampling its x and y coordinates
        else:
            agent_x = self.np_random.integers(0, self.city.grid_x_size)
            agent_y = self.np_random.integers(0, self.city.grid_y_size)
            self._update_agent_location(np.array([agent_x, agent_y]))

        self.stations_placed =  1
        self.covered_cells_vid = [self.city.grid_to_vector(self._agent_location[None, :]).item()]
        self.covered_cells_gid = [self._agent_location]
        self.covered_segments = []

        self._update_action_mask(self._agent_location)
        observation = self._get_obs()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, self._get_info()
    
    def step(self, action):
        # Map the action to the direction we walk in
        direction = self._action_to_direction[action]
        new_location = np.array([
            self._agent_location[0] + direction[0],
            self._agent_location[1] + direction[1]
        ])
        self.stations_placed += 1

        # We add a new dimension to the agent's location to match grid_to_vector's generalization
        from_idx = self.city.grid_to_vector(self._agent_location[None, :]).item()
        to_idx = self.city.grid_to_vector(new_location[None, :]).item()
        reward = self._calculate_reward([from_idx, to_idx])

        self.covered_segments.append([from_idx, to_idx])
        self.covered_segments.append([to_idx, from_idx])
        self.covered_cells_vid.append(to_idx)
        self.covered_cells_gid.append(new_location)

        # Update the agent's location
        self._update_agent_location(new_location)
        # Update the action mask
        self._update_action_mask(self._agent_location, action)

        # An episode is done if the agent has placed all stations under the budget or if there's no more actions to take
        terminated = self.stations_placed >= self.nr_stations or np.sum(self.action_mask) == 0

        observation = self._get_obs()
        info  = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # We return the observation, the reward, whether the episode is done, truncated=False and no info
        return observation, reward, terminated, False, info

