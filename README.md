[![Python](https://img.shields.io/pypi/pyversions/motndp.svg)](https://badge.fury.io/py/motndp)
[![PyPI](https://badge.fury.io/py/motndp.svg)](https://badge.fury.io/py/motndp)

# Multi-Objective Transport Network Design Problem
A Gymnasium environment to design public transport networks, by satisfying the Origin-Destination demand of multiple socio-economic groups (objectives). 

The environment can also be used for single-objective transport network design, by using one group.

![animation of transport network designer agent](/resources/motndp.gif "MOTNDP")

## Install
```bash
pip install motndp
```
Note: this includes all the structure to run MOTNDP environments, but not the city data. To run the provided cities, you need to download the 'cities' folder of this repository, or add your own city data. Read on to learn more about how to structure your data.

## Description
The Multi-Objective Transport Network Design Problem (MOTNDP) is a combinatorial optimization problem that involves designing the optimal transport line in a city to meet travel demand between various locations.

The MOTNDP environment is a highly modular environment that allows for the creation of different types of transport network design problems.
In this environemnt, an agent is tasked with placing stations within a city represented as a grid. Episodes start in a cell, and the agent can move between adjacent cells to place new stations.
The environment is based on the City object, which contains the city grid and group definitions for each cell, and the Constraints object, which contains the constraints on agent's movement in the grid and can be used to mask actions, creating different types of transport lines, such as straight lines, curves, or loops.
Each grid cell represents a location associated with a specific group. The reward reflects the total OD demand satisfied for each group, either in absolute or percentage terms.

## Table of Contents
- [Multi-Objective Transport Network Design Problem](#multi-objective-transport-network-design-problem)
  - [Install](#install)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Space](#reward-space)
  - [Starting State](#starting-state)
  - [Episode Termination](#episode-termination)
  - [Arguments](#arguments)
  - [Building Blocks](#building-blocks)
    - [City](#city)
    - [Constraints](#constraints)
  - [Example](#example)
  - [Available City Environments](#available-city-environments)
  - [Citing](#citing)

## Observation Space
The observation is a concatenation of:
- A multi-binary vector of length `grid_size` where each entry is 1 if the corresponding cell has been covered by a station.
- A one-hot vector of length `grid_size` representing the agent's current location.

The observation space is a `MultiBinary` space with shape `(2 * grid_size,)`.

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

Note that you can use MOTNDP to perform single objective transport network design, by assigning all grid cells to the same group and usign nr_groups=1. This will still return a vector reward, but with a single dimension, and you can then flatten it to get a scalar reward.

## Starting State
The starting state is the initial location of the agent in the grid. The initial location can be set with the `starting_loc` argument. If not set, the starting location is chosen randomly.

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

## Building Blocks
![motndp structure](/resources/motndp_structure.jpg "MOTNDP")
MOTNDP is a modular environment designed to train transport planning agents in any city, provided the necessary data is available. It relies on two key components: the City and Constraints objects.
- The City object calculates the satisfied origin-destination (OD) demand and determines connections with existing transport lines.
- The Constraints object defines the allowed agent actions at each timestep, creating action masks that respect the desired rules and constraints.

To create an MOTNDP environment, you need to define and specify these two objects.
### City
The City object encapsulates all the information about the city required for transport planning. It includes:
- OD matrix: Representing the demand between origin and destination cells in the grid.
- Group definitions: Defining the group membership of each cell in the city grid.
- Configuration: Including the city grid size and details of existing transport lines (for network expansion scenarios).

The City also handles key logic for reward calculation through the satisfied_od_mask() function. This function takes a transport segment (connection between two grid cells) as input and returns the satisfied OD pairs. Additionally, it can include an excluded_od_segments list, which specifies segments that yield zero rewards (e.g., areas where transport cannot be built or regions where transport lines are discouraged).

Additional Features:

- Existing Transport Line Integration: The City object can calculate connections with existing lines, boosting the reward by capturing additional satisfied OD flows.
Directory Structure:

A valid city directory should include the following files:
- config.txt: Configuration file for the city.
- od.txt: OD matrix data.
- Group definition files: One or more files defining group membership for each cell.

Note: The cities folder is not included in this package and must be downloaded separately. Example files are provided in the cities folder to guide you in setting up your data.

### Constraints
The Constraints object defines the set of allowed and disallowed actions at each timestep for the agent, enabling the creation of diverse transport line types. For example, metro lines often span long distances and follow straight, non-meandering paths. Other transport types may have different movement restrictions.

Constraints expose an abstract mask_actions() function that you can implement to tailor the agent’s allowed actions.

Example Constraints:
- BasicConstraints: Prevents the agent from exiting the grid or revisiting previously visited cells.
- MetroConstraints: Ensures the design of straight, one-directional lines without cyclical movements, mimicking the structure of most metro lines worldwide.

Once the City and Constraints objects are provided, the MOTNDP environment can be used like any other mo_gym environment. It exposes standard functions such as reset(), step(action) and render().

## Example
```python
from pathlib import Path
from motndp.city import City
import gymnasium
from gymnasium.envs.registration import register

from motndp.constraints import MetroConstraints

register(
    id="motndp_dilemma-v0",
    entry_point="motndp.motndp:MOTNDP"
)

if __name__ == '__main__':
    dilemma_dir = Path(f"./cities/dilemma_5x5")
    city = City(
            dilemma_dir, 
            groups_file="groups.txt",
            ignore_existing_lines=True
    )

    nr_episodes = 100
    nr_stations = 9
    seed = 42
    
    env = gymnasium.make(
        'motndp_dilemma-v0', 
        city=city, 
        constraints=MetroConstraints(city), 
        nr_stations=nr_stations)

    training_step = 0
    for _ in range(nr_episodes):
        state, info = env.reset()
        while True:
            env.render()
            # Random policy -- Implement your own agent policy here
            action = env.action_space.sample(mask=info['action_mask'])
            new_state, reward, done, _, info = env.step(action)

            training_step += 1
            print(f'state: {state}, action: {action}, reward: {reward} new_state: {new_state}')

            if done:
                break

            state = new_state
```
For an example of how tabular Q-learning is being used with MOTNDP, check out [this repository](https://github.com/dimichai/tabular-tndp).

## Available City Environments
I aim to gather data for various cities by referencing papers published on the topic. If you have data to contribute, feel free to reach out or open a pull request.

| City                    | Nr. Groups | Sources                                                                                                                     | OD Data                                                                     |
|-------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Amsterdam (Netherlands) | 1-10       | [paper](https://www.arxiv.org/abs/2411.18195), [files](https://github.com/dimichai/mo-tndp/tree/main/cities/amsterdam)                                    | Estimated with [model](https://www.nature.com/articles/s41586-021-03480-9 ) |
| Xi'an (China)           | 1-10       | [paper](https://dl.acm.org/doi/10.1145/3394486.3403315), [files](https://github.com/dimichai/mo-tndp/tree/main/cities/xian) | GPS                                                                         |
| Dilemma (Toy example)     | 2          | [paper](https://www.arxiv.org/abs/2411.18195), [files](https://github.com/dimichai/mo-tndp/tree/main/cities/dilemma_5x5)                                  | Fake                                                                        |


## Citing
```bibtex
If you use this environment in your paper, please cite:
@article{michailidis2026scalable,
  title={Scalable multi-objective reinforcement learning with fairness guarantees using lorenz dominance},
  author={Michailidis, Dimitris and R{\"o}pke, Willem and Roijers, Diederik M and Ghebreab, Sennay and Santos, Fernando P},
  journal={Journal of Artificial Intelligence Research},
  volume={85},
  year={2026}
}
```
