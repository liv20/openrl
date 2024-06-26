from typing import Optional

import numpy as np

from openrl.envs.mpe.core import Agent, Landmark, World
from openrl.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self, **kwargs):
        self.render_mode = None
        if "collision_penalty" not in kwargs or kwargs['collision_penalty'] is None:
            self.collision_penalty = 1
        else:
            self.collision_penalty = kwargs["collision_penalty"]

    def make_world(
        self,
        render_mode: Optional[str] = None,
        world_length: int = 25,
        num_agents: int = 3,
        num_landmarks: int = 3,
    ):
        self.render_mode = render_mode
        world = World()
        world.name = "simple_spread"
        world.world_length = world_length
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks  # 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(world.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, np_random: Optional[np.random.Generator] = None):
        # random properties for agents
        if np_random is None:
            np_random = np.random.default_rng()
        world.assign_agent_colors()

        world.assign_landmark_colors()

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = 0.8 * np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for landmark in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for landmark in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - landmark.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= self.collision_penalty
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )
