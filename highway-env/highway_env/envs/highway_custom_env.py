import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.road.lane import LineType, StraightLane, SineLane

class HighwayEnvMod(AbstractEnv):
    """
    A highway driving environment mod for ProjectS.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "screen_width": 1300,
            "screen_height": 150,
            "vehicles_count": 15,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._make_vehicles()

    def _create_road(self) -> None:
        """
                Make a road composed of a straight highway and a merging lane.
                :return: the road
                """
        net = RoadNetwork()
        start = 0
        length = 800
        lanes = 3
        nodes_str = ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            if lane == 1:
                # pass
                pos1 = [end[0] / 3 + 50, lane * StraightLane.DEFAULT_WIDTH]
            if lane == 2:
                pos2 = [pos1[0] + 50, lane * StraightLane.DEFAULT_WIDTH]
            line_types = [LineType.CONTINUOUS_LINE if lane == 0 else LineType.STRIPED,
                          LineType.CONTINUOUS_LINE if lane == lanes - 1 else LineType.NONE]
            net.add_lane(*nodes_str, StraightLane(origin, end, line_types=line_types))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, pos1))
        road.objects.append(Obstacle(road, pos2))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        num_CAV = self.config.get("controlled_vehicles")
        num_HDV = self.config.get("vehicles_count") - num_CAV

        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        spawn_points_s = []
        points = [5, 20, 40, 62, 75, 90, 105]
        tmp = []
        count = 0
        for lane in range(3):
            for point in points:
                spawn_points_s.append([point, lane])
                tmp.append(count)
                count = count + 1
        tmp.pop()
        spawn_point_s_c = np.random.choice(tmp, num_CAV + num_HDV, replace=False)

        spawn_point_s_c = list(spawn_point_s_c)
        # # remove the points to avoid duplicate
        # for a in spawn_point_s_c:
        #     spawn_point_s_c.remove(a)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)
        speed = 25

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV):
            p, l = spawn_points_s[spawn_point_s_c.pop(0)]
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("0", "1", l)).position(
                p + loc_noise.pop(0), 0), speed=speed)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV):
            p, l = spawn_points_s[spawn_point_s_c.pop(0)]
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("0", "1", l)).position(
                    p + loc_noise.pop(0), 0),
                                    speed=speed))

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        reward = \
            + self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        reward = utils.lmap(reward,
                            [self.config["collision_reward"],
                             self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                            [-1, 1])
        reward = 0 if not self.vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road) or \
            all(vehicle.position[0] > 800 for vehicle in self.controlled_vehicles)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvModCustom(HighwayEnvMod):
    """
    A variant of highway-mod-custom-v0 with custom execution:
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "vehicles_count": 20,
            "controlled_vehicles": 4
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._make_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


register(
    id='highway-mod-v0',
    entry_point='highway_env.envs:HighwayEnvMod',
)

register(
    id='highway-mod-custom-v0',
    entry_point='highway_env.envs:HighwayEnvModCustom',
)
