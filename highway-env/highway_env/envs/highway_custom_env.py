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
            "vehicles_count": 2,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -10,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [15, 35],
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
        length = 400
        lanes = 3
        nodes_str = ("0", "1")
        for lane in range(lanes):
            origin = np.array([start, lane * StraightLane.DEFAULT_WIDTH])
            end = np.array([start + length, lane * StraightLane.DEFAULT_WIDTH])
            if lane == 1:
                # pass
                pos1 = [end[0] / 2 + 100, lane * StraightLane.DEFAULT_WIDTH]
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
        points = [100, 125, 150, 175, 190]
        tmp = []
        count = 0
        x = StraightLane.DEFAULT_WIDTH
        p1 = [
            [(170, 1), (190, 1)],
            [(170, 1), (185, 1), (195, 2)],
            [(180, 0), (170, 1), (185, 1), (195, 2)],
            [(180, 0), (170, 1), (185, 1), (173, 2), (195, 2)]
        ]
        saiSo = 100
        for lane in range(3):
            for point in points:
                spawn_points_s.append([point, lane])
                tmp.append(count)
                count = count + 1
        tmp.pop()
        spawn_point_s_c = np.random.choice(tmp, num_CAV + num_HDV, replace=False)

        spawn_point_s_c = list(spawn_point_s_c)
        speed = 15

        num_total = num_CAV + num_HDV
        if num_total == 5:
            make_car_point = [
                [110, 0],
                [90, 1], [130, 1],
                [80, 2], [125, 2]
            ]
            make_cav_point = [
                [make_car_point[1]],
                [make_car_point[1], make_car_point[4]],
                [make_car_point[0], make_car_point[2], make_car_point[3]],
                [make_car_point[0], make_car_point[1], make_car_point[3], make_car_point[4]],
                [make_car_point[0], make_car_point[1], make_car_point[2], make_car_point[3], make_car_point[4]]
            ]
            make_hav_point = [
                [make_car_point[2]],
                [make_car_point[1], make_car_point[4]],
                [make_car_point[0], make_car_point[2], make_car_point[3]],
                [make_car_point[0], make_car_point[2], make_car_point[3], make_car_point[4]]
            ]
            for i in range(num_CAV):
                point, lane = make_cav_point[num_CAV - 1][i]
                ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("0", "1", lane)).position(point, 0), speed=speed)
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)

            for index in range(num_HDV):
                point, lane = make_hav_point[num_HDV - 1][index]
                road.vehicles.append(
                    other_vehicles_type(road, road.network.get_lane(("0", "1", lane)).position(point, 0), speed=speed))

        elif num_total == 2:
            if num_CAV == 1:
                p, l = p1[0][0]
                ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("0", "1", 2)).position(250 - saiSo, 0), speed=speed)
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)
                p = p1[0][1][0]
                road.vehicles.append(
                        other_vehicles_type(road, road.network.get_lane(("0", "1", 1)).position(250 - saiSo, 0), speed=speed))

            else:
                # num_CAV = 2 && HDV = 0
                pass
        elif num_total == 3 and num_CAV == 2:
                p, l = p1[0][0]
                ego_vehicle = self.action_type.vehicle_class(road,
                                                             road.network.get_lane(("0", "1", 2)).position(250 - saiSo,
                                                                                                           0),
                                                             speed=speed)
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)
                ego_vehicle = self.action_type.vehicle_class(road,
                                                             road.network.get_lane(("0", "1", 2)).position(200 - saiSo,
                                                                                                           0),
                                                             speed=speed)
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)
                p = p1[0][1][0]
                road.vehicles.append(
                    other_vehicles_type(road, road.network.get_lane(("0", "1", 1)).position(250 - saiSo, 0),
                                        speed=speed))

        else:
            """spawn the CAV on the straight road first"""
            for _ in range(num_CAV):
                p, l = spawn_points_s[spawn_point_s_c.pop(0)]
                ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("0", "1", l)).position(p, 0), speed=speed)
                self.controlled_vehicles.append(ego_vehicle)
                road.vehicles.append(ego_vehicle)

            """spawn the HDV on the main road first"""
            for _ in range(num_HDV):
                p, l = spawn_points_s[spawn_point_s_c.pop(0)]
                road.vehicles.append(
                    other_vehicles_type(road, road.network.get_lane(("0", "1", l)).position(p, 0), speed=speed))

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
            all(vehicle.position[0] > 500 for vehicle in self.controlled_vehicles)
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
            "vehicles_count": 2,
            "controlled_vehicles": 1
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
