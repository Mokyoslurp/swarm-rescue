from enum import Enum

from typing import NamedTuple, Optional

from abc import ABC, abstractmethod

import numpy as np
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import (
    DroneSemanticSensor,
)
from swarm_rescue.simulation.utils.utils import circular_mean
from swarm_rescue.solutions.my_solution.controllers.pid_controller import PIDController

from swarm_rescue.solutions.my_solution.drone_pose import DronePose
from swarm_rescue.solutions.my_solution.drone_template import DroneTemplate


class SemanticData(NamedTuple):
    distance: float
    angle: float
    entity_type: DroneSemanticSensor.TypeEntity
    grasped: int


class StateNames(Enum):
    EXPLORE = 0
    GRASP = 1
    RETURN_CENTER = 2


class State(ABC):
    rescue_center_pose: Optional[DronePose] = None
    controller = PIDController(Kp=(1.6, 9.0), Ki=(2, 2), Kd=(8.0, 0.6))

    @abstractmethod
    def get_command(self, drone: DroneTemplate) -> tuple[CommandsDict, StateNames]: ...

    def _get_command_from_pose(
        self, drone: DroneTemplate, pose: DronePose, grasper: int = 0
    ):
        command: CommandsDict = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": grasper,
        }

        if pose is not None:
            self.controller.setpoint = pose
            command = self.controller.get_command(drone.pose)

            command["grasper"] = grasper

        return command

    def detect_wounded(self, drone: DroneTemplate):
        detection_semantic: list[SemanticData] = drone.semantic_values()

        found_wounded = False

        best_angle = 0
        best_distance = 0

        if detection_semantic:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (
                    data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON
                    and not data.grasped
                ):
                    found_wounded = True
                    v = data.angle * data.angle + (
                        data.distance * data.distance / 10**5
                    )
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]
                    best_distance = score[2]

        if found_wounded:
            wounded_pose = self.semantic_measure_to_pose(
                best_angle, best_distance, drone
            )

            return found_wounded, wounded_pose

        else:
            return found_wounded, drone.pose

    def detect_rescue_center(self, drone: DroneTemplate):
        detection_semantic: list[SemanticData] = drone.semantic_values()

        found_rescue_center = False

        angles_list = []
        distances_list = []

        if detection_semantic:
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    distances_list.append(data.distance)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))
                best_distance = 0.9 * (float)(np.mean(distances_list))

                rescue_center_pose = self.semantic_measure_to_pose(
                    best_angle, best_distance, drone
                )

                return found_rescue_center, rescue_center_pose

        return found_rescue_center, drone.pose

    def semantic_measure_to_pose(
        self, angle: float, distance: float, drone: DroneTemplate
    ) -> DronePose:
        total_angle = angle + drone.pose.yaw

        delta_pose = DronePose(
            distance * np.array([np.cos(total_angle), np.sin(total_angle)]),
            angle,
        )
        return drone.pose + delta_pose
