from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.solutions.my_solution.drone_template import DroneTemplate
from swarm_rescue.solutions.my_solution.state_machine.state import State, StateNames


class ExploreState(State):
    def get_command(self, drone: DroneTemplate) -> tuple[CommandsDict, StateNames]:
        next_pose = drone.pose

        if State.rescue_center_pose is None:
            found_center, rescue_center_pose = self.detect_rescue_center(drone)

            if found_center:
                State.rescue_center_pose = rescue_center_pose

        found_wounded, wounded_pose = self.detect_wounded(drone)

        if found_wounded:
            next_pose = wounded_pose

        else:
            planned_pose = drone.path_planner.get_next_pose(
                drone.pose, drone.occupancy_grid
            )
            if planned_pose is not None:
                next_pose = planned_pose

        command = self._get_command_from_pose(drone, next_pose)

        if found_wounded:
            return command, StateNames.GRASP

        return command, StateNames.EXPLORE
