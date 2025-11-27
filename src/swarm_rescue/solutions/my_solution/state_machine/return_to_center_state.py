from swarm_rescue.solutions.my_solution.drone_template import DroneTemplate
from swarm_rescue.solutions.my_solution.state_machine.state import State
from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.solutions.my_solution.state_machine.state import StateNames


class ReturnToCenterState(State):
    def get_command(self, drone: DroneTemplate) -> tuple[CommandsDict, StateNames]:
        next_pose = drone.pose

        if State.rescue_center_pose is not None:
            next_pose = State.rescue_center_pose

        planned_pose = drone.path_planner.get_next_pose(
            drone.pose, drone.occupancy_grid, next_pose
        )
        if planned_pose is not None:
            next_pose = planned_pose

        command = self._get_command_from_pose(drone, next_pose, grasper=1)

        if not drone.grasper.grasped_wounded_persons:
            return command, StateNames.EXPLORE

        return command, StateNames.RETURN_CENTER
