from swarm_rescue.solutions.my_solution.drone_template import DroneTemplate
from swarm_rescue.solutions.my_solution.state_machine.state import State, StateNames
from swarm_rescue.simulation.drone.controller import CommandsDict


class GraspWoundedState(State):
    def get_command(self, drone: DroneTemplate) -> tuple[CommandsDict, StateNames]:
        found_wounded, wounded_pose = self.detect_wounded(drone)

        next_pose = wounded_pose

        command = self._get_command_from_pose(drone, next_pose, grasper=1)

        if drone.grasper.grasped_wounded_persons:
            return command, StateNames.RETURN_CENTER

        if not found_wounded:
            return command, StateNames.EXPLORE

        return command, StateNames.GRASP
