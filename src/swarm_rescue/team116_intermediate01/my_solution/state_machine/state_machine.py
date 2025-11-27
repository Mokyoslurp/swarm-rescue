from swarm_rescue.solutions.my_solution.drone_template import DroneTemplate
from swarm_rescue.solutions.my_solution.state_machine.explore_state import ExploreState
from swarm_rescue.solutions.my_solution.state_machine.grasp_wounded_state import (
    GraspWoundedState,
)
from swarm_rescue.solutions.my_solution.state_machine.return_to_center_state import (
    ReturnToCenterState,
)
from swarm_rescue.solutions.my_solution.state_machine.state import State, StateNames


class StateMachine:
    def __init__(self) -> None:
        self.states = {
            StateNames.EXPLORE: ExploreState(),
            StateNames.GRASP: GraspWoundedState(),
            StateNames.RETURN_CENTER: ReturnToCenterState(),
        }

        self.current_state: State = self.states[StateNames.EXPLORE]

    def get_command(self, drone: DroneTemplate):
        command, next_state = self.current_state.get_command(drone)

        self.current_state = self.states[next_state]

        return command
