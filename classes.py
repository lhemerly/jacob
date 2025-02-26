from abc import ABC, abstractmethod


class State(ABC):
    """
    Base interface/abstract class for all simulation state classes.
    """

    @abstractmethod
    def state(self) -> dict:
        """
        The state owned by the solver
        """
        pass


class Solver(ABC):
    """
    Base interface/abstract class for all simulation modules.
    """

    @property
    @abstractmethod
    def state(self):
        """
        The dictionary keys owned by this solver.
        This helps parse the global state.
        """
        pass

    @abstractmethod
    def solve(self, state: State, dt: float) -> State:
        """
        Solve the simulation for dt.
        """
        pass

    def parse_state(self, global_state: dict) -> State:
        """
        Extract the relevant portion of the global_state for this solver.
        """
        parsed_state = {}
        for key in self.state:
            parsed_state[key] = global_state[key]
        return parsed_state


class Coupler(ABC):
    """
    Base interface/abstract class for all simulation couplers.
    Couplers handle interactions between different solvers' states.
    """
    
    @property
    @abstractmethod
    def input_keys(self) -> list:
        """
        The dictionary keys read by this coupler from the global state.
        """
        pass
        
    @property
    @abstractmethod
    def output_keys(self) -> list:
        """
        The dictionary keys modified by this coupler in the global state.
        """
        pass
        
    @abstractmethod
    def couple(self, state: dict, dt: float) -> dict:
        """
        Process interactions between different parts of the state.
        
        :param state: The global state dictionary
        :param dt: Time step
        :return: Updated state values as a dict
        """
        pass

    def parse_state(self, global_state: dict) -> dict:
        """
        Extract the relevant portion of the global_state for this coupler.
        """
        parsed_state = {}
        for key in self.input_keys:
            if key in global_state:
                parsed_state[key] = global_state[key]
        return parsed_state
