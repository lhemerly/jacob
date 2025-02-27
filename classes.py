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
        
    @property
    def initial_state(self) -> dict:
        """
        Default initial values for state variables needed by this coupler.
        Override this in derived classes to provide default values.
        """
        return {}
        
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


class Scenario(ABC):
    """
    Base interface/abstract class for all simulation scenarios.
    Scenarios represent clinical events or conditions that unfold over time.
    """
    
    def __init__(self):
        """
        Initialize the scenario with default values.
        """
        self.is_active = False
        self.elapsed_time = 0.0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the scenario.
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description of the scenario.
        """
        pass
    
    @property
    @abstractmethod
    def affected_keys(self) -> list:
        """
        The dictionary keys that this scenario may modify in the global state.
        """
        pass
    
    @property
    def duration(self) -> float:
        """
        The total duration of the scenario in simulation time.
        Default is indefinite (represented by -1).
        Override this in derived classes to provide a specific duration.
        """
        return -1  # -1 represents indefinite duration
    
    @property
    def initial_state(self) -> dict:
        """
        Default initial values for state variables needed by this scenario.
        Override this in derived classes to provide default values.
        """
        return {}
    
    @abstractmethod
    def apply(self, state: dict, dt: float) -> dict:
        """
        Apply the scenario effects to the state.
        
        :param state: The global state dictionary
        :param dt: Time step
        :return: Dictionary of state changes to apply
        """
        pass
    
    def activate(self):
        """
        Activate the scenario.
        """
        self.is_active = True
        self.elapsed_time = 0.0
    
    def deactivate(self):
        """
        Deactivate the scenario.
        """
        self.is_active = False
        self.elapsed_time = 0.0
    
    def update(self, state: dict, dt: float) -> dict:
        """
        Update the scenario state and apply effects if active.
        
        :param state: The global state dictionary
        :param dt: Time step
        :return: Dictionary of state changes to apply
        """
        if not self.is_active:
            return {}
        
        # Update elapsed time
        self.elapsed_time += dt
        
        # Check if scenario should end due to duration
        if self.duration > 0 and self.elapsed_time >= self.duration:
            self.deactivate()
            return {}
        
        # Apply scenario effects
        return self.apply(state, dt)
