import logging
import taichi as ti
from typing import List, Optional, Dict, Any
from classes import Solver, Coupler, Scenario, Action

# Initialize Taichi with CPU/GPU backend
ti.init(arch=ti.cpu)  # Can be changed to ti.gpu if needed
logging.basicConfig(level=logging.INFO)


class Master:
    """
    The master simulation class.
    - Maintains a single global state dictionary.
    - Each step, it delegates to solvers in sequence and couplers.
    - Allows applying scenarios that unfold over time.
    - Uses Taichi for parallel computation.
    - We unify logging and track a simple time.
    """

    def __init__(self, solvers: List[Solver], dt: float = 1.0, 
                 couplers: Optional[List[Coupler]] = None,
                 scenarios: Optional[List[Scenario]] = None,
                 actions: Optional[List[Action]] = None):
        """
        :param solvers: list of Solver instances
        :param dt: default timestep
        :param couplers: list of Coupler instances for handling interactions between solvers
        :param scenarios: list of Scenario instances that can be activated
        :param actions: list of Action instances that can be performed
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.solvers = solvers
        self.couplers = couplers or []
        self.scenarios = scenarios or []
        self.actions = actions or []
        self.active_scenarios = []
        self.active_actions = []  # Track actions with duration > 0
        self.state = {}
        self.dt = dt  # can be float
        self.current_time = 0.0

        # Initialize the global state by collecting all solver values.
        for solver in self.solvers:
            try:
                self.state = {**self.state, **solver.state}
            except AttributeError:
                self.logger.error("Solver %s has no 'state' attribute", solver)
                raise
        
        # Initialize state with default values from couplers
        self._initialize_coupler_state()
        
        # Initialize state with default values from scenarios
        self._initialize_scenario_state()

        self.logger.info("Master initialized. Known keys: %s", list(self.state.keys()))
        
        # Verify couplers don't modify state owned exclusively by solvers
        self._validate_couplers()
    
    def _initialize_coupler_state(self):
        """
        Initialize state with default values from each coupler's initial_state.
        Only adds values for keys that don't already exist in the state.
        """
        for coupler in self.couplers:
            # Get initial state values defined by each coupler
            coupler_default_state = coupler.initial_state
            
            # Only add default values if they don't already exist in state
            for key, value in coupler_default_state.items():
                if key not in self.state:
                    self.state[key] = value
                    self.logger.info(f"Added default value for {key}: {value} from {coupler.__class__.__name__}")
    
    def _initialize_scenario_state(self):
        """
        Initialize state with default values from each scenario's initial_state.
        Only adds values for keys that don't already exist in the state.
        """
        for scenario in self.scenarios:
            # Get initial state values defined by each scenario
            scenario_default_state = scenario.initial_state
            
            # Only add default values if they don't already exist in state
            for key, value in scenario_default_state.items():
                if key not in self.state:
                    self.state[key] = value
                    self.logger.info(f"Added default value for {key}: {value} from scenario {scenario.name}")

    def _validate_couplers(self):
        """
        Ensure couplers only modify state they are allowed to.
        """
        solver_owned_keys = set()
        for solver in self.solvers:
            solver_owned_keys.update(solver.state.keys())
        
        for coupler in self.couplers:
            for key in coupler.output_keys:
                if key not in self.state:
                    self.logger.warning(f"Coupler will create new state key: {key}")

    def step(self):
        """
        Progress the simulation by dt, letting each solver update its part of the state
        and then applying couplers for interactions between solvers.
        Also applies active scenarios.
        Uses Taichi for parallelism where possible.
        """
        # Parallel solver step - prepare data structures
        solver_results = {}
        
        # Let each solver parse and solve (can be executed in parallel)
        for solver in self.solvers:
            local_state_dict = solver.parse_state(self.state)
            new_local_state = solver.solve(local_state_dict, self.dt)
            # Store results for later update
            solver_results[solver] = new_local_state.state
        
        # Update global state with all solver results
        for solver_state in solver_results.values():
            self.state.update(solver_state)
            
        # Apply couplers after all solvers have updated their states
        for coupler in self.couplers:
            local_state = coupler.parse_state(self.state)
            coupled_state = coupler.couple(local_state, self.dt)
            # Update only the keys that the coupler is allowed to modify
            for key in coupler.output_keys:
                if key in coupled_state:
                    self.state[key] = coupled_state[key]

        # Apply active scenarios
        self._apply_scenarios()
        
        # Apply ongoing effects from active actions
        self._apply_active_actions()

        self.current_time += self.dt
        self.logger.info(str(self))
        return self.state
    
    def _apply_scenarios(self):
        """
        Apply all active scenarios to the state.
        Remove scenarios that have completed.
        """
        # Create a copy of the list since we might modify it during iteration
        for scenario in list(self.active_scenarios):
            # Apply scenario effects
            changes = scenario.update(self.state, self.dt)
            
            # Apply changes to the state
            for key, value in changes.items():
                if key in self.state:
                    self.state[key] = value
                else:
                    # Add the key with the specified value
                    self.state[key] = value
                    self.logger.info(f"Created new state key from scenario: {key} = {value}")
            
            # Remove scenario if it's no longer active
            if not scenario.is_active:
                self.active_scenarios.remove(scenario)
                self.logger.info(f"Scenario {scenario.name} has completed")
                
    def _apply_active_actions(self):
        """
        Apply ongoing effects from actions with duration > 0.
        Remove actions that have completed their duration.
        """
        # Create a copy of the list since we might modify it during iteration
        active_actions_copy = list(self.active_actions)
        
        for action_data in active_actions_copy:
            action = action_data["action"]
            elapsed_time = action_data["elapsed_time"]
            duration = action.duration
            kwargs = action_data.get("kwargs", {})
            
            # Update elapsed time
            elapsed_time += self.dt
            action_data["elapsed_time"] = elapsed_time
            
            # Check if action duration has completed
            if elapsed_time >= duration:
                self.active_actions.remove(action_data)
                self.logger.info(f"Action {action.name} has completed its duration")
                continue
            
            # For actions with ongoing effects, we apply a fraction of the initial effect
            # based on remaining duration
            remaining_fraction = (duration - elapsed_time) / duration
            
            # Apply action effects adjusted by remaining fraction
            self._apply_action_changes(action, kwargs, remaining_fraction)

    def apply_scenario(self, scenario_name: str):
        """
        Activate a scenario by name.
        
        :param scenario_name: The name of the scenario to activate
        :return: True if scenario was found and activated, False otherwise
        """
        for scenario in self.scenarios:
            if scenario.name == scenario_name:
                if scenario not in self.active_scenarios:
                    scenario.activate()
                    self.active_scenarios.append(scenario)
                    self.logger.info(f"Scenario {scenario_name} activated")
                    return True
                else:
                    self.logger.warning(f"Scenario {scenario_name} is already active")
                    return True
        
        self.logger.warning(f"Scenario {scenario_name} not found")
        return False

    def deactivate_scenario(self, scenario_name: str):
        """
        Deactivate a scenario by name.
        
        :param scenario_name: The name of the scenario to deactivate
        :return: True if scenario was found and deactivated, False otherwise
        """
        for scenario in self.active_scenarios:
            if scenario.name == scenario_name:
                scenario.deactivate()
                self.active_scenarios.remove(scenario)
                self.logger.info(f"Scenario {scenario_name} deactivated")
                return True
        
        self.logger.warning(f"Active scenario {scenario_name} not found")
        return False
        
    def perform_action(self, action: Action, **kwargs) -> Dict[str, Any]:
        """
        Perform an action and apply its effects to the state.
        If the action has a duration > 0, it will be added to active_actions
        to continue its effects over time.
        
        :param action: The Action instance to perform
        :param kwargs: Additional parameters specific to this action
        :return: Dictionary of observable state values after performing the action
        """
        # Apply immediate effects of the action
        changes = self._apply_action_changes(action, kwargs)
        
        # If the action has duration > 0, add it to active_actions
        if action.duration > 0:
            self.active_actions.append({
                "action": action,
                "elapsed_time": 0.0,
                "kwargs": kwargs
            })
            self.logger.info(f"Action {action.name} with duration {action.duration:.2f} added to active actions")
        
        # Get observable state after action
        observable_state = action.get_observable_state(self.state)
        
        self.logger.info(f"Action {action.name} performed with changes: {changes}")
        return observable_state
    
    def perform_action_by_name(self, action_name: str, **kwargs) -> Dict[str, Any]:
        """
        Find an action by name and perform it.
        
        :param action_name: Name of the action to perform
        :param kwargs: Additional parameters specific to this action
        :return: Dictionary of observable state values after performing the action, empty if action not found
        """
        for action in self.actions:
            if action.name == action_name:
                return self.perform_action(action, **kwargs)
                
        self.logger.warning(f"Action {action_name} not found")
        return {}
    
    def _apply_action_changes(self, action: Action, kwargs: Dict[str, Any], multiplier: float = 1.0) -> Dict[str, float]:
        """
        Apply the effects of an action to the state, with an optional multiplier.
        
        :param action: The Action instance
        :param kwargs: Additional parameters specific to this action
        :param multiplier: Multiplier for the action effects (used for decaying effects)
        :return: Dictionary of changes applied
        """
        changes = action.apply(self.state, **kwargs)
        
        # Apply changes to the state with multiplier
        for key, delta in changes.items():
            adjusted_delta = delta * multiplier
            if key in self.state:
                self.state[key] += adjusted_delta
            else:
                # Add the key with the specified value
                self.state[key] = adjusted_delta
                self.logger.info(f"Created new state key from action: {key} = {adjusted_delta}")
                
        return changes

    def actions(self, actions: Dict[str, float]):
        """
        Perform immediate state changes according to an actions dictionary.
        For user/model actions like medications, pacemaker configurations, procedures, etc.
        
        Example:
        actions = {"heart_rate": +5, "blood_pressure": -1.2}
        
        :param actions: Dictionary of state keys and their delta values (or absolute values for new keys)
        :return: Updated global state
        """
        for key, value in actions.items():
            if key in self.state:
                self.state[key] += value
            else:
                # Add the key with the specified value instead of throwing an error
                self.state[key] = value
                self.logger.info(f"Created new state key: {key} = {value}")

        self.logger.info("Actions applied: %s", actions)
        self.logger.info(str(self))
        return self.state

    def __str__(self):
        """
        Pretty-print the current simulation time and all global state.
        """
        state_str = ", ".join([f"{k}={v:.2f}" for k, v in self.state.items()])
        active_scenarios = ", ".join([s.name for s in self.active_scenarios]) if self.active_scenarios else "None"
        active_actions = ", ".join([a["action"].name for a in self.active_actions]) if self.active_actions else "None"
        return f"Time={self.current_time:.2f}, Active Scenarios: {active_scenarios}, Active Actions: {active_actions}, State: {state_str}"
