import logging
import taichi as ti
from typing import List, Optional
from classes import Solver, Coupler

# Initialize Taichi with CPU/GPU backend
ti.init(arch=ti.cpu)  # Can be changed to ti.gpu if needed
logging.basicConfig(level=logging.INFO)


class Master:
    """
    The master simulation class.
    - Maintains a single global state dictionary.
    - Each step, it delegates to solvers in sequence and couplers.
    - Uses Taichi for parallel computation.
    - We unify logging and track a simple time.
    """

    def __init__(self, solvers: List[Solver], dt: float = 1.0, couplers: Optional[List[Coupler]] = None):
        """
        :param solvers: list of Solver instances
        :param dt: default timestep
        :param couplers: list of Coupler instances for handling interactions between solvers
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.solvers = solvers
        self.couplers = couplers or []
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

        self.logger.info("Master initialized. Known keys: %s", list(self.state.keys()))
        
        # Verify couplers don't modify state owned exclusively by solvers
        self._validate_couplers()

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

        self.current_time += self.dt
        self.logger.info(str(self))
        return self.state

    def actions(self, actions: dict):
        """
        Perform state changes according to an actions dictionary.
        Example:
        actions = {"heart_rate": +5, "blood_pressure": -1.2}
        """
        for key, value in actions.items():
            if key in self.state:
                self.state[key] += value
            else:
                self.logger.error("Key '%s' not found in global state", key)
                raise KeyError(f"Invalid state key: {key}")

        self.logger.info("Actions applied: %s", actions)
        self.logger.info(str(self))
        return self.state

    def __str__(self):
        """
        Pretty-print the current simulation time and all global state.
        """
        state_str = ", ".join([f"{k}={v:.2f}" for k, v in self.state.items()])
        return f"Time={self.current_time:.2f}, State: {state_str}"
