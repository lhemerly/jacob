import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class PCRState(State):
    def __init__(self, data: dict):
        # CRP level in mg/L (normal < 3.0)
        self._crp = data.get("crp", 1.0)
        # Inflammation score (0-100)
        self._inflammation = data.get("inflammation", 0.0)

    @property
    def state(self) -> dict:
        return {
            "crp": self._crp,
            "inflammation": self._inflammation
        }


class PCRSolver(Solver):
    def __init__(self,
                 baseline_production: float = 0.01,
                 inflammation_sensitivity: float = 0.2,
                 clearance_rate: float = 0.05):
        """
        Initialize PCR solver with inflammatory response parameters
        
        :param baseline_production: Base rate of CRP production
        :param inflammation_sensitivity: How strongly inflammation affects CRP
        :param clearance_rate: Rate at which CRP is cleared
        """
        self._state = PCRState({})
        self.baseline_production = baseline_production
        self.inflammation_sensitivity = inflammation_sensitivity
        self.clearance_rate = clearance_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ps = PCRState(state)
        
        # Get infection level if available (affects inflammation)
        infection_level = state.get("infection_level", 0.0)
        
        # Update inflammation based on infection and existing inflammation
        # Inflammation has some inertia and doesn't change instantly
        inflammation_target = infection_level
        inflammation_change = (inflammation_target - 
                             ps.state["inflammation"]) * 0.1 * dt
        new_inflammation = max(0, min(100, 
                             ps.state["inflammation"] + inflammation_change))
        
        # CRP production increases with inflammation
        production = (self.baseline_production + 
                     self.inflammation_sensitivity * new_inflammation)
        
        # Calculate CRP change (production minus clearance)
        crp_change = (production - 
                     self.clearance_rate * ps.state["crp"]) * dt
        new_crp = max(0, ps.state["crp"] + crp_change)

        logger.debug(
            f"PCRSolver: CRP={new_crp:.1f} mg/L, "
            f"inflammation={new_inflammation:.1f}%, "
            f"infection={infection_level:.1f}"
        )

        return PCRState({
            "crp": new_crp,
            "inflammation": new_inflammation
        })