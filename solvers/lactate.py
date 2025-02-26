import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class LactateState(State):
    def __init__(self, data: dict):
        # Lactate level in mmol/L (normal range: 0.5-1.0)
        self._lactate = data.get("lactate", 0.8)
        # Tissue perfusion indicator (0-100%)
        self._perfusion = data.get("perfusion", 100.0)

    @property
    def state(self) -> dict:
        return {
            "lactate": self._lactate,
            "perfusion": self._perfusion
        }


class LactateSolver(Solver):
    def __init__(self,
                 baseline_production: float = 0.02,
                 clearance_rate: float = 0.05,
                 perfusion_sensitivity: float = 0.1):
        """
        Initialize lactate solver with metabolic parameters
        
        :param baseline_production: Base rate of lactate production
        :param clearance_rate: Rate at which lactate is cleared
        :param perfusion_sensitivity: How strongly perfusion affects lactate
        """
        self._state = LactateState({})
        self.baseline_production = baseline_production
        self.clearance_rate = clearance_rate
        self.perfusion_sensitivity = perfusion_sensitivity

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ls = LactateState(state)
        
        # Get blood pressure if available (affects perfusion)
        blood_pressure = state.get("blood_pressure", 90.0)
        
        # Update perfusion based on blood pressure
        # Perfusion drops when BP < 65 mmHg
        if blood_pressure < 65.0:
            perfusion_drop = (65.0 - blood_pressure) * 2.0
            new_perfusion = max(0, min(100, ls.state["perfusion"] - perfusion_drop * dt))
        else:
            # Gradual recovery when BP is adequate
            new_perfusion = min(100, ls.state["perfusion"] + 5.0 * dt)
        
        # Lactate production increases with poor perfusion
        perfusion_factor = (100.0 - new_perfusion) / 100.0
        production = (self.baseline_production + 
                     self.perfusion_sensitivity * perfusion_factor)
        
        # Clearance is impaired with poor perfusion
        effective_clearance = self.clearance_rate * (new_perfusion / 100.0)
        
        # Calculate lactate change
        lactate_change = (production - 
                         effective_clearance * ls.state["lactate"]) * dt
        new_lactate = max(0, ls.state["lactate"] + lactate_change)

        logger.debug(
            f"LactateSolver: lactate={new_lactate:.2f} mmol/L, "
            f"perfusion={new_perfusion:.1f}%, "
            f"BP={blood_pressure:.1f}"
        )

        return LactateState({
            "lactate": new_lactate,
            "perfusion": new_perfusion
        })