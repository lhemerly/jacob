import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class FeverState(State):
    def __init__(self, data: dict):
        # Core temperature in Celsius
        self._temperature = data.get("temperature", 37.0)  # Normal is 37.0°C
        self._infection_level = data.get("infection_level", 0.0)  # 0-100 scale
        self._antipyretic_level = data.get("antipyretic_level", 0.0)  # medication level

    @property
    def state(self) -> dict:
        return {
            "temperature": self._temperature,
            "infection_level": self._infection_level,
            "antipyretic_level": self._antipyretic_level
        }


class FeverSolver(Solver):
    def __init__(self,
                 baseline_temp: float = 37.0,
                 max_fever: float = 41.5,
                 infection_temp_factor: float = 0.03,
                 antipyretic_effect: float = 0.02,
                 natural_regulation_rate: float = 0.01):
        """
        Initialize fever solver with physiological parameters
        
        :param baseline_temp: Normal body temperature in Celsius
        :param max_fever: Maximum allowable temperature
        :param infection_temp_factor: How much infection raises temperature
        :param antipyretic_effect: How strongly antipyretics reduce temperature
        :param natural_regulation_rate: Rate of natural temperature regulation
        """
        self._state = FeverState({})
        self.baseline_temp = baseline_temp
        self.max_fever = max_fever
        self.infection_temp_factor = infection_temp_factor
        self.antipyretic_effect = antipyretic_effect
        self.natural_regulation_rate = natural_regulation_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        fs = FeverState(state)
        
        # Calculate temperature change
        # 1. Effect of infection
        infection_effect = (self.infection_temp_factor * 
                          fs.state["infection_level"]) * dt
        
        # 2. Effect of antipyretics (reduces temperature)
        medication_effect = (self.antipyretic_effect * 
                           fs.state["antipyretic_level"]) * dt
        
        # 3. Natural regulation (tends toward baseline)
        natural_change = (self.natural_regulation_rate * 
                         (self.baseline_temp - fs.state["temperature"])) * dt
        
        # Calculate new temperature
        new_temp = (fs.state["temperature"] + 
                   infection_effect - 
                   medication_effect + 
                   natural_change)
        
        # Clamp temperature to physiological limits
        if new_temp > self.max_fever:
            new_temp = self.max_fever
        elif new_temp < 35.0:  # Severe hypothermia limit
            new_temp = 35.0
            
        # Update infection level (slowly decreases unless modified by other factors)
        new_infection = max(0, fs.state["infection_level"] - 0.1 * dt)
        
        # Antipyretic level decreases over time
        new_antipyretic = max(0, fs.state["antipyretic_level"] - 0.2 * dt)

        logger.debug(
            f"FeverSolver: temp={new_temp:.1f}°C, "
            f"infection={new_infection:.1f}, "
            f"antipyretic={new_antipyretic:.1f}"
        )

        return FeverState({
            "temperature": new_temp,
            "infection_level": new_infection,
            "antipyretic_level": new_antipyretic
        })