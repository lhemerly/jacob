import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class SedationState(State):
    def __init__(self, data: dict):
        # Sedation score (0=awake, through 5=unarousable)
        self._sedation_score = data.get("sedation_score", 0)
        # Consciousness level (0-100)
        self._consciousness = data.get("consciousness", 100.0)
        # Active sedative medications
        self._propofol = data.get("propofol", 0.0)  # mg/kg/hr
        self._midazolam = data.get("midazolam", 0.0)  # mg/hr
        self._dexmedetomidine = data.get("dexmedetomidine", 0.0)  # mcg/kg/hr

    @property
    def state(self) -> dict:
        return {
            "sedation_score": self._sedation_score,
            "consciousness": self._consciousness,
            "propofol": self._propofol,
            "midazolam": self._midazolam,
            "dexmedetomidine": self._dexmedetomidine
        }


class SedationSolver(Solver):
    def __init__(self,
                 propofol_potency: float = 0.2,
                 midazolam_potency: float = 0.1,
                 dexmed_potency: float = 0.3,
                 metabolism_rate: float = 0.1):
        """
        Initialize sedation solver with pharmacological parameters
        
        :param propofol_potency: How strongly propofol affects consciousness
        :param midazolam_potency: How strongly midazolam affects consciousness
        :param dexmed_potency: How strongly dexmedetomidine affects consciousness
        :param metabolism_rate: Rate at which medications are metabolized
        """
        self._state = SedationState({})
        self.propofol_potency = propofol_potency
        self.midazolam_potency = midazolam_potency
        self.dexmed_potency = dexmed_potency
        self.metabolism_rate = metabolism_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ss = SedationState(state)
        
        # Medication metabolism
        new_propofol = max(0, ss.state["propofol"] - 
                          self.metabolism_rate * ss.state["propofol"] * dt)
        new_midazolam = max(0, ss.state["midazolam"] - 
                           self.metabolism_rate * ss.state["midazolam"] * dt)
        new_dexmed = max(0, ss.state["dexmedetomidine"] - 
                        self.metabolism_rate * ss.state["dexmedetomidine"] * dt)
        
        # Calculate total sedative effect
        sedative_effect = (self.propofol_potency * new_propofol +
                          self.midazolam_potency * new_midazolam +
                          self.dexmed_potency * new_dexmed)
        
        # Update consciousness level
        # Natural tendency to wake up balanced against sedative effects
        wake_tendency = max(0, (100.0 - ss.state["consciousness"]) * 0.1 * dt)
        sedation_effect = sedative_effect * dt
        
        new_consciousness = max(0, min(100,
                                     ss.state["consciousness"] +
                                     wake_tendency - sedation_effect))
        
        # Determine sedation score based on consciousness level
        if new_consciousness >= 90:
            new_score = 0  # Awake and alert
        elif new_consciousness >= 70:
            new_score = 1  # Drowsy but responds to voice
        elif new_consciousness >= 50:
            new_score = 2  # Light sedation
        elif new_consciousness >= 30:
            new_score = 3  # Moderate sedation
        elif new_consciousness >= 10:
            new_score = 4  # Deep sedation
        else:
            new_score = 5  # Unarousable

        logger.debug(
            f"SedationSolver: score={new_score}, "
            f"consciousness={new_consciousness:.1f}%, "
            f"propofol={new_propofol:.1f}, "
            f"midazolam={new_midazolam:.1f}, "
            f"dexmed={new_dexmed:.2f}"
        )

        return SedationState({
            "sedation_score": new_score,
            "consciousness": new_consciousness,
            "propofol": new_propofol,
            "midazolam": new_midazolam,
            "dexmedetomidine": new_dexmed
        })