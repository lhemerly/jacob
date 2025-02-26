import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class ElectrolytesState(State):
    def __init__(self, data: dict):
        # Initialize electrolyte levels with normal ranges
        self._sodium = data.get("sodium", 140.0)      # Normal: 135-145 mEq/L
        self._potassium = data.get("potassium", 4.0)  # Normal: 3.5-5.0 mEq/L
        self._chloride = data.get("chloride", 102.0)  # Normal: 96-106 mEq/L
        self._calcium = data.get("calcium", 9.5)      # Normal: 8.5-10.5 mg/dL
        self._magnesium = data.get("magnesium", 2.0)  # Normal: 1.7-2.2 mg/dL
        self._phosphate = data.get("phosphate", 3.5)  # Normal: 2.5-4.5 mg/dL

    @property
    def state(self) -> dict:
        return {
            "sodium": self._sodium,
            "potassium": self._potassium,
            "chloride": self._chloride,
            "calcium": self._calcium,
            "magnesium": self._magnesium,
            "phosphate": self._phosphate
        }


class ElectrolytesSolver(Solver):
    def __init__(self,
                 na_regulation_rate: float = 0.02,
                 k_regulation_rate: float = 0.05,
                 cl_regulation_rate: float = 0.03,
                 ca_regulation_rate: float = 0.04,
                 mg_regulation_rate: float = 0.03,
                 phos_regulation_rate: float = 0.02):
        """
        Initialize electrolytes solver with homeostatic regulation rates
        
        :param na_regulation_rate: Rate at which sodium tends toward normal
        :param k_regulation_rate: Rate at which potassium tends toward normal
        :param cl_regulation_rate: Rate at which chloride tends toward normal
        :param ca_regulation_rate: Rate at which calcium tends toward normal
        :param mg_regulation_rate: Rate at which magnesium tends toward normal
        :param phos_regulation_rate: Rate at which phosphate tends toward normal
        """
        self._state = ElectrolytesState({})
        self.na_regulation_rate = na_regulation_rate
        self.k_regulation_rate = k_regulation_rate
        self.cl_regulation_rate = cl_regulation_rate
        self.ca_regulation_rate = ca_regulation_rate
        self.mg_regulation_rate = mg_regulation_rate
        self.phos_regulation_rate = phos_regulation_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        es = ElectrolytesState(state)
        
        # Homeostatic regulation for each electrolyte
        # Each tends toward its normal value at its specific rate
        
        # Sodium regulation (tends toward 140)
        na_change = self.na_regulation_rate * (140.0 - es.state["sodium"]) * dt
        new_sodium = es.state["sodium"] + na_change
        
        # Potassium regulation (tends toward 4.0)
        k_change = self.k_regulation_rate * (4.0 - es.state["potassium"]) * dt
        new_potassium = es.state["potassium"] + k_change
        
        # Chloride regulation (tends toward 102)
        cl_change = self.cl_regulation_rate * (102.0 - es.state["chloride"]) * dt
        new_chloride = es.state["chloride"] + cl_change
        
        # Calcium regulation (tends toward 9.5)
        ca_change = self.ca_regulation_rate * (9.5 - es.state["calcium"]) * dt
        new_calcium = es.state["calcium"] + ca_change
        
        # Magnesium regulation (tends toward 2.0)
        mg_change = self.mg_regulation_rate * (2.0 - es.state["magnesium"]) * dt
        new_magnesium = es.state["magnesium"] + mg_change
        
        # Phosphate regulation (tends toward 3.5)
        phos_change = self.phos_regulation_rate * (3.5 - es.state["phosphate"]) * dt
        new_phosphate = es.state["phosphate"] + phos_change

        logger.debug(
            f"ElectrolytesSolver: Na={new_sodium:.1f}, K={new_potassium:.1f}, "
            f"Cl={new_chloride:.1f}, Ca={new_calcium:.1f}, "
            f"Mg={new_magnesium:.1f}, Phos={new_phosphate:.1f}"
        )

        return ElectrolytesState({
            "sodium": new_sodium,
            "potassium": new_potassium,
            "chloride": new_chloride,
            "calcium": new_calcium,
            "magnesium": new_magnesium,
            "phosphate": new_phosphate
        })