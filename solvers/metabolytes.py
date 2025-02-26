import logging
import math
from classes import Solver, State

logger = logging.getLogger(__name__)


class MetabolytesState(State):
    def __init__(self, data: dict):
        # Blood gases and acid-base
        self._ph = data.get("ph", 7.4)           # Normal: 7.35-7.45
        self._pco2 = data.get("pco2", 40.0)      # Normal: 35-45 mmHg
        self._hco3 = data.get("hco3", 24.0)      # Normal: 22-26 mEq/L
        self._po2 = data.get("po2", 95.0)        # Normal: 80-100 mmHg
        self._base_excess = data.get("base_excess", 0.0)  # Normal: -2 to +2

        # Metabolic parameters
        self._glucose = data.get("glucose", 100.0)  # Normal: 70-140 mg/dL
        self._ketones = data.get("ketones", 0.1)    # Normal: < 0.6 mmol/L
        self._insulin = data.get("insulin", 10.0)    # μU/mL

    @property
    def state(self) -> dict:
        return {
            "ph": self._ph,
            "pco2": self._pco2,
            "hco3": self._hco3,
            "po2": self._po2,
            "base_excess": self._base_excess,
            "glucose": self._glucose,
            "ketones": self._ketones,
            "insulin": self._insulin
        }


class MetabolytesSolver(Solver):
    def __init__(self,
                 glucose_baseline: float = 100.0,
                 insulin_sensitivity: float = 0.1,
                 respiratory_compensation_rate: float = 0.02,
                 metabolic_compensation_rate: float = 0.01):
        """
        Initialize metabolytes solver with physiological parameters
        
        :param glucose_baseline: Target glucose level
        :param insulin_sensitivity: How strongly insulin affects glucose
        :param respiratory_compensation_rate: Rate of respiratory pH compensation
        :param metabolic_compensation_rate: Rate of metabolic pH compensation
        """
        self._state = MetabolytesState({})
        self.glucose_baseline = glucose_baseline
        self.insulin_sensitivity = insulin_sensitivity
        self.respiratory_compensation_rate = respiratory_compensation_rate
        self.metabolic_compensation_rate = metabolic_compensation_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ms = MetabolytesState(state)
        
        # Get oxygen saturation if available (affects metabolism)
        oxy_saturation = state.get("oxy_saturation", 98.0)
        
        # Update PO2 based on oxygen saturation
        # Rough approximation using sigmoid relationship
        new_po2 = 27.0 * oxy_saturation - 2560.0 / (oxy_saturation + 1.0)
        new_po2 = max(40, min(150, new_po2))
        
        # Glucose metabolism
        # Affected by insulin levels and stress response
        insulin_effect = self.insulin_sensitivity * ms.state["insulin"]
        glucose_change = (self.glucose_baseline - ms.state["glucose"]) * 0.05 * dt
        glucose_change -= insulin_effect * dt
        new_glucose = max(40, ms.state["glucose"] + glucose_change)
        
        # Ketone production (increases with high glucose and low insulin)
        ketone_production = max(0, (new_glucose - 180) / 100.0) * (1.0 / (ms.state["insulin"] + 1.0))
        new_ketones = max(0, ms.state["ketones"] + (ketone_production - 0.05 * ms.state["ketones"]) * dt)
        
        # Insulin dynamics (targets normal glucose)
        insulin_target = 10.0 + max(0, (new_glucose - 100.0) * 0.2)
        insulin_change = (insulin_target - ms.state["insulin"]) * 0.1 * dt
        new_insulin = max(0, ms.state["insulin"] + insulin_change)
        
        # pH dynamics
        # Respiratory component
        pco2_change = ((40.0 - ms.state["pco2"]) * 
                      self.respiratory_compensation_rate * dt)
        new_pco2 = max(20, min(80, ms.state["pco2"] + pco2_change))
        
        # Metabolic component (HCO3)
        hco3_change = ((24.0 - ms.state["hco3"]) * 
                      self.metabolic_compensation_rate * dt)
        new_hco3 = max(10, min(40, ms.state["hco3"] + hco3_change))
        
        # Calculate new pH using Henderson-Hasselbalch
        new_ph = 6.1 + math.log10((new_hco3 / 0.03) / new_pco2)
        new_ph = max(6.8, min(7.8, new_ph))
        
        # Base excess calculation
        new_base_excess = ((new_hco3 - 24.0) + 
                         ((new_ph - 7.4) * (new_pco2 - 40.0) * 0.008))

        logger.debug(
            f"MetabolytesSolver: pH={new_ph:.2f}, "
            f"pCO2={new_pco2:.1f}, HCO3={new_hco3:.1f}, "
            f"Glucose={new_glucose:.1f}, Ketones={new_ketones:.2f}"
        )

        return MetabolytesState({
            "ph": new_ph,
            "pco2": new_pco2,
            "hco3": new_hco3,
            "po2": new_po2,
            "base_excess": new_base_excess,
            "glucose": new_glucose,
            "ketones": new_ketones,
            "insulin": new_insulin
        })