import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class HemogramState(State):
    def __init__(self, data: dict):
        # Complete blood count parameters
        self._hemoglobin = data.get("hemoglobin", 14.0)  # g/dL (normal 12-16)
        self._hematocrit = data.get("hematocrit", 42.0)  # % (normal 36-48)
        self._wbc = data.get("wbc", 7.5)  # K/uL (normal 4.5-11.0)
        self._neutrophils = data.get("neutrophils", 60.0)  # % (normal 40-70)
        self._lymphocytes = data.get("lymphocytes", 30.0)  # % (normal 20-40)
        self._monocytes = data.get("monocytes", 7.0)  # % (normal 2-8)
        self._eosinophils = data.get("eosinophils", 2.0)  # % (normal 1-4)
        self._basophils = data.get("basophils", 1.0)  # % (normal 0.5-1)
        self._rbc = data.get("rbc", 5.0)  # M/uL (normal 4.5-5.9)

    @property
    def state(self) -> dict:
        return {
            "hemoglobin": self._hemoglobin,
            "hematocrit": self._hematocrit,
            "wbc": self._wbc,
            "neutrophils": self._neutrophils,
            "lymphocytes": self._lymphocytes,
            "monocytes": self._monocytes,
            "eosinophils": self._eosinophils,
            "basophils": self._basophils,
            "rbc": self._rbc
        }


class HemogramSolver(Solver):
    def __init__(self,
                 hgb_production_rate: float = 0.01,
                 wbc_response_rate: float = 0.05,
                 rbc_lifespan_days: float = 120.0):
        """
        Initialize hemogram solver with physiological parameters
        
        :param hgb_production_rate: Rate of hemoglobin production
        :param wbc_response_rate: Rate of WBC response to infection
        :param rbc_lifespan_days: Average RBC lifespan in days
        """
        self._state = HemogramState({})
        self.hgb_production_rate = hgb_production_rate
        self.wbc_response_rate = wbc_response_rate
        self.rbc_decay_rate = 1.0 / (rbc_lifespan_days * 24 * 60)  # Convert to per minute

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        hs = HemogramState(state)
        
        # Get infection level if available (affects WBC response)
        infection_level = state.get("infection_level", 0.0)
        
        # Hemoglobin and RBC changes
        # Production balanced with decay
        hgb_change = (self.hgb_production_rate - 
                     self.rbc_decay_rate * hs.state["hemoglobin"]) * dt
        new_hgb = max(0, hs.state["hemoglobin"] + hgb_change)
        
        # Hematocrit follows hemoglobin (roughly 3:1 ratio)
        new_hct = new_hgb * 3.0
        
        # RBC count changes
        rbc_change = (self.hgb_production_rate/3.0 - 
                     self.rbc_decay_rate * hs.state["rbc"]) * dt
        new_rbc = max(0, hs.state["rbc"] + rbc_change)
        
        # WBC response (influenced by infection)
        wbc_target = 7.5 + (infection_level * 0.1)  # Increases with infection
        wbc_change = self.wbc_response_rate * (wbc_target - hs.state["wbc"]) * dt
        new_wbc = max(0, hs.state["wbc"] + wbc_change)
        
        # Differential changes based on infection
        # Neutrophils and lymphocytes respond most to infection
        new_neutrophils = min(85, hs.state["neutrophils"] + 
                            (infection_level * 0.02 * dt))
        new_lymphocytes = max(10, hs.state["lymphocytes"] - 
                            (infection_level * 0.01 * dt))
        
        # Other white cells change more slowly
        new_monocytes = max(2, min(12, hs.state["monocytes"] + 
                          (infection_level * 0.001 * dt)))
        new_eosinophils = max(0, min(8, hs.state["eosinophils"] + 
                            (infection_level * 0.0005 * dt)))
        new_basophils = max(0, min(2, hs.state["basophils"]))  # Relatively stable

        logger.debug(
            f"HemogramSolver: Hgb={new_hgb:.1f}, Hct={new_hct:.1f}, "
            f"WBC={new_wbc:.1f}, Neutrophils={new_neutrophils:.1f}%, "
            f"RBC={new_rbc:.1f}"
        )

        return HemogramState({
            "hemoglobin": new_hgb,
            "hematocrit": new_hct,
            "wbc": new_wbc,
            "neutrophils": new_neutrophils,
            "lymphocytes": new_lymphocytes,
            "monocytes": new_monocytes,
            "eosinophils": new_eosinophils,
            "basophils": new_basophils,
            "rbc": new_rbc
        })