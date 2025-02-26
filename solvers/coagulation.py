import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class CoagulationState(State):
    def __init__(self, data: dict):
        # Track key coagulation parameters
        self._platelet_count = data.get("platelet_count", 150.0)  # Normal range 150-450 K/uL
        self._pt = data.get("pt", 12.0)  # Prothrombin Time, normal ~12 seconds
        self._ptt = data.get("ptt", 30.0)  # Partial Thromboplastin Time, normal 25-35 seconds
        self._fibrinogen = data.get("fibrinogen", 300.0)  # Normal range 200-400 mg/dL
        self._d_dimer = data.get("d_dimer", 0.5)  # Normal < 0.5 mg/L FEU

    @property
    def state(self) -> dict:
        return {
            "platelet_count": self._platelet_count,
            "pt": self._pt,
            "ptt": self._ptt,
            "fibrinogen": self._fibrinogen,
            "d_dimer": self._d_dimer
        }


class CoagulationSolver(Solver):
    def __init__(self, platelet_production_rate: float = 0.1,
                 platelet_decay_rate: float = 0.05,
                 pt_recovery_rate: float = 0.01,
                 ptt_recovery_rate: float = 0.02,
                 fibrinogen_production_rate: float = 0.5,
                 d_dimer_clearance_rate: float = 0.1):
        """
        Initialize coagulation solver with physiological parameters
        
        :param platelet_production_rate: Rate of platelet production per minute
        :param platelet_decay_rate: Rate of platelet consumption/decay
        :param pt_recovery_rate: Rate at which PT normalizes
        :param ptt_recovery_rate: Rate at which PTT normalizes
        :param fibrinogen_production_rate: Rate of fibrinogen production
        :param d_dimer_clearance_rate: Rate of d-dimer clearance
        """
        self._state = CoagulationState({})
        self.platelet_production_rate = platelet_production_rate
        self.platelet_decay_rate = platelet_decay_rate
        self.pt_recovery_rate = pt_recovery_rate
        self.ptt_recovery_rate = ptt_recovery_rate
        self.fibrinogen_production_rate = fibrinogen_production_rate
        self.d_dimer_clearance_rate = d_dimer_clearance_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        cs = CoagulationState(state)
        
        # Update platelets based on production and consumption
        platelet_change = (self.platelet_production_rate - 
                          self.platelet_decay_rate * cs.state["platelet_count"]) * dt
        new_platelets = max(0, cs.state["platelet_count"] + platelet_change)

        # PT tends to normalize to 12 seconds
        pt_change = self.pt_recovery_rate * (12.0 - cs.state["pt"]) * dt
        new_pt = max(0, cs.state["pt"] + pt_change)

        # PTT tends to normalize to 30 seconds
        ptt_change = self.ptt_recovery_rate * (30.0 - cs.state["ptt"]) * dt
        new_ptt = max(0, cs.state["ptt"] + ptt_change)

        # Fibrinogen production and consumption
        fibrinogen_change = (self.fibrinogen_production_rate - 
                            0.01 * cs.state["fibrinogen"]) * dt
        new_fibrinogen = max(0, cs.state["fibrinogen"] + fibrinogen_change)

        # D-dimer clearance
        d_dimer_change = -self.d_dimer_clearance_rate * cs.state["d_dimer"] * dt
        new_d_dimer = max(0, cs.state["d_dimer"] + d_dimer_change)

        logger.debug(
            f"CoagulationSolver: platelets={new_platelets:.1f}, "
            f"PT={new_pt:.1f}, PTT={new_ptt:.1f}, "
            f"fibrinogen={new_fibrinogen:.1f}, d-dimer={new_d_dimer:.2f}"
        )

        return CoagulationState({
            "platelet_count": new_platelets,
            "pt": new_pt,
            "ptt": new_ptt,
            "fibrinogen": new_fibrinogen,
            "d_dimer": new_d_dimer
        })