import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class DrainsState(State):
    def __init__(self, data: dict):
        # Track multiple drains and their outputs
        self._chest_tube_output = data.get("chest_tube_output", 0.0)  # mL/hr
        self._jp_drain_output = data.get("jp_drain_output", 0.0)  # mL/hr
        self._ng_tube_output = data.get("ng_tube_output", 0.0)  # mL/hr
        self._total_output = data.get("total_drain_output", 0.0)  # Total accumulated output

    @property
    def state(self) -> dict:
        return {
            "chest_tube_output": self._chest_tube_output,
            "jp_drain_output": self._jp_drain_output,
            "ng_tube_output": self._ng_tube_output,
            "total_drain_output": self._total_output
        }


class DrainsSolver(Solver):
    def __init__(self, 
                chest_tube_base_rate: float = 10.0,  # mL/hr
                jp_drain_base_rate: float = 5.0,     # mL/hr
                ng_tube_base_rate: float = 20.0):    # mL/hr
        """
        Initialize drains solver with baseline drainage rates
        
        :param chest_tube_base_rate: Baseline chest tube output rate in mL/hr
        :param jp_drain_base_rate: Baseline JP drain output rate in mL/hr
        :param ng_tube_base_rate: Baseline NG tube output rate in mL/hr
        """
        self._state = DrainsState({})
        self.chest_tube_base_rate = chest_tube_base_rate
        self.jp_drain_base_rate = jp_drain_base_rate
        self.ng_tube_base_rate = ng_tube_base_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ds = DrainsState(state)
        
        # Calculate new outputs (dt is in minutes, convert rates from per hour)
        hour_fraction = dt / 60.0
        
        # Chest tube output (varies with underlying condition)
        new_chest = self.chest_tube_base_rate * hour_fraction
        
        # JP drain output (decreases over time)
        new_jp = max(0, self.jp_drain_base_rate * 
                    (1.0 - ds.state["total_drain_output"] / 5000.0)) * hour_fraction
        
        # NG tube output (constant unless modified by actions)
        new_ng = self.ng_tube_base_rate * hour_fraction
        
        # Update total accumulated output
        new_total = (ds.state["total_drain_output"] + 
                    new_chest + new_jp + new_ng)

        logger.debug(
            f"DrainsSolver: chest={new_chest:.1f}, "
            f"JP={new_jp:.1f}, NG={new_ng:.1f}, "
            f"total={new_total:.1f}"
        )

        return DrainsState({
            "chest_tube_output": new_chest,
            "jp_drain_output": new_jp,
            "ng_tube_output": new_ng,
            "total_drain_output": new_total
        })