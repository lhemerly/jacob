import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class FluidsState(State):
    def __init__(self, data: dict):
        # We'll track total fluid volume in ml (just as an example)
        self._fluid_volume = data.get(
            "fluid_volume", 2000.0
        )  # e.g. 2000 ml as a baseline
        # Also track net fluid gain or loss each step.

    @property
    def state(self) -> dict:
        return {"fluid_volume": self._fluid_volume}


class FluidsSolver(Solver):
    def __init__(self):
        self._state = FluidsState({})

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        fs = FluidsState(state)
        vol_old = fs.state["fluid_volume"]

        # Example logic: fluid volume slowly decreases by 1 ml/min plus some factor * dt
        # We'll assume dt is in minutes for demonstration.
        # Also, we might have external infusion from actions, but let's do a simple baseline.

        baseline_loss = 1.0 * dt  # 1 ml per minute
        new_vol = vol_old - baseline_loss

        # clamp for safety
        if new_vol < 0:
            new_vol = 0

        logger.debug(f"FluidsSolver: volume from {vol_old:.2f} to {new_vol:.2f}")

        return FluidsState({"fluid_volume": new_vol})
