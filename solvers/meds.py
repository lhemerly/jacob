import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class MedsState(State):
    def __init__(self, data: dict):
        # We'll track some medication levels, e.g. epinephrine.
        self._epinephrine = data.get("epinephrine", 0.0)
        # Could add more meds as needed.

    @property
    def state(self) -> dict:
        return {"epinephrine": self._epinephrine}


class MedsSolver(Solver):
    def __init__(self):
        # This solver handles these keys:
        self._state = MedsState({})

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        # Convert dict to typed state
        ms = MedsState(state)
        epi_old = ms.state["epinephrine"]

        # Example model:
        # We have an infusion that might be set externally as an "action" (or could be an internal rule).
        # For now, let's do a simple elimination model: epi_new = epi_old - 0.05 * epi_old * dt
        # so half-life type behavior.
        elimination_rate = 0.05
        new_epi = epi_old - elimination_rate * epi_old * dt
        if new_epi < 0:
            new_epi = 0

        logger.debug(f"MedsSolver: epinephrine from {epi_old:.2f} to {new_epi:.2f}")

        return MedsState({"epinephrine": new_epi})
