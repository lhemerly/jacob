import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class PressureHROxyState(State):
    """
    A more advanced approach for a single-step model of:
      - Systolic Blood Pressure stored in 'systolic_bp'
      - Diastolic Blood Pressure stored in 'diastolic_bp'
      - Mean Arterial Pressure (MAP) stored in 'blood_pressure' (derived from systolic/diastolic)
      - Heart Rate stored in 'heart_rate'
      - Oxygen Saturation stored in 'oxy_saturation'
      - Oxygen Debt stored in 'oxygen_debt' (cumulative measure of insufficient O2)
    """

    def __init__(self, data: dict):
        self._systolic = data.get("systolic_bp", 120.0)  # Systolic BP in mmHg
        self._diastolic = data.get("diastolic_bp", 80.0)  # Diastolic BP in mmHg
        # Calculate MAP from systolic and diastolic if not provided
        self._bp = data.get("blood_pressure", self._calculate_map())
        self._hr = data.get("heart_rate", 80.0)  # Beats per minute
        self._oxy = data.get("oxy_saturation", 98.0)  # Percent
        # We'll track oxygen debt as well, defaulting to 0.
        self._oxy_debt = data.get("oxygen_debt", 0.0)
    
    def _calculate_map(self):
        """Calculate Mean Arterial Pressure from systolic and diastolic values"""
        # Standard formula: MAP ≈ DBP + 1/3(SBP - DBP) or MAP = (SBP + 2*DBP)/3
        return (self._systolic + 2 * self._diastolic) / 3

    @property
    def state(self) -> dict:
        # Always ensure MAP is calculated from current systolic/diastolic values
        map_value = self._calculate_map()
        return {
            "systolic_bp": self._systolic,
            "diastolic_bp": self._diastolic,
            "blood_pressure": map_value,  # Keep for backward compatibility
            "heart_rate": self._hr,
            "oxy_saturation": self._oxy,
            "oxygen_debt": self._oxy_debt,
        }

    def __iter__(self):
        self._iter = iter(self.state.items())
        return self

    def __next__(self):
        return next(self._iter)


class PressureHROxySolver(Solver):
    """
    A more physiologically inspired solver for:
      - Systolic Blood Pressure
      - Diastolic Blood Pressure
      - Mean Arterial Pressure (MAP) - derived from systolic/diastolic
      - Heart Rate (HR)
      - Oxygen Saturation (SpO2)
      - Oxygen Debt (cumulative penalty if O2 is below a threshold)

    Using:
      - A simplified Windkessel-like approach for blood pressure.
      - A baroreflex-based approach for HR.
      - A rough approach to oxygen saturation.
      - Oxygen debt accumulation if SpO2 < 'optimal_oxy' threshold.

    Variables in global state:
      'systolic_bp'    (float) -> Systolic BP in mmHg
      'diastolic_bp'   (float) -> Diastolic BP in mmHg
      'blood_pressure' (float) -> MAP in mmHg (derived)
      'heart_rate'     (float) -> BPM
      'oxy_saturation' (float) -> O2 sat in %
      'oxygen_debt'    (float) -> Cumulative measure of insufficient oxygenation

    This solver optionally reads 'epinephrine' from the global state to mimic
    inotropic/chronotropic effects.

    All calculations: Euler step for dt in seconds.
    """

    def __init__(
        self,
        stroke_volume: float = 1.0,
        compliance: float = 1.0,
        systemic_vascular_resistance: float = 1.0,
        baroreflex_gain: float = 0.1,
        map_setpoint: float = 90.0,
        pulse_pressure_factor: float = 0.4,  # Factor to determine pulse pressure
        oxy_recovery_rate: float = 0.01,
        oxy_drop_rate: float = 0.02,
        epi_hr_factor: float = 0.5,
        epi_bp_factor: float = 0.3,  # How strongly epinephrine affects BP
        min_oxy: float = 75.0,
        max_oxy: float = 100.0,
        min_systolic: float = 40.0,
        max_systolic: float = 220.0,
        min_diastolic: float = 20.0,
        max_diastolic: float = 120.0,
        dt_unit_in_seconds: bool = True,
        initial_state: PressureHROxyState = PressureHROxyState({}),
        # Oxygen debt related:
        optimal_oxy: float = 95.0,
        oxy_debt_accum_factor: float = 0.1,
    ):
        """
        :param stroke_volume: (mL/beat) For CO = HR * stroke_volume (very simplified).
        :param compliance: Arterial compliance in windkessel eq.
        :param systemic_vascular_resistance: The effective afterload.
        :param baroreflex_gain: How strongly HR is adjusted based on difference from map_setpoint.
        :param map_setpoint: The target MAP for baroreflex.
        :param pulse_pressure_factor: Determines ratio of systolic/diastolic changes.
        :param oxy_recovery_rate: Rate at which O2 sat returns to normal if perfusion is adequate.
        :param oxy_drop_rate: Rate at which O2 sat drops if MAP is too low.
        :param epi_hr_factor: How strongly epinephrine from global state raises HR.
        :param epi_bp_factor: How strongly epinephrine from global state raises BP.
        :param min_oxy, max_oxy: Hard clamp on oxygen saturation.
        :param min_systolic, max_systolic: Hard clamp on systolic BP.
        :param min_diastolic, max_diastolic: Hard clamp on diastolic BP.
        :param dt_unit_in_seconds: If True, dt is in seconds. If your Master uses minutes or hours, you can set this accordingly.
        :param initial_state: Initial state object.
        :param optimal_oxy: O2 threshold above which no oxygen debt accumulates.
        :param oxy_debt_accum_factor: scaling factor for how fast oxygen debt accumulates below optimal O2.
        """
        # Store parameters
        self.stroke_volume = stroke_volume
        self.compliance = compliance
        self.svr = systemic_vascular_resistance
        self.baro_gain = baroreflex_gain
        self.map_setpoint = map_setpoint
        self.pulse_pressure_factor = pulse_pressure_factor
        self.oxy_recovery_rate = oxy_recovery_rate
        self.oxy_drop_rate = oxy_drop_rate
        self.epi_hr_factor = epi_hr_factor
        self.epi_bp_factor = epi_bp_factor
        self.min_oxy = min_oxy
        self.max_oxy = max_oxy
        self.min_systolic = min_systolic
        self.max_systolic = max_systolic
        self.min_diastolic = min_diastolic
        self.max_diastolic = max_diastolic
        self.dt_unit_in_seconds = dt_unit_in_seconds
        self._state = initial_state

        # Oxygen debt parameters
        self.optimal_oxy = optimal_oxy
        self.oxy_debt_accum_factor = oxy_debt_accum_factor

    @property
    def state(self):
        # Return the dictionary from the state object, not the state object itself
        return self._state.state

    @state.setter
    def state(self, state):
        self._state = state

    def solve(self, state: dict, dt: float) -> State:
        # Convert to typed state object
        ps = PressureHROxyState(state)
        systolic_old = ps.state["systolic_bp"]
        diastolic_old = ps.state["diastolic_bp"]
        map_old = ps.state["blood_pressure"]
        hr_old = ps.state["heart_rate"]
        oxy_old = ps.state["oxy_saturation"]
        debt_old = ps.state["oxygen_debt"]

        # Retrieve epinephrine if it exists
        epi = 0.0
        if "epinephrine" in state:
            epi = state["epinephrine"]

        # dt conversion to seconds if needed
        dt_seconds = dt if self.dt_unit_in_seconds else dt * 60.0

        """
        1) Blood Pressure update (Windkessel-like with separate systolic and diastolic components)
           First calculate change in MAP
        """
        cardiac_output = hr_old * self.stroke_volume  # simplistic
        dmap_dt = (cardiac_output - (map_old / self.svr)) / self.compliance
        
        # Add epinephrine effect on blood pressure (vasoconstriction)
        dmap_dt += epi * self.epi_bp_factor
        
        # Calculate base MAP change
        map_change = dmap_dt * dt_seconds
        
        # Distribute the change between systolic and diastolic
        # Systolic changes more than diastolic based on pulse_pressure_factor
        systolic_change = map_change * (1 + self.pulse_pressure_factor)
        diastolic_change = map_change * (1 - self.pulse_pressure_factor)
        
        # Apply changes
        systolic_new = systolic_old + systolic_change
        diastolic_new = diastolic_old + diastolic_change
        
        # Enforce limits on systolic and diastolic
        systolic_new = max(min(systolic_new, self.max_systolic), self.min_systolic)
        diastolic_new = max(min(diastolic_new, self.max_diastolic), self.min_diastolic)
        
        # Ensure diastolic is always less than systolic
        if diastolic_new >= systolic_new:
            diastolic_new = systolic_new - 10  # Minimum 10 mmHg pulse pressure
        
        # Calculate new MAP based on new systolic and diastolic
        map_new = (systolic_new + 2 * diastolic_new) / 3

        """
        2) Heart Rate update (Baroreflex + Epi effect)
           dHR/dt = baro_gain*(map_setpoint - MAP_old) + epi_hr_factor*epi
        """
        dhr_dt = self.baro_gain * (self.map_setpoint - map_old) + (
            self.epi_hr_factor * epi
        )
        hr_new = hr_old + dhr_dt * dt_seconds
        if hr_new < 0:
            hr_new = 0

        """
        3) Oxygen saturation update
           If MAP < 60, we drop O2. If MAP >= 60, we recover.
        """
        map_threshold = 60.0
        if map_new >= map_threshold:
            oxy_new = (
                oxy_old + self.oxy_recovery_rate * (self.max_oxy - oxy_old) * dt_seconds
            )
        else:
            drop_factor = 1.0 + max(0, (map_threshold - map_new) / map_threshold)
            oxy_new = oxy_old - (self.oxy_drop_rate * drop_factor) * dt_seconds
        if oxy_new < self.min_oxy:
            oxy_new = self.min_oxy
        elif oxy_new > self.max_oxy:
            oxy_new = self.max_oxy

        """
        4) Oxygen Debt update
           If oxy_new < optimal_oxy, accumulate debt.
           e.g., dDebt/dt = (optimal_oxy - oxy_new) * factor
        """
        debt_new = debt_old
        if oxy_new < self.optimal_oxy:
            debt_new += (
                (self.optimal_oxy - oxy_new) * self.oxy_debt_accum_factor * dt_seconds
            )

        logger.debug(
            f"PressureHROxySolver:\n"
            f"  Systolic BP: {systolic_old:.2f} -> {systolic_new:.2f}\n"
            f"  Diastolic BP: {diastolic_old:.2f} -> {diastolic_new:.2f}\n"
            f"  MAP: {map_old:.2f} -> {map_new:.2f}, dMAP/dt={dmap_dt:.3f}\n"
            f"  HR: {hr_old:.1f} -> {hr_new:.1f}, dHR/dt={dhr_dt:.3f}, Epi={epi:.2f}\n"
            f"  O2: {oxy_old:.1f} -> {oxy_new:.1f}\n"
            f"  O2 Debt: {debt_old:.2f} -> {debt_new:.2f}"
        )

        return PressureHROxyState(
            {
                "systolic_bp": systolic_new,
                "diastolic_bp": diastolic_new,
                "heart_rate": hr_new,
                "oxy_saturation": oxy_new,
                "oxygen_debt": debt_new,
            }
        )
