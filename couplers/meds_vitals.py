import logging
from classes import Coupler
import taichi as ti

logger = logging.getLogger(__name__)

@ti.data_oriented
class MedsVitalsCoupler(Coupler):
    """
    Coupler to handle interactions between medications (like epinephrine) 
    and vital signs (heart rate, blood pressure).
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def input_keys(self) -> list:
        # Keys we need to read to determine interactions
        return ["epinephrine", "heart_rate", "blood_pressure", "fluid_volume"]
    
    @property
    def output_keys(self) -> list:
        # Keys we'll potentially modify based on interactions
        return ["heart_rate", "blood_pressure"]
    
    @ti.kernel
    def _calculate_effects(self, epinephrine: float, fluid_volume: float) -> ti.math.vec2:
        """
        Taichi kernel to calculate medication effects in parallel
        Returns [heart_rate_change, blood_pressure_change]
        """
        hr_effect = 0.0
        bp_effect = 0.0
        
        # Epinephrine increases heart rate and blood pressure
        if epinephrine > 0:
            hr_effect = epinephrine * 2.0  # Each unit increases HR by 2 bpm
            bp_effect = epinephrine * 0.5  # Each unit increases BP by 0.5 mmHg
            
            # Effect is reduced if fluid volume is low
            if fluid_volume < 1500:
                factor = ti.max(0.5, fluid_volume / 2000.0)
                hr_effect *= factor
                bp_effect *= factor
        
        return ti.math.vec2(hr_effect, bp_effect)
    
    def couple(self, state: dict, dt: float) -> dict:
        """
        Process interactions between medications and vital signs
        """
        # Extract values from state with defaults if missing
        epinephrine = state.get("epinephrine", 0.0)
        heart_rate = state.get("heart_rate", 70.0)
        blood_pressure = state.get("blood_pressure", 120.0)
        fluid_volume = state.get("fluid_volume", 2000.0)
        
        # Calculate effects using Taichi kernel
        effects = self._calculate_effects(epinephrine, fluid_volume)
        hr_effect = effects[0]
        bp_effect = effects[1]
        
        # Apply changes with time factor
        new_hr = heart_rate + hr_effect * dt
        new_bp = blood_pressure + bp_effect * dt
        
        # Log significant changes
        if abs(new_hr - heart_rate) > 1.0 or abs(new_bp - blood_pressure) > 1.0:
            self.logger.info(f"Coupling: Meds affecting vitals - HR: {heart_rate:.1f} -> {new_hr:.1f}, BP: {blood_pressure:.1f} -> {new_bp:.1f}")
        
        # Return the updated values
        return {
            "heart_rate": new_hr,
            "blood_pressure": new_bp
        }