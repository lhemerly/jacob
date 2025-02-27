import logging
import taichi as ti
from classes import Coupler

logger = logging.getLogger(__name__)

@ti.data_oriented
class FeverMetabolicCoupler(Coupler):
    """
    Coupler to handle interactions between fever and metabolic processes.
    Fever increases heart rate, oxygen consumption, and affects metabolic rates.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def input_keys(self) -> list:
        return ["temperature", "heart_rate", "oxygen_saturation", "metabolic_rate"]
    
    @property
    def output_keys(self) -> list:
        return ["heart_rate", "oxygen_saturation", "metabolic_rate"]
    
    @property
    def initial_state(self) -> dict:
        """
        Default initial values for state variables needed by this coupler.
        """
        return {
            "temperature": 37.0,       # Normal body temperature in °C
            "heart_rate": 70.0,        # Normal resting heart rate in bpm
            "oxygen_saturation": 98.0, # Normal oxygen saturation in %
            "metabolic_rate": 1.0      # Baseline metabolic rate (unitless factor)
        }
    
    @ti.kernel
    def _calculate_fever_effects(self, temperature: float, heart_rate: float, 
                                oxygen_saturation: float, metabolic_rate: float) -> ti.math.vec3:
        """
        Taichi kernel to calculate effects of fever on metabolism and vital signs
        Returns [new_heart_rate, new_oxygen_saturation, new_metabolic_rate]
        """
        # Normal reference values
        normal_temp = 37.0
        
        # Clamp temperature to physiological range (30-43°C)
        safe_temp = ti.min(43.0, ti.max(30.0, temperature))
        temp_delta = safe_temp - normal_temp
        
        # Heart rate increases ~10 bpm for each °C increase in temperature
        # Capped to prevent unrealistic values
        hr_change = ti.min(40.0, temp_delta * 10.0)  # Cap at 40 bpm increase
        new_heart_rate = ti.min(180.0, heart_rate + hr_change)  # Cap at 180 bpm
        
        # Oxygen saturation decreases slightly with fever due to increased consumption
        oxy_change = ti.max(-10.0, temp_delta * -0.3)  # Limit decrease to max 10%
        new_oxygen_saturation = ti.max(85.0, oxygen_saturation + oxy_change)  # Minimum 85%
        
        # Metabolic rate increases with fever (about 7% per °C)
        # Capped at reasonable maximum
        metabolic_factor = ti.min(2.0, 1.0 + (temp_delta * 0.07))  # Cap at 2x
        new_metabolic_rate = ti.min(3.0, metabolic_rate * metabolic_factor)  # Cap at 3x
        
        return ti.math.vec3(new_heart_rate, new_oxygen_saturation, new_metabolic_rate)
    
    def couple(self, state: dict, dt: float) -> dict:
        """
        Process interactions between fever and related physiological parameters
        """
        # Extract values with defaults
        temperature = state.get("temperature", 37.0)
        heart_rate = state.get("heart_rate", 70.0)
        oxygen_saturation = state.get("oxygen_saturation", 98.0)
        metabolic_rate = state.get("metabolic_rate", 1.0)
        
        # Only apply coupling if temperature is abnormal
        if abs(temperature - 37.0) < 0.2:
            return {}
            
        # Apply fractional time scaling (avoid excessive changes in large dt)
        time_factor = min(0.1, dt / 60.0)  # Scale by time, maxing at 6 seconds equivalent
        
        # Calculate effects using Taichi kernel
        effects = self._calculate_fever_effects(temperature, heart_rate, oxygen_saturation, metabolic_rate)
        
        # Apply time factor to changes
        current_heart_rate = heart_rate
        current_oxygen_saturation = oxygen_saturation
        current_metabolic_rate = metabolic_rate
        
        target_heart_rate = float(effects[0])
        target_oxygen_saturation = float(effects[1])
        target_metabolic_rate = float(effects[2])
        
        # Interpolate toward target values based on time factor
        new_heart_rate = current_heart_rate + (target_heart_rate - current_heart_rate) * time_factor
        new_oxygen_saturation = current_oxygen_saturation + (target_oxygen_saturation - current_oxygen_saturation) * time_factor
        new_metabolic_rate = current_metabolic_rate + (target_metabolic_rate - current_metabolic_rate) * time_factor
        
        # Only apply significant changes
        result = {}
        if abs(new_heart_rate - heart_rate) > 0.5:
            result["heart_rate"] = new_heart_rate
            
        if abs(new_oxygen_saturation - oxygen_saturation) > 0.1:
            # Ensure oxygen saturation stays in physiological range
            result["oxygen_saturation"] = max(70.0, min(100.0, new_oxygen_saturation))
            
        if abs(new_metabolic_rate - metabolic_rate) > 0.01:
            result["metabolic_rate"] = new_metabolic_rate
            
        if result:
            self.logger.info(f"Coupling: Fever affecting metabolism - "
                           f"HR: {heart_rate:.1f} -> {result.get('heart_rate', heart_rate):.1f}, "
                           f"O2: {oxygen_saturation:.1f}% -> {result.get('oxygen_saturation', oxygen_saturation):.1f}%")
            
        return result