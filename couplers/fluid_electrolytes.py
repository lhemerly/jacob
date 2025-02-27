import logging
import taichi as ti
from classes import Coupler

logger = logging.getLogger(__name__)

@ti.data_oriented
class FluidElectrolyteCoupler(Coupler):
    """
    Coupler to handle interactions between fluid balance and electrolytes.
    This coupler monitors fluid volume and affects electrolyte concentrations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def input_keys(self) -> list:
        return ["fluid_volume", "sodium", "potassium"]
    
    @property
    def output_keys(self) -> list:
        return ["sodium", "potassium"]
    
    @property
    def initial_state(self) -> dict:
        """
        Default initial values for state variables needed by this coupler.
        """
        return {
            "fluid_volume": 2000.0,  # Standard fluid volume in ml
            "sodium": 140.0,         # Normal sodium level in mEq/L
            "potassium": 4.0,        # Normal potassium level in mEq/L
        }
    
    @ti.func
    def _is_valid_value(self, value: float) -> bool:
        """Check if a value is valid (not NaN or infinite)"""
        return not (ti.math.isnan(value) or ti.math.isinf(value))
    
    @ti.func
    def _dilute_electrolyte(self, concentration: float, fluid_volume: float, baseline_volume: float) -> float:
        """
        Calculate how electrolyte concentration changes with fluid volume
        """
        # Add small epsilon to prevent division by zero
        epsilon = 1e-6
        safe_volume = ti.max(fluid_volume, epsilon)
        safe_concentration = ti.select(concentration > 0, concentration, epsilon)
        # Simple dilution model: C1*V1 = C2*V2 -> C2 = C1*V1/V2
        result = safe_concentration * (baseline_volume / safe_volume)
        return ti.select(self._is_valid_value(result), result, concentration)
    
    @ti.kernel
    def _calculate_electrolyte_effects(self, 
                                      sodium: float,
                                      potassium: float, 
                                      fluid_volume: float) -> ti.math.vec2:
        """
        Taichi kernel to calculate electrolyte changes based on fluid balance
        Returns [new_sodium, new_potassium]
        """
        baseline_volume = 2000.0  # Standard fluid volume in ml
        
        # Ensure input values are valid
        safe_fluid_volume = ti.select(fluid_volume > 0, fluid_volume, baseline_volume)
        safe_sodium = ti.select(sodium > 0, sodium, 140.0)
        safe_potassium = ti.select(potassium > 0, potassium, 4.0)
        
        # Calculate new electrolyte values based on fluid dilution
        new_sodium = self._dilute_electrolyte(safe_sodium, safe_fluid_volume, baseline_volume)
        new_potassium = self._dilute_electrolyte(safe_potassium, safe_fluid_volume, baseline_volume)
        
        # Return original values if new values are invalid
        result_sodium = ti.select(self._is_valid_value(new_sodium), new_sodium, safe_sodium)
        result_potassium = ti.select(self._is_valid_value(new_potassium), new_potassium, safe_potassium)
        
        return ti.math.vec2(result_sodium, result_potassium)
    
    def couple(self, state: dict, dt: float) -> dict:
        """
        Process interactions between fluid volume and electrolytes
        """
        # Extract values from state with defaults if missing
        fluid_volume = state.get("fluid_volume", 2000.0)
        sodium = state.get("sodium", 140.0)  # Normal sodium: 135-145 mEq/L
        potassium = state.get("potassium", 4.0)  # Normal potassium: 3.5-5.0 mEq/L
        
        # No need for calculations if fluid volume is at baseline
        if abs(fluid_volume - 2000.0) < 50.0:
            return {}
        
        # Calculate effects using Taichi kernel
        new_values = self._calculate_electrolyte_effects(sodium, potassium, fluid_volume)
        new_sodium = float(new_values[0])
        new_potassium = float(new_values[1])
        
        # Only apply significant changes if values are valid
        result = {}
        if abs(new_sodium - sodium) > 0.1:
            result["sodium"] = new_sodium
            
        if abs(new_potassium - potassium) > 0.1:
            result["potassium"] = new_potassium
            
        if result:
            self.logger.info(f"Coupling: Fluid affecting electrolytes - "
                            f"Na: {sodium:.1f} -> {new_sodium:.1f}, "
                            f"K: {potassium:.1f} -> {new_potassium:.1f}")
            
        return result