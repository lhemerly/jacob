import logging
import taichi as ti
from classes import Coupler

logger = logging.getLogger(__name__)

@ti.data_oriented
class CoagulationFluidCoupler(Coupler):
    """
    Coupler to handle interactions between coagulation parameters, fluid status, and hemodynamics.
    Models how bleeding affects fluid volume and how coagulation parameters affect bleeding risk.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def input_keys(self) -> list:
        return ["platelets", "inr", "ptt", "fibrinogen", "bleeding_rate", "fluid_volume", "hemoglobin"]
    
    @property
    def output_keys(self) -> list:
        return ["bleeding_rate", "fluid_volume", "hemoglobin"]
    
    @ti.kernel
    def _calculate_coagulation_effects(self, platelets: float, inr: float, ptt: float, 
                                      fibrinogen: float, current_bleeding: float, 
                                      fluid_volume: float, hemoglobin: float) -> ti.math.vec3:
        """
        Taichi kernel to calculate effects between coagulation and fluid parameters
        Returns [bleeding_rate_effect, fluid_volume_change, hemoglobin_change]
        """
        # References for normal values
        normal_platelets = 250.0  # x10^9/L
        normal_inr = 1.0
        normal_ptt = 30.0  # seconds
        normal_fibrinogen = 300.0  # mg/dL
        
        # Ensure parameters are in physiological ranges
        safe_platelets = ti.min(600.0, ti.max(10.0, platelets))
        safe_inr = ti.min(10.0, ti.max(0.5, inr))
        safe_ptt = ti.min(120.0, ti.max(15.0, ptt))
        safe_fibrinogen = ti.min(800.0, ti.max(50.0, fibrinogen))
        
        # Calculate coagulation risk factors
        platelet_factor = ti.min(2.0, normal_platelets / ti.max(20.0, safe_platelets))
        inr_factor = safe_inr / normal_inr
        ptt_factor = safe_ptt / normal_ptt
        fibrinogen_factor = ti.min(2.0, normal_fibrinogen / ti.max(50.0, safe_fibrinogen))
        
        # Combine factors - higher value means worse coagulation
        coagulation_risk = (platelet_factor * 0.3 + 
                           inr_factor * 0.3 + 
                           ptt_factor * 0.2 + 
                           fibrinogen_factor * 0.2)
        
        # Calculate bleeding rate modification based on coagulation risk
        bleeding_effect = 0.0
        if coagulation_risk > 1.0:
            bleeding_effect = ti.min(5.0, (coagulation_risk - 1.0) * 10.0)  # Cap at 5.0 units
        
        # Calculate fluid volume loss due to bleeding
        # Cap the bleeding-based fluid loss
        fluid_loss = ti.min(300.0, current_bleeding * 5.0)  # ml per minute per bleeding unit, max 300ml
        
        # Calculate hemoglobin drop from bleeding
        hgb_drop = 0.0
        if fluid_volume > 0 and current_bleeding > 0:
            # Proportional to bleeding rate and inversely to fluid volume
            # Cap the hemoglobin drop
            hgb_drop = ti.min(0.5, current_bleeding * 0.05 * (3000.0 / ti.max(1000.0, fluid_volume)))
        
        return ti.math.vec3(bleeding_effect, fluid_loss, hgb_drop)
    
    def couple(self, state: dict, dt: float) -> dict:
        """
        Process interactions between coagulation status, bleeding, and fluid balance
        """
        # Extract values with defaults
        platelets = state.get("platelets", 250.0)
        inr = state.get("inr", 1.0)
        ptt = state.get("ptt", 30.0)
        fibrinogen = state.get("fibrinogen", 300.0)
        bleeding_rate = state.get("bleeding_rate", 0.0)
        fluid_volume = state.get("fluid_volume", 2000.0)
        hemoglobin = state.get("hemoglobin", 14.0)
        
        # Only apply coupling if there's abnormal coagulation or active bleeding
        if (abs(platelets - 250.0) < 10.0 and 
            abs(inr - 1.0) < 0.1 and 
            abs(ptt - 30.0) < 2.0 and
            abs(fibrinogen - 300.0) < 20.0 and
            bleeding_rate < 0.1):
            return {}
        
        # Apply fractional time scaling to avoid excessive changes
        time_factor = min(0.1, dt / 60.0)  # Scale by time, maxing at 6 seconds equivalent
        
        # Calculate effects using Taichi kernel
        effects = self._calculate_coagulation_effects(
            platelets, inr, ptt, fibrinogen, bleeding_rate, fluid_volume, hemoglobin)
        
        # Apply time scaling for changes
        bleeding_effect = float(effects[0]) * time_factor
        fluid_loss = float(effects[1]) * time_factor
        hgb_drop = float(effects[2]) * time_factor
        
        # Calculate new values
        new_bleeding = bleeding_rate
        # Only increase bleeding if coagulation is abnormal
        if bleeding_effect > 0:
            new_bleeding = min(15.0, new_bleeding + bleeding_effect)  # Cap at 15 units
        # Natural decrease in bleeding rate
        elif bleeding_rate > 0:
            new_bleeding = max(0, bleeding_rate - 0.1 * time_factor)
        
        # Cap the minimum fluid volume at 500ml and ensure it doesn't drop too quickly
        new_fluid_volume = max(500.0, fluid_volume - min(100.0, fluid_loss))
        
        # Ensure hemoglobin stays in physiological range and doesn't drop too quickly
        new_hemoglobin = max(3.0, hemoglobin - min(0.5, hgb_drop))
        
        # Only apply significant changes
        result = {}
        if abs(new_bleeding - bleeding_rate) > 0.05:
            result["bleeding_rate"] = new_bleeding
            
        if abs(new_fluid_volume - fluid_volume) > 1.0:
            result["fluid_volume"] = new_fluid_volume
            
        if abs(new_hemoglobin - hemoglobin) > 0.05:
            result["hemoglobin"] = new_hemoglobin
            
        if result:
            self.logger.info(f"Coupling: Coagulation affecting fluid status - "
                           f"Bleeding: {bleeding_rate:.1f} -> {result.get('bleeding_rate', bleeding_rate):.1f}, "
                           f"Fluid: {fluid_volume:.0f} -> {result.get('fluid_volume', fluid_volume):.0f} ml")
            
        return result