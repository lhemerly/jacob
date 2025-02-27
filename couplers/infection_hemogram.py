import logging
import taichi as ti
from classes import Coupler

logger = logging.getLogger(__name__)

@ti.data_oriented
class InfectionHemogramCoupler(Coupler):
    """
    Coupler to handle interactions between infection severity (TSS) and blood parameters.
    Simulates how infections affect white blood cell counts, platelets, and other hemogram values.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def input_keys(self) -> list:
        return ["tss_severity", "infection_level", "wbc", "platelets", "hemoglobin"]
    
    @property
    def output_keys(self) -> list:
        return ["wbc", "platelets", "hemoglobin"]
    
    @ti.kernel
    def _calculate_infection_effects(self, tss_severity: float, infection_level: float, 
                                    wbc: float, platelets: float, hemoglobin: float) -> ti.math.vec3:
        """
        Taichi kernel to calculate effects of infection on blood parameters
        Returns [new_wbc, new_platelets, new_hemoglobin]
        """
        # Cap infection metrics to physiological ranges
        safe_tss = ti.min(100.0, ti.max(0.0, tss_severity))
        safe_infection = ti.min(100.0, ti.max(0.0, infection_level))
        
        # Combine infection metrics (use the higher of the two)
        infection_severity = ti.max(safe_tss, safe_infection)
        
        # Early infection: WBC increases (immune response)
        # Severe infection: WBC may decrease (immune exhaustion)
        wbc_factor = 0.0
        if infection_severity < 50.0:
            # Mild to moderate infection - WBC increases
            wbc_factor = 1.0 + (infection_severity / 25.0)  # Up to 3x increase
        else:
            # Severe infection with immune dysfunction - WBC may decrease
            severity_above_threshold = infection_severity - 50.0
            wbc_factor = 3.0 - (severity_above_threshold / 16.7)  # Can go down to 0.25x
        
        # Clamp WBC to physiological range
        new_wbc = ti.min(30.0, wbc * wbc_factor)  # Cap at 30 K/μL
        
        # Platelets typically decrease with severe infection (consumption)
        platelet_factor = 1.0 - (infection_severity / 200.0)  # Up to 50% decrease
        new_platelets = ti.min(600.0, platelets * ti.max(0.5, platelet_factor))  # Cap at 600 K/μL
        
        # Hemoglobin decreases with severe/prolonged infection (anemia of inflammation)
        hgb_factor = 1.0 - (infection_severity / 250.0)  # Up to 40% decrease
        new_hemoglobin = ti.min(18.0, hemoglobin * ti.max(0.6, hgb_factor))  # Cap at 18 g/dL
        
        return ti.math.vec3(new_wbc, new_platelets, new_hemoglobin)
    
    def couple(self, state: dict, dt: float) -> dict:
        """
        Process interactions between infection severity and blood parameters
        """
        # Extract values with defaults
        tss_severity = state.get("tss_severity", 0.0)
        infection_level = state.get("infection_level", 0.0)
        wbc = state.get("wbc", 7.5)  # 4-11 x10^9/L normal range
        platelets = state.get("platelets", 250.0)  # 150-450 x10^9/L normal
        hemoglobin = state.get("hemoglobin", 14.0)  # 12-16 g/dL normal
        
        # Only apply coupling if infection is present
        if tss_severity < 1.0 and infection_level < 1.0:
            return {}
        
        # Calculate effects using Taichi kernel
        effects = self._calculate_infection_effects(tss_severity, infection_level, wbc, platelets, hemoglobin)
        
        # Extract values and apply time scaling for gradual changes
        # Use a smaller time factor to avoid drastic changes
        time_factor = min(0.05, dt / 3600.0)  # Scale by time, maxing at 0.05 (5% per step)
        
        current_wbc = wbc
        current_platelets = platelets
        current_hemoglobin = hemoglobin
        
        target_wbc = float(effects[0])
        target_platelets = float(effects[1])
        target_hemoglobin = float(effects[2])
        
        # Interpolate toward target values based on time factor
        new_wbc = current_wbc + (target_wbc - current_wbc) * time_factor
        new_platelets = current_platelets + (target_platelets - current_platelets) * time_factor
        new_hemoglobin = current_hemoglobin + (target_hemoglobin - current_hemoglobin) * time_factor
        
        # Only apply significant changes and ensure they're within physiological limits
        result = {}
        if abs(new_wbc - current_wbc) > 0.1:
            result["wbc"] = max(0.5, min(30.0, new_wbc))  # 0.5-30 K/μL range
            
        if abs(new_platelets - current_platelets) > 1.0:
            result["platelets"] = max(10.0, min(600.0, new_platelets))  # 10-600 K/μL range
            
        if abs(new_hemoglobin - current_hemoglobin) > 0.1:
            result["hemoglobin"] = max(3.0, min(18.0, new_hemoglobin))  # 3-18 g/dL range
            
        if result:
            self.logger.info(f"Coupling: Infection affecting hemogram - "
                           f"WBC: {current_wbc:.1f} -> {result.get('wbc', current_wbc):.1f}, "
                           f"PLT: {current_platelets:.0f} -> {result.get('platelets', current_platelets):.0f}")
            
        return result