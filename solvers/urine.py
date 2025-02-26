import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class UrineState(State):
    def __init__(self, data: dict):
        # Urine output in mL/hr
        self._output = data.get("urine_output", 60.0)  # Normal: 30-100 mL/hr
        # Urine specific gravity (1.005-1.030)
        self._specific_gravity = data.get("urine_specific_gravity", 1.015)
        # Urine sodium concentration (mmol/L)
        self._sodium = data.get("urine_sodium", 100.0)
        # Kidney function (GFR proxy, 0-100%)
        self._kidney_function = data.get("kidney_function", 100.0)
        # Osmolality (mOsm/kg)
        self._osmolality = data.get("urine_osmolality", 600.0)
        # Protein content (mg/dL)
        self._protein = data.get("urine_protein", 0.0)  # Normal < 20 mg/dL

    @property
    def state(self) -> dict:
        return {
            "urine_output": self._output,
            "urine_specific_gravity": self._specific_gravity,
            "urine_sodium": self._sodium,
            "kidney_function": self._kidney_function,
            "urine_osmolality": self._osmolality,
            "urine_protein": self._protein
        }


class UrineSolver(Solver):
    def __init__(self,
                 base_output_rate: float = 60.0,
                 kidney_recovery_rate: float = 0.01,
                 osmolality_adjustment_rate: float = 0.05,
                 protein_clearance_rate: float = 0.1):
        """
        Initialize urine solver with physiological parameters
        
        :param base_output_rate: Baseline urine output in mL/hr
        :param kidney_recovery_rate: Rate of kidney function recovery
        :param osmolality_adjustment_rate: Rate of osmolality adjustment
        :param protein_clearance_rate: Rate of protein clearance
        """
        self._state = UrineState({})
        self.base_output_rate = base_output_rate
        self.kidney_recovery_rate = kidney_recovery_rate
        self.osmolality_adjustment_rate = osmolality_adjustment_rate
        self.protein_clearance_rate = protein_clearance_rate

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        us = UrineState(state)
        
        # Get relevant physiological parameters if available
        blood_pressure = state.get("blood_pressure", 90.0)
        fluid_volume = state.get("fluid_volume", 2000.0)
        serum_sodium = state.get("sodium", 140.0)
        
        # Update kidney function
        # Function decreases with low BP, slowly recovers with normal BP
        if blood_pressure < 65.0:
            kidney_damage = (65.0 - blood_pressure) * 0.02 * dt
            new_kidney_function = max(0, us.state["kidney_function"] - kidney_damage)
        else:
            recovery = (self.kidney_recovery_rate * 
                       (100.0 - us.state["kidney_function"])) * dt
            new_kidney_function = min(100, us.state["kidney_function"] + recovery)
        
        # Calculate urine output
        # Affected by kidney function, BP, and fluid volume
        volume_factor = max(0.2, min(2.0, fluid_volume / 2000.0))
        bp_factor = max(0.2, min(1.5, blood_pressure / 90.0))
        base_output = (self.base_output_rate * 
                      volume_factor * 
                      bp_factor * 
                      (new_kidney_function / 100.0))
        
        output_change = (base_output - us.state["urine_output"]) * 0.1 * dt
        new_output = max(0, us.state["urine_output"] + output_change)
        
        # Update specific gravity and osmolality
        # Inversely related to urine output
        target_sg = 1.015 + (60.0 - new_output) * 0.0003
        sg_change = (target_sg - us.state["urine_specific_gravity"]) * 0.1 * dt
        new_sg = max(1.001, min(1.040, 
                    us.state["urine_specific_gravity"] + sg_change))
        
        target_osm = 600.0 + (60.0 - new_output) * 10.0
        osm_change = (self.osmolality_adjustment_rate * 
                     (target_osm - us.state["urine_osmolality"])) * dt
        new_osm = max(50, min(1200, us.state["urine_osmolality"] + osm_change))
        
        # Update urine sodium
        # Influenced by serum sodium and kidney function
        target_sodium = serum_sodium * (new_kidney_function / 100.0)
        sodium_change = (target_sodium - us.state["urine_sodium"]) * 0.1 * dt
        new_sodium = max(0, us.state["urine_sodium"] + sodium_change)
        
        # Update protein content
        # Increases with kidney damage, cleared over time
        protein_production = (100 - new_kidney_function) * 0.2
        protein_clearance = self.protein_clearance_rate * us.state["urine_protein"]
        protein_change = (protein_production - protein_clearance) * dt
        new_protein = max(0, us.state["urine_protein"] + protein_change)

        logger.debug(
            f"UrineSolver: output={new_output:.1f} mL/hr, "
            f"kidney={new_kidney_function:.1f}%, "
            f"sg={new_sg:.3f}, Na={new_sodium:.1f}, "
            f"protein={new_protein:.1f}"
        )

        return UrineState({
            "urine_output": new_output,
            "urine_specific_gravity": new_sg,
            "urine_sodium": new_sodium,
            "kidney_function": new_kidney_function,
            "urine_osmolality": new_osm,
            "urine_protein": new_protein
        })