import logging
from classes import Solver, State

logger = logging.getLogger(__name__)


class TSSState(State):
    def __init__(self, data: dict):
        # Overall TSS severity score (0-100)
        self._severity = data.get("tss_severity", 0.0)
        # Tissue damage from toxins (0-100)
        self._tissue_damage = data.get("tissue_damage", 0.0)
        # Toxin level in blood (0-100)
        self._toxin_level = data.get("toxin_level", 0.0)
        # Immune response level (0-100)
        self._immune_response = data.get("immune_response", 50.0)

    @property
    def state(self) -> dict:
        return {
            "tss_severity": self._severity,
            "tissue_damage": self._tissue_damage,
            "toxin_level": self._toxin_level,
            "immune_response": self._immune_response
        }


class TSSSolver(Solver):
    def __init__(self,
                 toxin_production_rate: float = 0.05,
                 toxin_clearance_rate: float = 0.03,
                 tissue_damage_rate: float = 0.02,
                 tissue_healing_rate: float = 0.01,
                 immune_response_rate: float = 0.1):
        """
        Initialize TSS solver with physiological parameters
        
        :param toxin_production_rate: Rate of toxin production
        :param toxin_clearance_rate: Rate at which toxins are cleared
        :param tissue_damage_rate: Rate of tissue damage from toxins
        :param tissue_healing_rate: Rate of tissue healing
        :param immune_response_rate: Rate of immune system response
        """
        self._state = TSSState({})
        self.toxin_production_rate = toxin_production_rate
        self.toxin_clearance_rate = toxin_clearance_rate
        self.tissue_damage_rate = tissue_damage_rate
        self.tissue_healing_rate = tissue_healing_rate
        self.immune_response_rate = immune_response_rate
        
        # Flag to track if we need to sync other values with severity
        self.severity_manually_set = False

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ts = TSSState(state)
        
        # Get relevant physiological parameters if available
        body_temp = state.get("temperature", 37.0)
        wbc_count = state.get("wbc", 7.5)
        
        # Check if tss_severity was manually set via actions
        current_severity = ts.state["tss_severity"]
        previous_severity = getattr(self, '_last_calculated_severity', 0.0)
        
        # Detect if severity was changed externally (via actions)
        # by comparing with our last calculated value
        severity_changed_externally = abs(current_severity - previous_severity) > 0.1 and hasattr(self, '_last_calculated_severity')
        
        # If severity was changed externally, adjust internal values to match the new severity
        if severity_changed_externally:
            # If severity was changed externally, adjust toxin and damage levels to match
            # This ensures the solver honors external severity changes
            target_toxin_damage = min(100, current_severity / 0.8 * 1.5)  # Rough inverse of severity calculation
            
            # Update toxin level and tissue damage to be consistent with the new severity
            # while maintaining their relative proportions
            if ts.state["toxin_level"] + ts.state["tissue_damage"] > 0:
                proportion = ts.state["toxin_level"] / (ts.state["toxin_level"] + ts.state["tissue_damage"])
            else:
                proportion = 0.5  # Equal split if both are zero
                
            # Set new values while maintaining relative proportions
            new_toxin = target_toxin_damage * proportion
            new_damage = target_toxin_damage * (1 - proportion)
            
            # Set these as the starting points for this solve iteration
            ts = TSSState({
                "tss_severity": current_severity,
                "tissue_damage": new_damage,
                "toxin_level": new_toxin,
                "immune_response": ts.state["immune_response"]
            })
        
        # Update toxin levels
        # Production increases with tissue damage, clearance depends on immune response
        toxin_production = (self.toxin_production_rate * 
                          (1 + ts.state["tissue_damage"] / 50.0))
        toxin_clearance = (self.toxin_clearance_rate * 
                          ts.state["immune_response"] / 50.0)
        
        toxin_change = (toxin_production - 
                       toxin_clearance * ts.state["toxin_level"]) * dt
        new_toxin = max(0, min(100, ts.state["toxin_level"] + toxin_change))
        
        # Update tissue damage
        # Damage from toxins, healing depends on immune response
        damage_rate = self.tissue_damage_rate * new_toxin
        healing_rate = (self.tissue_healing_rate * 
                       ts.state["immune_response"] / 50.0)
        
        damage_change = (damage_rate - 
                        healing_rate * ts.state["tissue_damage"]) * dt
        new_damage = max(0, min(100, ts.state["tissue_damage"] + damage_change))
        
        # Update immune response
        # Strengthens with infection but can become overwhelmed
        target_response = min(100, 50 + new_toxin)
        if body_temp > 38.5:  # Fever boosts immune response
            target_response *= 1.2
        if wbc_count < 4.0:  # Low WBC impairs immune response
            target_response *= 0.5
            
        response_change = (self.immune_response_rate * 
                         (target_response - ts.state["immune_response"])) * dt
        new_response = max(0, min(100, ts.state["immune_response"] + response_change))
        
        # Calculate overall severity score
        new_severity = (new_toxin * 0.4 + 
                       new_damage * 0.4 + 
                       (100 - new_response) * 0.2)

        # Save the calculated severity for next comparison
        self._last_calculated_severity = new_severity

        logger.debug(
            f"TSSSolver: severity={new_severity:.1f}, "
            f"toxin={new_toxin:.1f}, damage={new_damage:.1f}, "
            f"immune={new_response:.1f}"
        )

        return TSSState({
            "tss_severity": new_severity,
            "tissue_damage": new_damage,
            "toxin_level": new_toxin,
            "immune_response": new_response
        })