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

    @property
    def state(self):
        return self._state.state

    def solve(self, state: dict, dt: float) -> State:
        ts = TSSState(state)
        
        # Get relevant physiological parameters if available
        body_temp = state.get("temperature", 37.0)
        wbc_count = state.get("wbc", 7.5)
        
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