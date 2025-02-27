from typing import Dict, Optional
from classes import Action

class FluidAdministrationAction(Action):
    """
    Action for administering IV fluids to increase blood volume.
    """
    
    def __init__(self, fluid_type: str, volume: float, electrolyte_content: Optional[Dict[str, float]] = None):
        """
        Initialize a fluid administration action.
        
        :param fluid_type: Type of fluid (e.g., "Normal Saline", "Lactated Ringer's")
        :param volume: Volume in liters
        :param electrolyte_content: Optional dictionary of electrolyte concentrations (mmol/L)
        """
        self._fluid_type = fluid_type
        self._volume = volume
        self._electrolyte_content = electrolyte_content or {}
        
    @property
    def name(self) -> str:
        return f"{self._fluid_type} Administration"
    
    @property
    def description(self) -> str:
        return f"Administer {self._volume * 1000} mL of {self._fluid_type} IV fluid"
    
    @property
    def affected_keys(self) -> list:
        keys = ["blood_volume"]
        if self._electrolyte_content:
            keys.extend(list(self._electrolyte_content.keys()))
        return keys
    
    @property
    def required_keys(self) -> list:
        keys = ["blood_volume"]
        if self._electrolyte_content:
            keys.extend(list(self._electrolyte_content.keys()))
        return keys
    
    def apply(self, state: dict, **kwargs) -> dict:
        """
        Apply the effects of fluid administration.
        
        :param state: The global state dictionary
        :return: Dictionary of state changes to apply
        """
        changes = {
            "blood_volume": self._volume
        }
        
        # Calculate electrolyte changes based on dilution and added content
        if self._electrolyte_content and "blood_volume" in state:
            current_volume = state["blood_volume"]
            for electrolyte, concentration in self._electrolyte_content.items():
                if electrolyte in state:
                    # Calculate new concentration based on mixing
                    current_concentration = state[electrolyte]
                    current_total = current_concentration * current_volume
                    added_total = concentration * self._volume
                    new_total = current_total + added_total
                    new_concentration = new_total / (current_volume + self._volume)
                    
                    # Record the delta (change in concentration)
                    changes[electrolyte] = new_concentration - current_concentration
        
        return changes

# Common fluid administration examples
FLUIDS = {
    "normal_saline": FluidAdministrationAction(
        "Normal Saline",
        0.5,  # 500 mL
        {"sodium": 154, "chloride": 154}
    ),
    "lactated_ringers": FluidAdministrationAction(
        "Lactated Ringer's",
        0.5,  # 500 mL
        {"sodium": 130, "potassium": 4, "calcium": 3, "chloride": 109, "lactate": 28}
    ),
    "d5w": FluidAdministrationAction(
        "D5W",
        0.5,  # 500 mL
        {"glucose": 50}  # 5% dextrose = 50 g/L
    )
}