from typing import Dict, Optional
from classes import Action

class MedicationAction(Action):
    """
    Action for administering medication with immediate and possibly lasting effects.
    """
    
    def __init__(self, medication_name: str, effects: Dict[str, float], 
                 duration: float = 0.0, description: Optional[str] = None):
        """
        Initialize a medication action.
        
        :param medication_name: The name of the medication
        :param effects: Dictionary mapping state keys to their delta values
        :param duration: Duration of effect in simulation time (0 = immediate)
        :param description: Optional description, defaults to standard format
        """
        self._name = medication_name
        self._effects = effects
        self._duration = duration
        self._description = description or f"Administer {medication_name} to the patient"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def affected_keys(self) -> list:
        return list(self._effects.keys())
    
    @property
    def duration(self) -> float:
        return self._duration
    
    def apply(self, state: dict, **kwargs) -> dict:
        """
        Apply the medication effects to the state.
        
        :param state: The global state dictionary
        :param kwargs: Additional parameters (dose multiplier, etc.)
        :return: Dictionary of state changes to apply
        """
        # Allow for dose adjustments through kwargs
        dose_multiplier = kwargs.get("dose_multiplier", 1.0)
        
        changes = {}
        for key, delta in self._effects.items():
            changes[key] = delta * dose_multiplier
            
        return changes

# Common medication examples
MEDICATIONS = {
    "epinephrine": MedicationAction(
        "Epinephrine", 
        {"heart_rate": +30, "blood_pressure": +20, "cardiac_output": +0.5}, 
        duration=10.0,
        description="Administer epinephrine to increase heart rate and blood pressure"
    ),
    "propofol": MedicationAction(
        "Propofol",
        {"blood_pressure": -10, "heart_rate": -5, "sedation_level": +2},
        duration=30.0,
        description="Administer propofol for sedation and anesthesia"
    ),
    "norepinephrine": MedicationAction(
        "Norepinephrine",
        {"blood_pressure": +15, "svr": +200},
        duration=20.0,
        description="Administer norepinephrine to increase blood pressure"
    ),
    "morphine": MedicationAction(
        "Morphine",
        {"pain_level": -3, "respiratory_rate": -2, "blood_pressure": -5},
        duration=240.0,
        description="Administer morphine for pain management"
    ),
    "antibiotics": MedicationAction(
        "Antibiotics",
        {"infection_level": -0.5, "wbc": -0.2},
        duration=720.0,  # 12 hours effect
        description="Administer antibiotics to fight infection"
    ),
}