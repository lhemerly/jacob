from classes import Action

class BloodTestAction(Action):
    """
    Action for performing a blood test, which reveals hemogram, coagulation, and other blood values.
    Has a minimal effect on the patient's blood volume.
    """
    
    @property
    def name(self) -> str:
        return "Blood Test"
    
    @property
    def description(self) -> str:
        return "Perform a blood test to measure hemogram, coagulation, and other blood values"
    
    @property
    def affected_keys(self) -> list:
        return ["blood_volume"]
    
    @property
    def required_keys(self) -> list:
        return ["blood_volume", "hemoglobin", "platelets", "wbc", "inr", "aptt", "crp"]
    
    def apply(self, state: dict, **kwargs) -> dict:
        """
        Apply the effect of taking a blood sample - a small decrease in blood volume.
        
        :param state: The global state dictionary
        :return: Dictionary of state changes to apply
        """
        # A typical blood test takes about 10-20 mL of blood
        blood_draw_volume = 0.015  # Liters, assuming 15 mL
        
        changes = {
            "blood_volume": -blood_draw_volume
        }
        
        return changes
    
    def get_observable_state(self, state: dict) -> dict:
        """
        Return the blood test results from the state.
        
        :param state: The global state dictionary
        :return: Dictionary of observable state values
        """
        observable = {}
        
        # Define which keys are observable through this action
        observable_keys = [
            "hemoglobin", "platelets", "wbc", "inr", "aptt", "crp",
            "sodium", "potassium", "chloride", "glucose", "lactate"
        ]
        
        # Add only the keys that exist in the state
        for key in observable_keys:
            if key in state:
                observable[key] = state[key]
                
        return observable