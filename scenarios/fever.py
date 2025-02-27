"""
Fever scenario that gradually raises body temperature over time and then resolves.
"""
from classes import Scenario


class FeverScenario(Scenario):
    """
    Simulates a fever that develops gradually, peaks, and then resolves over time.
    """
    
    def __init__(self, peak_temp=39.5, onset_duration=3600, peak_duration=7200, resolution_duration=5400):
        """
        Initialize the fever scenario.
        
        :param peak_temp: Peak temperature in Celsius
        :param onset_duration: Time to reach peak temperature in seconds
        :param peak_duration: Time fever stays at peak in seconds
        :param resolution_duration: Time for fever to resolve in seconds
        """
        super().__init__()
        self.peak_temp = peak_temp
        self.onset_duration = onset_duration  # 1 hour by default
        self.peak_duration = peak_duration    # 2 hours by default
        self.resolution_duration = resolution_duration  # 1.5 hours by default
        self.baseline_temp = 37.0  # Normal body temperature
        self._phase = 'onset'
    
    @property
    def name(self) -> str:
        return "Fever"
    
    @property
    def description(self) -> str:
        return f"Fever scenario that peaks at {self.peak_temp}°C"
    
    @property
    def affected_keys(self) -> list:
        return ["body_temperature"]
    
    @property
    def duration(self) -> float:
        # Total duration of the scenario
        return self.onset_duration + self.peak_duration + self.resolution_duration
    
    @property
    def initial_state(self) -> dict:
        return {"body_temperature": self.baseline_temp}
    
    def apply(self, state: dict, dt: float) -> dict:
        """
        Apply fever effects to the state.
        
        The fever progresses through three phases:
        1. Onset - temperature rises gradually
        2. Peak - temperature stays at peak
        3. Resolution - temperature returns to normal gradually
        """
        current_temp = state.get("body_temperature", self.baseline_temp)
        
        if self._phase == 'onset':
            # Temperature rising phase
            if self.elapsed_time <= self.onset_duration:
                # Linear increase to peak temperature
                progress = self.elapsed_time / self.onset_duration
                new_temp = self.baseline_temp + (self.peak_temp - self.baseline_temp) * progress
            else:
                # Move to peak phase
                new_temp = self.peak_temp
                self._phase = 'peak'
                
        elif self._phase == 'peak':
            # Temperature stays at peak
            if self.elapsed_time <= (self.onset_duration + self.peak_duration):
                new_temp = self.peak_temp
            else:
                # Move to resolution phase
                self._phase = 'resolution'
                new_temp = self.peak_temp
                
        elif self._phase == 'resolution':
            # Temperature going back to normal
            resolution_progress = (self.elapsed_time - self.onset_duration - self.peak_duration) / self.resolution_duration
            resolution_progress = min(resolution_progress, 1.0)  # Cap at 1.0
            new_temp = self.peak_temp - (self.peak_temp - self.baseline_temp) * resolution_progress
            
        return {"body_temperature": new_temp}