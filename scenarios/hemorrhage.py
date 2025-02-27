"""
Hemorrhage scenario that simulates bleeding with effects on multiple physiological parameters.
"""
from classes import Scenario
import math
import random


class HemorrhageScenario(Scenario):
    """
    Simulates hemorrhage (bleeding) with effects on hemodynamics, 
    coagulation, and hemogram parameters.
    """
    
    def __init__(self, severity=1.0, onset_duration=900, spontaneous_recovery=True, 
                 recovery_threshold=7200, recovery_duration=3600):
        """
        Initialize the hemorrhage scenario.
        
        :param severity: Severity factor from 0.1 (minimal) to 2.0 (massive)
        :param onset_duration: Time for initial bleeding to fully develop in seconds
        :param spontaneous_recovery: Whether bleeding can stop spontaneously
        :param recovery_threshold: Time after which spontaneous recovery may begin (if enabled)
        :param recovery_duration: Duration of recovery phase if spontaneous recovery occurs
        """
        super().__init__()
        self.severity = min(max(severity, 0.1), 2.0)  # Clamp to valid range
        self.onset_duration = onset_duration  # 15 minutes by default
        self.spontaneous_recovery = spontaneous_recovery
        self.recovery_threshold = recovery_threshold  # 2 hours by default
        self.recovery_duration = recovery_duration  # 1 hour by default
        self._phase = 'onset'
        self._blood_lost = 0  # Track cumulative blood loss in mL
    
    @property
    def name(self) -> str:
        return "Hemorrhage"
    
    @property
    def description(self) -> str:
        severity_desc = "minimal" if self.severity < 0.5 else "moderate" if self.severity < 1.0 else "severe" if self.severity < 1.5 else "massive"
        recovery_desc = "with potential spontaneous recovery" if self.spontaneous_recovery else "without spontaneous recovery"
        return f"{severity_desc.capitalize()} hemorrhage (severity: {self.severity:.1f}) {recovery_desc}"
    
    @property
    def affected_keys(self) -> list:
        return [
            "heart_rate", "systolic_bp", "diastolic_bp", 
            "blood_pressure", "bleeding_rate", "hemoglobin", 
            "platelets", "fluid_volume", "hematocrit", 
            "inr", "urine_output"
        ]
    
    @property
    def duration(self) -> float:
        # If no spontaneous recovery, scenario continues indefinitely
        if not self.spontaneous_recovery:
            return -1
        return self.recovery_threshold + self.recovery_duration
    
    @property
    def initial_state(self) -> dict:
        # Default initial values if they don't exist in the global state
        return {
            "heart_rate": 75,           # bpm
            "systolic_bp": 120,         # mmHg
            "diastolic_bp": 80,         # mmHg
            "blood_pressure": 93,       # MAP in mmHg
            "bleeding_rate": 0.0,       # units (0-10)
            "hemoglobin": 14.0,         # g/dL
            "hematocrit": 42.0,         # %
            "platelets": 250.0,         # K/µL
            "inr": 1.0,                 # ratio
            "fluid_volume": 5000.0,     # mL
            "urine_output": 60.0        # mL/hr
        }
    
    def apply(self, state: dict, dt: float) -> dict:
        """
        Apply hemorrhage effects to the state.
        
        The hemorrhage progresses through phases:
        1. Onset - rapid bleeding with worsening parameters
        2. Continued bleeding - ongoing blood loss
        3. Recovery - IF spontaneous recovery is enabled AND recovery threshold is reached
        """
        result = {}
        
        # Calculate bleeding intensity and blood loss
        if self._phase == 'onset':
            if self.elapsed_time < self.onset_duration:
                # Progressive worsening during onset phase
                bleeding_intensity = (self.elapsed_time / self.onset_duration) * self.severity * 10
            else:
                # Transition to continued bleeding phase
                bleeding_intensity = self.severity * 10
                self._phase = 'continued'
        
        elif self._phase == 'continued':
            if self.spontaneous_recovery and self.elapsed_time >= self.recovery_threshold:
                # Transition to recovery phase
                self._phase = 'recovery'
                bleeding_intensity = self.severity * 5  # Initial reduction in bleeding
            else:
                # Continue bleeding at severity-determined rate
                # Add some randomness to make it more realistic
                variation = (random.random() - 0.5) * 2  # Random factor between -1 and 1
                bleeding_intensity = self.severity * (10 + variation)
        
        else:  # recovery phase
            # Gradually reduce bleeding
            recovery_progress = (self.elapsed_time - self.recovery_threshold) / self.recovery_duration
            recovery_progress = min(recovery_progress, 1.0)
            bleeding_intensity = self.severity * 5 * (1 - recovery_progress)
        
        # Calculate blood loss for this time step (mL)
        blood_loss_rate = bleeding_intensity * 20  # mL per second based on intensity
        current_blood_loss = blood_loss_rate * dt  # mL lost in this time step
        self._blood_lost += current_blood_loss
        
        # Update bleeding rate in simulation
        result["bleeding_rate"] = bleeding_intensity
        
        # Heart rate increases (compensatory tachycardia)
        baseline_hr = state.get("heart_rate", 75)
        # As blood loss increases, heart rate increases, but eventually drops if severe
        if self._blood_lost < 1500:
            # Compensatory phase - increased heart rate
            hr_change = 25 * (self._blood_lost / 1500)
            result["heart_rate"] = baseline_hr + hr_change
        else:
            # Decompensation phase - heart rate may drop in severe hemorrhage
            hr_drop_factor = (self._blood_lost - 1500) / 1000
            hr_drop_factor = min(hr_drop_factor, 1.0)
            hr_change = 25 - (50 * hr_drop_factor)  # From +25 down to -25
            result["heart_rate"] = baseline_hr + hr_change
        
        # Blood pressure decreases
        baseline_sys = state.get("systolic_bp", 120)
        baseline_dia = state.get("diastolic_bp", 80)
        
        # Blood pressure drops proportionally to blood loss
        bp_drop_factor = (self._blood_lost / 2000)  # Normalized blood loss factor
        bp_drop_factor = min(bp_drop_factor, 1.0)  # Cap at 1.0
        
        # Systolic drops more than diastolic, narrowing pulse pressure
        result["systolic_bp"] = baseline_sys - (50 * bp_drop_factor)
        result["diastolic_bp"] = baseline_dia - (30 * bp_drop_factor)
        
        # Update mean arterial pressure
        result["blood_pressure"] = (result["systolic_bp"] + 2 * result["diastolic_bp"]) / 3
        
        # Hemoglobin and hematocrit decrease
        baseline_hgb = state.get("hemoglobin", 14.0)
        baseline_hct = state.get("hematocrit", 42.0)
        
        # Calculate hemodilution effect based on blood loss and fluid shifts
        fluid_volume = state.get("fluid_volume", 5000.0)
        
        # Assume blood volume is approximately 5L (5000mL)
        # As blood is lost, the body compensates by pulling fluid from interstitial space
        # This causes hemodilution, reducing hemoglobin concentration
        
        if fluid_volume > 0:
            # Factor in fluid shifts from blood loss
            hemodilution_factor = 1 - (self._blood_lost / (fluid_volume + self._blood_lost))
            result["hemoglobin"] = baseline_hgb * hemodilution_factor
            result["hematocrit"] = baseline_hct * hemodilution_factor
        else:
            # Avoid division by zero
            result["hemoglobin"] = baseline_hgb * 0.6
            result["hematocrit"] = baseline_hct * 0.6
        
        # Reduce platelets due to consumption
        baseline_plt = state.get("platelets", 250.0)
        plt_drop_factor = (self._blood_lost / 2000)  # Normalized blood loss factor
        plt_drop_factor = min(plt_drop_factor, 0.6)  # Maximum 60% drop
        result["platelets"] = baseline_plt * (1 - plt_drop_factor)
        
        # INR increases slightly due to consumption of coagulation factors
        baseline_inr = state.get("inr", 1.0)
        inr_increase_factor = (self._blood_lost / 3000) * self.severity
        inr_increase_factor = min(inr_increase_factor, 1.0)  # Cap at 1.0
        result["inr"] = baseline_inr + (0.8 * inr_increase_factor)
        
        # Fluid volume decreases
        baseline_fluid = state.get("fluid_volume", 5000.0)
        result["fluid_volume"] = baseline_fluid - current_blood_loss
        
        # Urine output decreases due to compensatory mechanisms
        baseline_urine = state.get("urine_output", 60.0)
        urine_drop_factor = (self._blood_lost / 1500) * self.severity
        urine_drop_factor = min(urine_drop_factor, 0.9)  # Maximum 90% drop
        result["urine_output"] = baseline_urine * (1 - urine_drop_factor)
        
        return result