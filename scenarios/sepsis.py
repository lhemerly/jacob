"""
Sepsis scenario that gradually alters multiple physiological parameters over time.
"""
from classes import Scenario
import math


class SepsisScenario(Scenario):
    """
    Simulates sepsis progression with effects on multiple physiological parameters.
    """
    
    def __init__(self, severity=1.0, onset_duration=3600, duration=14400):
        """
        Initialize the sepsis scenario.
        
        :param severity: Severity factor from 0.1 (mild) to 2.0 (severe)
        :param onset_duration: Time for sepsis to fully develop in seconds
        :param duration: Total duration before spontaneous improvement in seconds
        """
        super().__init__()
        self.severity = min(max(severity, 0.1), 2.0)  # Clamp to valid range
        self.onset_duration = onset_duration  # 1 hour by default
        self.total_duration = duration        # 4 hours by default
        self._phase = 'onset'
    
    @property
    def name(self) -> str:
        return "Sepsis"
    
    @property
    def description(self) -> str:
        severity_desc = "mild" if self.severity < 0.7 else "moderate" if self.severity < 1.3 else "severe"
        return f"{severity_desc.capitalize()} sepsis scenario (severity factor: {self.severity:.1f})"
    
    @property
    def affected_keys(self) -> list:
        return [
            "heart_rate", "systolic_bp", "diastolic_bp", "body_temperature",
            "respiratory_rate", "wbc_count", "lactate", "crp"
        ]
    
    @property
    def duration(self) -> float:
        # Total duration of the scenario
        return self.total_duration
    
    @property
    def initial_state(self) -> dict:
        # Default initial values if they don't exist in the global state
        return {
            "heart_rate": 75,           # bpm
            "systolic_bp": 120,         # mmHg
            "diastolic_bp": 80,         # mmHg
            "body_temperature": 37.0,   # Celsius
            "respiratory_rate": 14,     # breaths per minute
            "wbc_count": 7.5,           # x10^9/L
            "lactate": 1.0,             # mmol/L
            "crp": 5.0                  # mg/L
        }
    
    def apply(self, state: dict, dt: float) -> dict:
        """
        Apply sepsis effects to the state.
        
        The sepsis progresses through phases:
        1. Onset - parameters gradually worsen
        2. Progression - parameters continue to deteriorate at a slower rate
        3. Resolution - if reached the time limit, parameters slowly improve
        """
        # Calculate the intensity of effect based on phase and time
        if self._phase == 'onset':
            if self.elapsed_time < self.onset_duration:
                # Progressive worsening during onset phase
                intensity = self.elapsed_time / self.onset_duration * self.severity
            else:
                # Transition to progression phase
                intensity = self.severity
                self._phase = 'progression'
        
        elif self._phase == 'progression':
            # Continue to worsen slightly during middle phase
            mid_phase_duration = self.total_duration * 0.6  # 60% of total time
            if self.elapsed_time < (self.onset_duration + mid_phase_duration):
                additional = 0.2 * (self.elapsed_time - self.onset_duration) / mid_phase_duration
                intensity = self.severity + (additional * self.severity)
            else:
                # Transition to resolution phase
                self._phase = 'resolution'
                intensity = self.severity * 1.2  # Peak intensity
        
        else:  # resolution phase
            # Gradually improve
            resolution_progress = (self.elapsed_time - self.onset_duration - (self.total_duration * 0.6)) / (self.total_duration * 0.4)
            resolution_progress = min(resolution_progress, 1.0)
            intensity = self.severity * 1.2 * (1 - (resolution_progress * 0.5))  # Improve by up to 50%
        
        # Apply effects proportional to intensity
        result = {}
        
        # Heart rate increases (tachycardia)
        baseline_hr = state.get("heart_rate", 75)
        result["heart_rate"] = baseline_hr + (30 * intensity)
        
        # Blood pressure decreases (hypotension)
        baseline_sys = state.get("systolic_bp", 120)
        baseline_dia = state.get("diastolic_bp", 80)
        result["systolic_bp"] = baseline_sys - (20 * intensity)
        result["diastolic_bp"] = baseline_dia - (15 * intensity)
        
        # Temperature increases (fever)
        baseline_temp = state.get("body_temperature", 37.0)
        # Add some oscillation to make it more realistic
        temp_oscillation = math.sin(self.elapsed_time / 1800) * 0.3  # Small oscillation with 30-minute period
        result["body_temperature"] = baseline_temp + (1.5 * intensity) + temp_oscillation
        
        # Respiratory rate increases
        baseline_rr = state.get("respiratory_rate", 14)
        result["respiratory_rate"] = baseline_rr + (8 * intensity)
        
        # WBC count changes (often elevated but can decrease in severe cases)
        baseline_wbc = state.get("wbc_count", 7.5)
        if self.severity < 1.5:
            # Increase in less severe cases
            result["wbc_count"] = baseline_wbc + (8 * intensity)
        else:
            # Potential decrease in severe cases
            result["wbc_count"] = baseline_wbc - (2 * intensity) + (3 * math.sin(self.elapsed_time / 3600))
        
        # Lactate increases
        baseline_lactate = state.get("lactate", 1.0)
        result["lactate"] = baseline_lactate + (3 * intensity)
        
        # CRP increases
        baseline_crp = state.get("crp", 5.0)
        result["crp"] = baseline_crp + (80 * intensity)
        
        return result