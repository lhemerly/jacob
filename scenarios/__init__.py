"""
Scenarios for the Jacob simulation.
Each scenario represents a clinical event or condition that unfolds over time.
"""

from scenarios.fever import FeverScenario
from scenarios.sepsis import SepsisScenario
from scenarios.hemorrhage import HemorrhageScenario

# Export scenarios for easy importing elsewhere
__all__ = ['FeverScenario', 'SepsisScenario', 'HemorrhageScenario']