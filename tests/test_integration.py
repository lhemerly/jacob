import unittest
import logging
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from master import Master

# Import scenarios
from scenarios import FeverScenario, HemorrhageScenario, SepsisScenario

# Import actions
from actions import BloodTestAction, MEDICATIONS, FLUIDS

# Import all solvers
from solvers.pressure_HR_Oxy import PressureHROxySolver
from solvers.meds import MedsSolver
from solvers.fluids import FluidsSolver
from solvers.tss import TSSSolver
from solvers.urine import UrineSolver
from solvers.coagulation import CoagulationSolver
from solvers.drains import DrainsSolver
from solvers.electrolytes import ElectrolytesSolver
from solvers.fever import FeverSolver
from solvers.hemogram import HemogramSolver
from solvers.lactate import LactateSolver
from solvers.metabolytes import MetabolytesSolver
from solvers.crp import CRPSolver
from solvers.rhythm import RhythmSolver
from solvers.sedation import SedationSolver

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO)

class TestIntegration(unittest.TestCase):
    """
    Integration tests for the entire simulation system.
    Tests focus on interface behavior and mechanics rather than specific physiological changes.
    """
    
    def setUp(self):
        # Create actions list from imports
        blood_test = BloodTestAction()
        medication_actions = list(MEDICATIONS.values())
        fluid_actions = list(FLUIDS.values())
        all_actions = [blood_test] + medication_actions + fluid_actions
        
        # Create scenarios with shorter durations for testing
        scenarios = [
            FeverScenario(peak_temp=39.5, onset_duration=3.0, peak_duration=4.0, resolution_duration=3.0),
            HemorrhageScenario(severity=0.8, onset_duration=5.0, recovery_threshold=10.0),
            SepsisScenario(severity=1.2, onset_duration=3.0, duration=12.0)
        ]
        
        # Set up all implemented solvers
        solvers = [
            PressureHROxySolver(),
            MedsSolver(),
            FluidsSolver(),
            TSSSolver(),
            UrineSolver(),
            CoagulationSolver(),
            DrainsSolver(),
            ElectrolytesSolver(),
            FeverSolver(),
            HemogramSolver(),
            LactateSolver(),
            MetabolytesSolver(),
            CRPSolver(),
            RhythmSolver(),
            SedationSolver()
        ]
        
        # Set up the master with our components
        self.master = Master(
            solvers=solvers,
            dt=1.0,
            scenarios=scenarios,
            actions=all_actions
        )
        
        # Store initial state for comparison
        self.initial_state = self.master.state.copy()
    
    def test_solver_state_modification(self):
        """Test that solver state modification interface works by verifying state changes happen."""
        # Run one step
        new_state = self.master.step()
        
        # Verify the state dict is returned
        self.assertIsInstance(new_state, dict)
        
        # Verify the state contains all expected keys from solvers
        expected_keys = {
            # Key physiological parameters from different solvers
            "heart_rate", "blood_pressure", "oxy_saturation",  # PressureHROxy
            "fluid_volume",                                    # Fluids
            "temperature",                                     # Fever
            "lactate",                                        # Lactate
            "glucose", "ph",                                  # Metabolytes
            "hemoglobin", "platelets", "wbc",                 # Hemogram
            "sodium", "potassium",                            # Electrolytes
        }
        for key in expected_keys:
            self.assertIn(key, new_state, f"Expected key {key} missing from state")
        
        # Verify that at least some values have changed
        # We don't care which ones specifically, just that the simulation is running
        changes = [k for k in new_state if new_state[k] != self.initial_state[k]]
        self.assertGreater(len(changes), 0, "No state values changed after step")
    
    def test_scenario_activation_mechanics(self):
        """Test scenario activation/deactivation mechanics."""
        # Initially no active scenarios
        self.assertEqual(len(self.master.active_scenarios), 0)
        
        # Test activation
        for scenario in ["Fever", "Hemorrhage", "Sepsis"]:
            # Activate scenario
            success = self.master.apply_scenario(scenario)
            self.assertTrue(success)
            
            # Verify it's in active scenarios
            active_names = [s.name for s in self.master.active_scenarios]
            self.assertIn(scenario, active_names)
            
            # Deactivate scenario
            success = self.master.deactivate_scenario(scenario)
            self.assertTrue(success)
            
            # Verify it's no longer active
            active_names = [s.name for s in self.master.active_scenarios]
            self.assertNotIn(scenario, active_names)
    
    def test_scenario_duration_mechanics(self):
        """Test that scenarios handle their duration properly."""
        # Get a scenario with finite duration
        sepsis = next(s for s in self.master.scenarios if s.name == "Sepsis")
        duration = sepsis.duration
        self.assertGreater(duration, 0)  # Verify it has a duration
        
        # Activate the scenario
        self.master.apply_scenario("Sepsis")
        self.assertEqual(len(self.master.active_scenarios), 1)
        
        # Run until just before duration
        for _ in range(int(duration) - 1):
            self.master.step()
        self.assertEqual(len(self.master.active_scenarios), 1)
        
        # Run one more step to complete duration
        self.master.step()
        self.assertEqual(len(self.master.active_scenarios), 0)
    
    def test_multiple_scenario_handling(self):
        """Test that multiple scenarios can be active simultaneously."""
        # Activate two scenarios
        self.master.apply_scenario("Fever")
        self.master.apply_scenario("Hemorrhage")
        
        # Verify both are active
        self.assertEqual(len(self.master.active_scenarios), 2)
        active_names = {s.name for s in self.master.active_scenarios}
        self.assertEqual(active_names, {"Fever", "Hemorrhage"})
        
        # Run a few steps
        for _ in range(5):
            self.master.step()
            
        # Verify they're still active
        self.assertEqual(len(self.master.active_scenarios), 2)
    
    def test_action_scenario_interaction_mechanics(self):
        """Test that actions and scenarios can coexist."""
        # Start a scenario
        self.master.apply_scenario("Hemorrhage")
        
        # Perform some actions
        self.master.perform_action_by_name("Normal Saline Administration")
        self.master.perform_action_by_name("Epinephrine")
        
        # Verify everything is tracked
        self.assertEqual(len(self.master.active_scenarios), 1)
        self.assertGreaterEqual(len(self.master.active_actions), 1)
        
        # Run a few steps
        for _ in range(5):
            self.master.step()
            
        # Verify states are still tracking properly
        self.assertEqual(len(self.master.active_scenarios), 1)
        active_action_names = [a["action"].name for a in self.master.active_actions]
        self.assertIn("Epinephrine", active_action_names)
    
    def test_scenario_state_interface(self):
        """Test that scenarios properly implement their state interface."""
        for scenario in self.master.scenarios:
            # Check interface methods
            self.assertIsInstance(scenario.name, str)
            self.assertIsInstance(scenario.description, str)
            self.assertIsInstance(scenario.affected_keys, list)
            self.assertIsInstance(scenario.duration, float)
            
            # Verify initial_state is a dict
            self.assertIsInstance(scenario.initial_state, dict)
            
            # Activate and verify is_active flag
            scenario.activate()
            self.assertTrue(scenario.is_active)
            
            # Deactivate and verify is_active flag
            scenario.deactivate()
            self.assertFalse(scenario.is_active)


if __name__ == "__main__":
    unittest.main()