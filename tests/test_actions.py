import unittest
import logging
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

from master import Master
from classes import Solver, State
from actions import BloodTestAction, MEDICATIONS, FLUIDS

# Configure logging to show info messages
logging.basicConfig(level=logging.INFO)


class MockState(State):
    def __init__(self, state_dict):
        self._state = state_dict
    
    @property
    def state(self):
        return self._state


class MockSolver(Solver):
    def __init__(self):
        """Initialize with all state variables needed for testing."""
        self._state = {
            # Vital signs
            "heart_rate": 80.0,
            "blood_pressure": 120.0,
            "respiratory_rate": 16.0,
            "temperature": 37.0,
            "oxygen_saturation": 98.0,
            
            # Fluids and blood values
            "blood_volume": 5000.0,  # 5000 mL
            "fluid_volume": 2000.0,  # Initial fluid volume from FluidsSolver
            
            # Lab values
            "hemoglobin": 14.0,   # g/dL
            "platelets": 250.0,   # x10^9/L
            "wbc": 7.5,          # x10^9/L
            "inr": 1.0,
            "aptt": 30.0,        # seconds
            "crp": 5.0,          # mg/L
            
            # Electrolytes
            "sodium": 140.0,     # mmol/L
            "potassium": 4.0,    # mmol/L
            "calcium": 2.4,      # mmol/L (normal range 2.2-2.7)
            "chloride": 100.0,   # mmol/L
            "glucose": 90.0,     # mg/dL
            "lactate": 1.0,      # mmol/L
            
            # Clinical status
            "infection_level": 0.0,
            "pain_level": 0.0,
            "sedation_level": 0.0,
            "bleeding_rate": 0.0,
            "cardiac_output": 5.0,
            "svr": 1200.0
        }
    
    @property
    def state(self):
        return self._state
    
    def solve(self, state: State, dt: float) -> State:
        # Just return the same state for testing interfaces
        return MockState(state)


class TestActions(unittest.TestCase):
    
    def setUp(self):
        # Create mock solver with comprehensive initial state
        mock_solver = MockSolver()
        
        # Create actions list from imports
        blood_test = BloodTestAction()
        medication_actions = list(MEDICATIONS.values())
        fluid_actions = list(FLUIDS.values())
        all_actions = [blood_test] + medication_actions + fluid_actions
        
        # Set up the master with our mock solver and actions
        self.master = Master(
            solvers=[mock_solver],
            dt=1.0,
            actions=all_actions
        )
        
        # Store initial state for comparison
        self.initial_state = self.master.state.copy()
    
    def test_action_state_access(self):
        """Test that actions can access their required state keys."""
        for action in self.master.actions:
            # Verify all required keys are present
            for key in action.required_keys:
                self.assertIn(key, self.master.state,
                            f"Action {action.name} required key {key} should be present in state")
    
    def test_action_state_modification(self):
        """Test that actions can modify their affected state keys."""
        # Test each action
        for action in self.master.actions:
            # Reset state before each action test
            self.master = Master(solvers=[MockSolver()], dt=1.0, actions=self.master.actions)
            before_state = self.master.state.copy()
            
            # Perform the action
            self.master.perform_action(action)
            
            # For each affected key that exists in state, verify it can be modified
            affected_keys = [k for k in action.affected_keys if k in before_state]
            self.assertTrue(len(affected_keys) > 0,
                          f"Action {action.name} should have at least one affected key present in state")
            
            # Verify at least one affected key changed
            # (we don't assume which ones should change, just that the action has an effect)
            changes = [
                self.master.state[key] != before_state[key]
                for key in affected_keys
            ]
            self.assertTrue(any(changes),
                          f"Action {action.name} should modify at least one of its affected keys")
            
            # Check non-affected keys remained unchanged
            unchanged_keys = set(self.master.state.keys()) - set(action.affected_keys)
            for key in unchanged_keys:
                self.assertEqual(
                    self.master.state[key],
                    before_state[key],
                    f"Action {action.name} should not modify non-affected key: {key}"
                )
    
    def test_medication_duration_interface(self):
        """Test that medication actions properly implement duration interface."""
        for name, med in MEDICATIONS.items():
            # Verify duration is accessible
            self.assertIsInstance(med.duration, float,
                                f"Medication {name} should have float duration")
            
            # Verify duration is non-negative
            self.assertGreaterEqual(med.duration, 0,
                                  f"Medication {name} should have non-negative duration")
    
    def test_action_observable_state_interface(self):
        """Test that action observable state interface works correctly."""
        for action in self.master.actions:
            # Get observable state
            observable = self.master.perform_action(action)
            
            # Verify it's a dict
            self.assertIsInstance(observable, dict,
                                f"Action {action.name} observable state should be a dict")
            
            # Verify all returned keys exist in main state
            for key in observable:
                self.assertIn(key, self.master.state,
                            f"Action {action.name} observable key {key} should exist in state")
                self.assertEqual(observable[key], self.master.state[key],
                               f"Action {action.name} observable value should match state")
    
    def test_fluid_administration_interface(self):
        """Test that fluid administration actions properly implement their interface."""
        for name, fluid in FLUIDS.items():
            # Check interface properties
            self.assertIsInstance(fluid.name, str)
            self.assertIsInstance(fluid.description, str)
            self.assertIsInstance(fluid.affected_keys, list)
            self.assertIsInstance(fluid.required_keys, list)
            
            # Verify blood_volume or fluid_volume is affected
            self.assertTrue(
                "blood_volume" in fluid.affected_keys or "fluid_volume" in fluid.affected_keys,
                f"Fluid action {name} should affect blood or fluid volume"
            )
    
    def test_action_name_uniqueness(self):
        """Test that all actions have unique names."""
        action_names = [action.name for action in self.master.actions]
        unique_names = set(action_names)
        self.assertEqual(len(action_names), len(unique_names),
                        "All actions should have unique names")
    
    def test_action_chaining(self):
        """Test that multiple actions can be performed in sequence."""
        # Get a few different actions with duration
        actions_with_duration = [a for a in self.master.actions if a.duration > 0]
        self.assertGreaterEqual(len(actions_with_duration), 2,
                              "Need at least 2 actions with duration for this test")
        
        # Take the first two
        action1, action2 = actions_with_duration[:2]
        
        # Perform actions in sequence
        self.master.perform_action(action1)
        self.master.perform_action(action2)
        
        # Verify both are tracked as active
        self.assertEqual(len(self.master.active_actions), 2)
        active_names = {a["action"].name for a in self.master.active_actions}
        self.assertEqual(active_names, {action1.name, action2.name})
    
    def test_perform_action_by_name(self):
        """Test that actions can be performed by name lookup."""
        # Test with every registered action
        for action in self.master.actions:
            result = self.master.perform_action_by_name(action.name)
            self.assertIsInstance(result, dict)
            
        # Test with non-existent action
        result = self.master.perform_action_by_name("NonexistentAction")
        self.assertEqual(result, {})
    
    def test_action_kwargs_handling(self):
        """Test that actions properly handle additional parameters."""
        # Test both with and without kwargs for each action
        for action in self.master.actions:
            # Should handle no kwargs
            try:
                self.master.perform_action(action)
            except Exception as e:
                self.fail(f"Action {action.name} should handle missing kwargs: {e}")
            
            # Should handle arbitrary kwargs (actions should ignore unknown kwargs)
            try:
                self.master.perform_action(action, 
                                         dose_multiplier=2.0,
                                         arbitrary_kwarg="test")
            except Exception as e:
                self.fail(f"Action {action.name} should handle arbitrary kwargs: {e}")


if __name__ == "__main__":
    unittest.main()