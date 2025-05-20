import unittest
import math
from solvers.pressure_HR_Oxy import PressureHROxySolver, PressureHROxyState

class TestAlveolarOxygenCalculation(unittest.TestCase):
    def setUp(self):
        self.solver = PressureHROxySolver()

    def test_pao2_room_air(self):
        # PAO2 = (FiO2 * (Patm - PH2O)) - (PaCO2 / RQ)
        # PAO2 = (0.21 * (760 - 47)) - (40 / 0.8) = (0.21 * 713) - 50 = 149.73 - 50 = 99.73
        expected_pao2 = (0.21 * (760 - 47)) - (40 / 0.8)
        self.assertAlmostEqual(
            self.solver._calculate_alveolar_oxygen(fio2=0.21),
            expected_pao2,
            places=2
        )

    def test_pao2_high_fio2(self):
        # PAO2 = (1.0 * (760 - 47)) - (40 / 0.8) = (1.0 * 713) - 50 = 713 - 50 = 663.0
        expected_pao2 = (1.0 * (760 - 47)) - (40 / 0.8)
        self.assertAlmostEqual(
            self.solver._calculate_alveolar_oxygen(fio2=1.0),
            expected_pao2,
            places=2
        )

    def test_pao2_high_altitude(self):
        # PAO2 = (0.21 * (500 - 47)) - (40 / 0.8) = (0.21 * 453) - 50 = 95.13 - 50 = 45.13
        patm_high_altitude = 500
        expected_pao2 = (0.21 * (patm_high_altitude - 47)) - (40 / 0.8)
        self.assertAlmostEqual(
            self.solver._calculate_alveolar_oxygen(fio2=0.21, patm_mmHg=patm_high_altitude),
            expected_pao2,
            places=2
        )

class TestSpO2Calculation(unittest.TestCase):
    def setUp(self):
        self.solver = PressureHROxySolver()
        self.hill_k = 26.0
        self.hill_n = 2.7

    def _expected_spo2(self, pao2):
        if pao2 < 0: return 0.0
        pao2_n = math.pow(pao2, self.hill_n)
        k_n = math.pow(self.hill_k, self.hill_n)
        spo2 = (pao2_n / (pao2_n + k_n)) * 100.0
        return max(0.0, min(100.0, spo2))

    def test_spo2_with_pao2_100(self):
        pao2 = 100.0
        expected = self._expected_spo2(pao2) # Approx 97.53%
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2), expected, places=2)

    def test_spo2_with_pao2_60(self):
        pao2 = 60.0
        expected = self._expected_spo2(pao2) # Approx 89.93%
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2), expected, places=2)

    def test_spo2_with_pao2_26_p50(self):
        pao2 = 26.0 # P50
        expected = self._expected_spo2(pao2) # Should be 50.0%
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2), 50.0, places=2)

    def test_spo2_with_low_pao2_10(self):
        pao2 = 10.0
        expected = self._expected_spo2(pao2) # Approx 4.8%
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2), expected, places=2)
        self.assertTrue(self.solver._calculate_spO2(pao2) >= 0)


    def test_spo2_with_high_pao2_600(self):
        pao2 = 600.0
        expected = self._expected_spo2(pao2) # Approx 99.99... should be clamped or very close to 100
        # The Hill equation will give values very close to 100 but not exactly 100.
        #The implementation clamps it.
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2), 100.0, places=2)


    def test_spo2_clamping_above_100(self):
        # Simulate a PAO2 that would mathematically result in >100% if not clamped
        # This is theoretical as Hill equation naturally asymptotes at 100
        # but _calculate_spO2 has explicit clamping
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2=1000), 100.0, places=4)

    def test_spo2_clamping_below_0(self):
        # PAO2 can be negative if FiO2 is very low, e.g. FiO2=0.05
        # PAO2 = (0.05 * (760-47)) - 50 = 35.65 - 50 = -14.35
        # The _calculate_spO2 method handles pao2 < 0 and returns 0.0
        self.assertAlmostEqual(self.solver._calculate_spO2(pao2=-10.0), 0.0, places=4)

class TestPressureHROxySolverIntegration(unittest.TestCase):
    def setUp(self):
        self.initial_state_dict = {
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "heart_rate": 70.0,
            "oxy_saturation": 98.0, # This will be overridden by calculation
            "oxygen_debt": 0.0,
            "respiratory_rate": 12.0, # Default
            "tidal_volume": 0.5,    # Default
            # fio2 will be set per test
        }

    def test_solve_room_air_fio2_0_21(self):
        solver = PressureHROxySolver() # Default min_oxy=75, max_oxy=100
        state = self.initial_state_dict.copy()
        state["fio2"] = 0.21

        # Expected PAO2 for FiO2 0.21 is ~99.73 mmHg
        # Expected SpO2 for PAO2 ~99.73 is ~97.5%
        # This is within default solver min_oxy (75) and max_oxy (100)
        
        # Calculate expected SpO2 for assertion
        pao2_expected = solver._calculate_alveolar_oxygen(fio2=0.21)
        spo2_expected = solver._calculate_spO2(pao2_expected)
        
        new_state_obj = solver.solve(state, dt=1.0)
        self.assertIsInstance(new_state_obj, PressureHROxyState)
        self.assertAlmostEqual(new_state_obj.state["oxy_saturation"], spo2_expected, places=2)
        self.assertTrue(new_state_obj.state["oxy_saturation"] > 95.0)


    def test_solve_high_fio2_0_50(self):
        solver = PressureHROxySolver()
        state = self.initial_state_dict.copy()
        state["fio2"] = 0.50

        # Expected PAO2 for FiO2 0.50: (0.50 * 713) - 50 = 356.5 - 50 = 306.5 mmHg
        # Expected SpO2 for PAO2 ~306.5 is very high, likely >99%
        pao2_expected = solver._calculate_alveolar_oxygen(fio2=0.50)
        spo2_expected = solver._calculate_spO2(pao2_expected)

        new_state_obj = solver.solve(state, dt=1.0)
        self.assertIsInstance(new_state_obj, PressureHROxyState)
        self.assertAlmostEqual(new_state_obj.state["oxy_saturation"], spo2_expected, places=2)
        self.assertTrue(new_state_obj.state["oxy_saturation"] > 98.0) # Should be very close to 100%

    def test_solve_low_fio2_min_oxy_clamping(self):
        # FiO2 = 0.10 --> PAO2 = (0.10 * 713) - 50 = 71.3 - 50 = 21.3 mmHg
        # SpO2 for PAO2 21.3 mmHg: (21.3^2.7 / (21.3^2.7 + 26^2.7)) * 100 approx 36.8%
        # This is below the default min_oxy of 75. Let's set min_oxy to 10.0
        # The calculated SpO2 (36.8%) should be higher than this new min_oxy (10.0),
        # so it should not clamp to 10.0 but to 36.8%.
        # Let's use an even lower FiO2 to test clamping.
        # FiO2 = 0.05 --> PAO2 = (0.05 * 713) - 50 = 35.65 - 50 = -14.35 mmHg
        # SpO2 for PAO2 -14.35 mmHg is 0.0% (due to _calculate_spO2 handling negative PAO2)
        # This 0.0% is below the solver's min_oxy=10.0, so it should clamp to 10.0.

        solver = PressureHROxySolver(min_oxy=10.0, max_oxy=100.0) # Set custom min_oxy
        state = self.initial_state_dict.copy()
        state["fio2"] = 0.05 # This will result in PAO2 < 0, so SpO2 calculation will return 0

        pao2_calculated = solver._calculate_alveolar_oxygen(fio2=0.05) # Will be negative
        spo2_calculated_by_hill = solver._calculate_spO2(pao2_calculated) # Will be 0.0
        
        # The solver's solve method will then clamp this 0.0 to solver.min_oxy (10.0)
        
        new_state_obj = solver.solve(state, dt=1.0)
        self.assertIsInstance(new_state_obj, PressureHROxyState)
        
        # Check that the internal calculation before solver clamping is indeed 0
        self.assertAlmostEqual(spo2_calculated_by_hill, 0.0, places=2)
        # And that the final state is clamped by solver's min_oxy
        self.assertAlmostEqual(new_state_obj.state["oxy_saturation"], solver.min_oxy, places=2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
