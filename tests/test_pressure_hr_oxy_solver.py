import unittest
from solvers.pressure_HR_Oxy import PressureHROxySolver, PressureHROxyState
import logging

# Suppress logging for tests if not needed, or set to DEBUG to see solver logs
# logging.basicConfig(level=logging.DEBUG) # Uncomment to see solver logs
# logging.getLogger("solvers.pressure_HR_Oxy").setLevel(logging.WARNING)


class TestPressureHROxySolver(unittest.TestCase):

    def _get_map(self, systolic, diastolic):
        return (systolic + 2 * diastolic) / 3

    def test_baseline_stability(self):
        solver = PressureHROxySolver() # Uses default sv_to_systolic_factor=0.5, svr_to_diastolic_factor=50.0
        initial_hr = 75.0
        initial_systolic = 115.0 # Start SBP
        initial_diastolic = 75.0  # Start DBP
        initial_map = self._get_map(initial_systolic, initial_diastolic) # Approx 88.33
        initial_edv = solver.target_edv # 120.0

        initial_data = {
            "systolic_bp": initial_systolic,
            "diastolic_bp": initial_diastolic,
            "heart_rate": initial_hr,
            "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        
        dt = 1.0

        # Calculate expected SV (preload and afterload effects should be minimal here if MAP is close to setpoint)
        # SV = base_sv + k_preload*(edv - target_edv) - k_afterload*(map_old - map_setpoint)
        # SV = 70 + 0.5*(120-120) - 0.3*(88.33 - 90) = 70 - 0.3*(-1.67) = 70 + 0.5 = 70.5
        expected_sv = solver.base_stroke_volume + \
                      solver.k_preload * (initial_edv - solver.target_edv) - \
                      solver.k_afterload * (initial_map - solver.map_setpoint)
        expected_sv = max(5.0, min(expected_sv, solver.max_stroke_volume))


        # Expected targets
        # systolic_target = diastolic_old + sv_to_systolic_factor * stroke_volume_actual
        # diastolic_target = base_diastolic_reference + svr_to_diastolic_factor * (svr - 1.0)
        expected_systolic_target = initial_diastolic + solver.sv_to_systolic_factor * expected_sv
        expected_diastolic_target = 70.0 + solver.svr_to_diastolic_factor * (solver.svr - 1.0) # SVR is 1.0 by default

        new_state_obj = solver.solve(initial_data.copy(), dt)
        new_state_dict = new_state_obj.state

        # SBP should move towards its target
        if expected_systolic_target > initial_systolic:
            self.assertTrue(new_state_dict["systolic_bp"] > initial_systolic, "SBP should increase towards target")
        elif expected_systolic_target < initial_systolic:
            self.assertTrue(new_state_dict["systolic_bp"] < initial_systolic, "SBP should decrease towards target")
        
        # DBP should move towards its target
        if expected_diastolic_target > initial_diastolic:
            self.assertTrue(new_state_dict["diastolic_bp"] > initial_diastolic, "DBP should increase towards target")
        elif expected_diastolic_target < initial_diastolic:
            self.assertTrue(new_state_dict["diastolic_bp"] < initial_diastolic, "DBP should decrease towards target")

        self.assertAlmostEqual(new_state_dict["end_diastolic_volume"], initial_edv, delta=1.0, msg="EDV should be stable at target_edv if filling_ratio=1")
        
        # HR should change based on baroreflex: map_setpoint (90) - initial_map (88.33) = 1.67
        # dhr_dt = gain * diff = 0.1 * 1.67 = 0.167. So HR should increase slightly.
        self.assertTrue(new_state_dict["heart_rate"] > initial_hr, msg="HR should increase slightly as MAP is below setpoint")
        self.assertTrue(new_state_dict["systolic_bp"] > new_state_dict["diastolic_bp"])


    def test_increased_preload_increases_sv_effect(self):
        solver = PressureHROxySolver()
        initial_edv_baseline = solver.target_edv # 120
        initial_edv_increased = initial_edv_baseline + 40.0 # 160 mL

        base_data = {
            "systolic_bp": 115.0, "diastolic_bp": 75.0, # MAP ~ 88.33
            "heart_rate": 75.0, "oxy_saturation": 98.0, "oxygen_debt": 0.0
        }

        initial_data_baseline = base_data.copy()
        initial_data_baseline["end_diastolic_volume"] = initial_edv_baseline
        
        initial_data_increased_preload = base_data.copy()
        initial_data_increased_preload["end_diastolic_volume"] = initial_edv_increased

        dt = 1.0
        state_baseline = solver.solve(initial_data_baseline.copy(), dt).state
        state_increased_preload = solver.solve(initial_data_increased_preload.copy(), dt).state

        # Increased preload (EDV) should lead to higher SV, thus higher SBP primarily
        delta_sbp_baseline = state_baseline["systolic_bp"] - base_data["systolic_bp"]
        delta_dbp_baseline = state_baseline["diastolic_bp"] - base_data["diastolic_bp"]
        
        delta_sbp_increased = state_increased_preload["systolic_bp"] - base_data["systolic_bp"]
        delta_dbp_increased = state_increased_preload["diastolic_bp"] - base_data["diastolic_bp"]

        self.assertTrue(state_increased_preload["systolic_bp"] > state_baseline["systolic_bp"],
                        "Systolic BP with increased preload should be higher than baseline's new SBP")
        # Check that the change in SBP is more significant than DBP for the increased preload case
        self.assertTrue(abs(delta_sbp_increased) > abs(delta_dbp_increased) or abs(delta_sbp_increased) > 0.1, # ensure some change if DBP is flat
                        "With increased preload, SBP change should be more pronounced than DBP change.")
        self.assertTrue(state_baseline["systolic_bp"] > state_baseline["diastolic_bp"])
        self.assertTrue(state_increased_preload["systolic_bp"] > state_increased_preload["diastolic_bp"])


    def test_decreased_preload_decreases_sv_effect(self):
        solver = PressureHROxySolver()
        initial_edv_baseline = solver.target_edv # 120
        initial_edv_decreased = initial_edv_baseline - 40.0 # 80 mL

        base_data = {
            "systolic_bp": 115.0, "diastolic_bp": 75.0, # MAP ~ 88.33
            "heart_rate": 75.0, "oxy_saturation": 98.0, "oxygen_debt": 0.0
        }

        initial_data_baseline = base_data.copy()
        initial_data_baseline["end_diastolic_volume"] = initial_edv_baseline

        initial_data_decreased_preload = base_data.copy()
        initial_data_decreased_preload["end_diastolic_volume"] = initial_edv_decreased
        
        dt = 1.0
        state_baseline = solver.solve(initial_data_baseline.copy(), dt).state
        state_decreased_preload = solver.solve(initial_data_decreased_preload.copy(), dt).state
        
        delta_sbp_decreased = state_decreased_preload["systolic_bp"] - base_data["systolic_bp"]
        delta_dbp_decreased = state_decreased_preload["diastolic_bp"] - base_data["diastolic_bp"]

        self.assertTrue(state_decreased_preload["systolic_bp"] < state_baseline["systolic_bp"],
                        "Systolic BP with decreased preload should be lower than baseline's new SBP")
        self.assertTrue(abs(delta_sbp_decreased) > abs(delta_dbp_decreased) or abs(delta_sbp_decreased) > 0.1,
                        "With decreased preload, SBP change should be more pronounced than DBP change.")
        self.assertTrue(state_baseline["systolic_bp"] > state_baseline["diastolic_bp"])
        self.assertTrue(state_decreased_preload["systolic_bp"] > state_decreased_preload["diastolic_bp"])


    def test_increased_afterload_decreases_sv_effect(self):
        # This test checks if increased MAP_old (afterload) correctly reduces SV,
        # leading to a different evolution of BP compared to a baseline MAP_old.
        solver = PressureHROxySolver()
        
        map_baseline_start = solver.map_setpoint # 90
        map_increased_afterload_start = solver.map_setpoint + 20 # 110
        
        base_data = {
             "end_diastolic_volume": solver.target_edv, # 120
             "heart_rate": 75.0, "oxy_saturation": 98.0, "oxygen_debt": 0.0
        }

        initial_data_baseline = base_data.copy()
        initial_data_baseline["systolic_bp"] = 110.0 # SBP for MAP=90
        initial_data_baseline["diastolic_bp"] = 80.0  # DBP for MAP=90

        initial_data_increased_afterload = base_data.copy()
        initial_data_increased_afterload["systolic_bp"] = 130.0 # SBP for MAP=110
        initial_data_increased_afterload["diastolic_bp"] = 100.0 # DBP for MAP=110
        
        dt = 1.0
        
        state_baseline_obj = solver.solve(initial_data_baseline.copy(), dt)
        map_new_baseline = state_baseline_obj.state["blood_pressure"]
        sbp_new_baseline = state_baseline_obj.state["systolic_bp"]
        dbp_new_baseline = state_baseline_obj.state["diastolic_bp"]

        state_increased_afterload_obj = solver.solve(initial_data_increased_afterload.copy(), dt)
        map_new_increased_afterload = state_increased_afterload_obj.state["blood_pressure"]
        sbp_new_increased_afterload = state_increased_afterload_obj.state["systolic_bp"]
        dbp_new_increased_afterload = state_increased_afterload_obj.state["diastolic_bp"]
        
        # SV is reduced by higher map_old (110 vs 90).
        # SV_baseline_calc = 70 (base) - 0.3*(90-90) = 70
        # SV_increased_afterload_calc = 70 (base) - 0.3*(110-90) = 70 - 0.3*20 = 70 - 6 = 64
        # This lower SV in the 'increased_afterload' case should generally lead to lower subsequent pressures,
        # or at least a smaller increase if pressures are rising.
        # The exact values are complex due to feedback, but the relative effect of SV should be observable.
        self.assertTrue(map_new_increased_afterload < map_new_baseline + (map_increased_afterload_start - map_baseline_start) - 1.0, # Check relative change
                        "New MAP with initially higher afterload (and thus lower SV) should be comparatively lower.")
        self.assertTrue(sbp_new_baseline > dbp_new_baseline)
        self.assertTrue(sbp_new_increased_afterload > dbp_new_increased_afterload)


    def test_edv_dynamics_filling_ratio_gt_1(self):
        solver = PressureHROxySolver(filling_ratio_factor=1.2)
        initial_edv = solver.target_edv # 120
        initial_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0, 
            "heart_rate": 75.0, "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state
        # dEDV/dt = (HR/60 * SV * (fill_ratio - 1)) + edv_recovery_rate * (target_edv - edv_old)
        # SV at baseline EDV and MAP (approx 90) = base_stroke_volume (70)
        # HR/60 * SV * (1.2 - 1.0) = (75/60) * 70 * 0.2 = 1.25 * 70 * 0.2 = 87.5 * 0.2 = 17.5
        # edv_recovery_rate * (target_edv - edv_old) = 0 since edv_old = target_edv
        # So, dEDV_dt = 17.5. edv_new = 120 + 17.5 * 1 = 137.5
        self.assertAlmostEqual(new_state["end_diastolic_volume"], 137.5, delta=0.1,
                               "EDV should increase with filling_ratio > 1")
        self.assertTrue(new_state["systolic_bp"] > new_state["diastolic_bp"])


    def test_edv_dynamics_filling_ratio_lt_1(self):
        solver = PressureHROxySolver(filling_ratio_factor=0.8)
        initial_edv = solver.target_edv # 120
        initial_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0,
            "heart_rate": 75.0, "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state
        # dEDV/dt = (HR/60 * SV * (fill_ratio - 1)) + edv_recovery_rate * (target_edv - edv_old)
        # SV = 70
        # HR/60 * SV * (0.8 - 1.0) = 1.25 * 70 * (-0.2) = -17.5
        # edv_new = 120 - 17.5 * 1 = 102.5
        self.assertAlmostEqual(new_state["end_diastolic_volume"], 102.5, delta=0.1,
                               "EDV should decrease with filling_ratio < 1")
        self.assertTrue(new_state["systolic_bp"] > new_state["diastolic_bp"])


    def test_stroke_volume_max_clamp_effect(self):
        solver = PressureHROxySolver(
            base_stroke_volume=100.0,
            k_preload=2.0,
            max_stroke_volume=130.0, # Clamp here
            target_edv=120.0,
            map_setpoint=90.0, 
            k_afterload=0.1 
        )
        initial_edv = 140.0 # Preload: k_preload * (140-120) = 2.0 * 20 = 40
        # Calculated SV before clamp: base_sv + 40 = 100 + 40 = 140 mL -> clamped to 130 mL

        initial_sbp = 110.0
        initial_dbp = 80.0 # MAP = 90
        initial_data = {
            "systolic_bp": initial_sbp, "diastolic_bp": initial_dbp, 
            "heart_rate": 60.0, 
            "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        
        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state
        
        # Expected change with SV = 130 mL
        # systolic_target = initial_dbp + sv_factor * SV_clamped = 80 + 0.5 * 130 = 80 + 65 = 145
        # systolic_change_potential = 145 - 110 = 35
        # systolic_rate_of_change = 35 / (compliance * 5.0) = 35 / 5 = 7
        # new_sbp = 110 + 7 * 1 = 117
        self.assertAlmostEqual(new_state["systolic_bp"], 117.0, delta=0.1,
                               "SBP should reflect max_stroke_volume clamping effect on systolic_target")
        self.assertTrue(new_state["systolic_bp"] > new_state["diastolic_bp"])


    def test_stroke_volume_min_clamp_effect(self):
        solver = PressureHROxySolver(
            base_stroke_volume=10.0,
            k_afterload=0.5,
            target_edv=120.0, 
            map_setpoint=90.0,
            k_preload=0.1 
        )
        # SV = base (10) + k_preload*(EDV-target) - k_afterload*(MAP_old - map_setpoint)
        # MAP_old = 120. Afterload term: 0.5 * (120 - 90) = 15
        # Calculated SV before clamp: 10 - 15 = -5 mL. Should be clamped to 5 mL.

        initial_sbp = 160.0
        initial_dbp = 100.0 # MAP = 120
        initial_data = {
            "systolic_bp": initial_sbp, 
            "diastolic_bp": initial_dbp, 
            "heart_rate": 60.0, 
            "oxy_saturation": 98.0,
            "end_diastolic_volume": solver.target_edv, 
            "oxygen_debt": 0.0
        }

        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state

        # Expected change with SV = 5 mL
        # systolic_target = initial_dbp + sv_factor * SV_clamped = 100 + 0.5 * 5 = 100 + 2.5 = 102.5
        # systolic_change_potential = 102.5 - 160 = -57.5
        # systolic_rate_of_change = -57.5 / 5.0 = -11.5
        # new_sbp = 160 - 11.5 * 1 = 148.5
        self.assertAlmostEqual(new_state["systolic_bp"], 148.5, delta=0.1,
                               "SBP should reflect min_stroke_volume clamping effect on systolic_target")
        self.assertTrue(new_state["systolic_bp"] > new_state["diastolic_bp"])


    def test_stroke_volume_effect_on_systolic(self):
        solver = PressureHROxySolver() # Defaults: sv_to_systolic_factor=0.5, compliance=1.0
        dt = 1.0
        base_hr = 60.0
        base_diastolic = 70.0 # Keep DBP somewhat stable initially for isolating SBP effect

        def _run_scenario(edv_offset):
            edv = solver.target_edv + edv_offset
            initial_data = {
                "systolic_bp": 110.0,
                "diastolic_bp": base_diastolic,
                "heart_rate": base_hr,
                "end_diastolic_volume": edv,
                "oxy_saturation": 98.0,
            }
            state = solver.solve(initial_data.copy(), dt).state
            return state["systolic_bp"], state["diastolic_bp"], state

        # Scenario 1: Baseline SV
        sbp1, dbp1, state1 = _run_scenario(0)

        # Scenario 2: Higher SV
        sbp2, dbp2, state2 = _run_scenario(20)
        self.assertTrue(sbp2 > sbp1, "SBP with higher SV should be greater than SBP with baseline SV")
        self.assertAlmostEqual(dbp2, dbp1, delta=2.0, "DBP should remain relatively unchanged with SV change")
        self.assertTrue(state1["systolic_bp"] > state1["diastolic_bp"])
        self.assertTrue(state2["systolic_bp"] > state2["diastolic_bp"])

        # Scenario 3: Lower SV
        sbp3, dbp3, state3 = _run_scenario(-20)
        self.assertTrue(sbp3 < sbp1, "SBP with lower SV should be less than SBP with baseline SV")
        self.assertAlmostEqual(dbp3, dbp1, delta=2.0, "DBP should remain relatively unchanged with SV change")
        self.assertTrue(state3["systolic_bp"] > state3["diastolic_bp"])


    def test_svr_effect_on_diastolic(self):
        solver = PressureHROxySolver() # Defaults: svr_to_diastolic_factor=50.0, compliance=1.0
        dt = 1.0
        base_sbp = 120.0
        base_dbp = 80.0 # Initial DBP
        base_edv = solver.target_edv

        # Scenario 1: Baseline SVR
        solver.svr = 1.0
        initial_data1 = {"systolic_bp": base_sbp, "diastolic_bp": base_dbp, "heart_rate": 75.0, "end_diastolic_volume": base_edv, "oxy_saturation": 98.0}
        state1 = solver.solve(initial_data1.copy(), dt).state
        sbp1, dbp1 = state1["systolic_bp"], state1["diastolic_bp"]

        # Scenario 2: Higher SVR
        solver.svr = 1.2 # 20% increase
        initial_data2 = {"systolic_bp": base_sbp, "diastolic_bp": base_dbp, "heart_rate": 75.0, "end_diastolic_volume": base_edv, "oxy_saturation": 98.0}
        state2 = solver.solve(initial_data2.copy(), dt).state
        sbp2, dbp2 = state2["systolic_bp"], state2["diastolic_bp"]

        self.assertTrue(dbp2 > dbp1, "DBP with higher SVR should be greater than DBP with baseline SVR")
        # SBP might also change due to SVR's effect on afterload (reducing SV) or DBP feedback.
        # diastolic_target for svr=1.0 is 70.0 + 50*(1-1) = 70.0
        # diastolic_target for svr=1.2 is 70.0 + 50*(1.2-1) = 70 + 50*0.2 = 70 + 10 = 80.0
        # Initial DBP is 80. So for SVR=1.2, DBP should stay close to 80 or move slightly towards it.
        # For SVR=1.0, DBP (80) should move towards 70. So dbp2 (SVR 1.2) > dbp1 (SVR 1.0) is expected.
        self.assertTrue(state1["systolic_bp"] > state1["diastolic_bp"])
        self.assertTrue(state2["systolic_bp"] > state2["diastolic_bp"])

        # Scenario 3: Lower SVR
        solver.svr = 0.8 # 20% decrease
        initial_data3 = {"systolic_bp": base_sbp, "diastolic_bp": base_dbp, "heart_rate": 75.0, "end_diastolic_volume": base_edv, "oxy_saturation": 98.0}
        state3 = solver.solve(initial_data3.copy(), dt).state
        sbp3, dbp3 = state3["systolic_bp"], state3["diastolic_bp"]
        
        self.assertTrue(dbp3 < dbp1, "DBP with lower SVR should be less than DBP with baseline SVR")
        # diastolic_target for svr=0.8 is 70.0 + 50*(0.8-1) = 70 - 50*0.2 = 70 - 10 = 60.0
        # So dbp3 (SVR 0.8, target 60) < dbp1 (SVR 1.0, target 70) is expected.
        self.assertTrue(state3["systolic_bp"] > state3["diastolic_bp"])


    def test_epinephrine_effect(self):
        solver = PressureHROxySolver(epi_bp_factor=0.3) # Default epi_bp_factor
        dt = 1.0
        initial_data_no_epi = {
            "systolic_bp": 120.0, "diastolic_bp": 80.0, "heart_rate": 70.0, 
            "end_diastolic_volume": solver.target_edv, "oxy_saturation": 98.0,
            "epinephrine": 0.0 # Explicitly zero
        }
        state_no_epi = solver.solve(initial_data_no_epi.copy(), dt).state
        sbp_no_epi, dbp_no_epi = state_no_epi["systolic_bp"], state_no_epi["diastolic_bp"]

        initial_data_with_epi = initial_data_no_epi.copy()
        initial_data_with_epi["epinephrine"] = 1.0 # Arbitrary unit of epi
        
        state_with_epi = solver.solve(initial_data_with_epi.copy(), dt).state
        sbp_with_epi, dbp_with_epi = state_with_epi["systolic_bp"], state_with_epi["diastolic_bp"]

        # epi_bp_factor (0.3) is split for SBP and DBP, so 0.15 effective factor for each per dt
        # Change due to epi should be positive: (0.15 / (compliance*5)) * dt if it were main driver.
        # More simply, epi adds a positive term to both SBP and DBP calculations.
        self.assertTrue(sbp_with_epi > sbp_no_epi, "SBP should increase with epinephrine")
        self.assertTrue(dbp_with_epi > dbp_no_epi, "DBP should increase with epinephrine")
        self.assertTrue(state_no_epi["systolic_bp"] > state_no_epi["diastolic_bp"])
        self.assertTrue(state_with_epi["systolic_bp"] > state_with_epi["diastolic_bp"])


if __name__ == '__main__':
    unittest.main()
