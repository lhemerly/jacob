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
        solver = PressureHROxySolver() # Default parameters
        initial_hr = 75.0
        initial_systolic = 120.0
        initial_diastolic = 80.0
        initial_edv = solver.target_edv # Should be 120.0 by default

        initial_data = {
            "systolic_bp": initial_systolic,
            "diastolic_bp": initial_diastolic,
            "heart_rate": initial_hr,
            "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        # Initial MAP should be the setpoint for stability
        # The solver's internal map_setpoint is 90.0.
        # Our initial SBP/DBP (120/80) gives MAP = (120 + 2*80)/3 = 280/3 = 93.33
        # Let's adjust initial SBP/DBP to actually match the solver's map_setpoint (90)
        # If MAP = 90, SBP = 110, DBP = 80 => (110 + 160)/3 = 270/3 = 90
        initial_systolic_at_setpoint = 110.0
        initial_diastolic_at_setpoint = 80.0
        initial_data["systolic_bp"] = initial_systolic_at_setpoint
        initial_data["diastolic_bp"] = initial_diastolic_at_setpoint
        
        # With edv = target_edv and map_old = map_setpoint, stroke_volume_calculated should be base_stroke_volume
        # dmap_dt should be (hr * base_sv - map_setpoint/svr) / compliance
        # If hr * base_sv = map_setpoint/svr, then dmap_dt = 0 and MAP is stable.
        # Default: base_sv=70, hr=75 (1.25bps), CO = 87.5. svr=1. map_setpoint/svr = 90. CO != map_setpoint/svr.
        # So MAP will change. HR will also change due to baroreflex. EDV will also change if filling_ratio != 1.

        dt = 1.0
        new_state_obj = solver.solve(initial_data.copy(), dt) # Use copy to avoid modification
        new_state_dict = new_state_obj.state

        # Assertions:
        # 1. EDV should be relatively stable if edv_old = target_edv and filling_ratio = 1 (default)
        #    dEDV/dt = (HR/60 * SV * (fill_ratio - 1)) + edv_recovery_rate * (target_edv - edv_old)
        #    If edv_old = target_edv and fill_ratio = 1, then dEDV/dt = 0
        self.assertAlmostEqual(new_state_dict["end_diastolic_volume"], initial_edv, delta=1.0, msg="EDV should be stable at target_edv")

        # 2. MAP will change because CO (75*70 = 5250 ml/min) is not perfectly balanced with SVR pressure (90 mmHg)
        #    Let's check that it moves in the expected direction. CO > MAP/SVR (5250 > 90/1), so MAP should increase.
        #    Initial MAP was 90.
        self.assertTrue(new_state_dict["blood_pressure"] > 90.0, msg="MAP should increase slightly")

        # 3. HR should change due to baroreflex (MAP_setpoint - MAP_old)
        #    Since initial MAP (90) is at setpoint, dHR/dt from baroreflex should be 0.
        #    No epi. So HR should be stable.
        self.assertAlmostEqual(new_state_dict["heart_rate"], initial_hr, delta=1.0, msg="HR should be relatively stable")

    def test_increased_preload_increases_sv_effect(self):
        solver = PressureHROxySolver()
        initial_edv_baseline = solver.target_edv
        initial_edv_increased = initial_edv_baseline + 40.0 # 160 mL

        base_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0, # MAP = 90 (setpoint)
            "heart_rate": 75.0, "oxy_saturation": 98.0, "oxygen_debt": 0.0
        }

        initial_data_baseline = base_data.copy()
        initial_data_baseline["end_diastolic_volume"] = initial_edv_baseline
        
        initial_data_increased_preload = base_data.copy()
        initial_data_increased_preload["end_diastolic_volume"] = initial_edv_increased

        dt = 1.0
        state_baseline = solver.solve(initial_data_baseline, dt).state
        state_increased_preload = solver.solve(initial_data_increased_preload, dt).state

        # Increased preload (EDV) should lead to higher SV, thus higher CO, thus higher BP
        self.assertTrue(state_increased_preload["blood_pressure"] > state_baseline["blood_pressure"],
                        "MAP with increased preload should be higher than baseline MAP")
        self.assertTrue(state_increased_preload["systolic_bp"] > state_baseline["systolic_bp"],
                        "Systolic BP with increased preload should be higher")

    def test_decreased_preload_decreases_sv_effect(self):
        solver = PressureHROxySolver()
        initial_edv_baseline = solver.target_edv
        initial_edv_decreased = initial_edv_baseline - 40.0 # 80 mL (must be > 30)

        base_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0, # MAP = 90
            "heart_rate": 75.0, "oxy_saturation": 98.0, "oxygen_debt": 0.0
        }

        initial_data_baseline = base_data.copy()
        initial_data_baseline["end_diastolic_volume"] = initial_edv_baseline

        initial_data_decreased_preload = base_data.copy()
        initial_data_decreased_preload["end_diastolic_volume"] = initial_edv_decreased
        
        dt = 1.0
        state_baseline = solver.solve(initial_data_baseline, dt).state
        state_decreased_preload = solver.solve(initial_data_decreased_preload, dt).state

        self.assertTrue(state_decreased_preload["blood_pressure"] < state_baseline["blood_pressure"],
                        "MAP with decreased preload should be lower than baseline MAP")
        self.assertTrue(state_decreased_preload["systolic_bp"] < state_baseline["systolic_bp"],
                        "Systolic BP with decreased preload should be lower")

    def test_increased_afterload_decreases_sv_effect(self):
        solver = PressureHROxySolver()
        # Baseline: MAP = map_setpoint
        # Increased afterload: MAP_old > map_setpoint
        # SV calc uses map_old: SV = base + k_preload*(EDV-target) - k_afterload*(MAP_old - map_setpoint)
        
        map_baseline = solver.map_setpoint # 90
        map_increased_afterload = solver.map_setpoint + 20 # 110

        # To achieve MAP of 110: SBP=150, DBP=95 => (150 + 190)/3 = 340/3 = 113. Close enough.
        # Or SBP=130, DBP=100 => (130 + 200)/3 = 330/3 = 110
        
        base_data = {
             "end_diastolic_volume": solver.target_edv, # 120
             "heart_rate": 75.0, "oxy_saturation": 98.0, "oxygen_debt": 0.0
        }

        initial_data_baseline = base_data.copy()
        initial_data_baseline["systolic_bp"] = 110.0 # MAP = 90
        initial_data_baseline["diastolic_bp"] = 80.0

        initial_data_increased_afterload = base_data.copy()
        initial_data_increased_afterload["systolic_bp"] = 130.0 # MAP = 110
        initial_data_increased_afterload["diastolic_bp"] = 100.0
        
        dt = 1.0
        
        # Calculate CO for baseline:
        # SV_baseline = base_sv + k_preload*(target_edv - target_edv) - k_afterload*(map_setpoint - map_setpoint) = base_sv (70)
        # CO_baseline = 75 * 70 = 5250
        state_baseline_obj = solver.solve(initial_data_baseline.copy(), dt)
        map_new_baseline = state_baseline_obj.state["blood_pressure"]

        # Calculate CO for increased afterload:
        # SV_increased_afterload = base_sv + 0 - k_afterload*(110 - 90) = 70 - 0.3*20 = 70 - 6 = 64
        # CO_increased_afterload = 75 * 64 = 4800
        state_increased_afterload_obj = solver.solve(initial_data_increased_afterload.copy(), dt)
        map_new_increased_afterload = state_increased_afterload_obj.state["blood_pressure"]
        
        # dMAP/dt = (CO - MAP/SVR)/C.
        # Baseline dMAP/dt approx (5250/60 - 90/1)/1 = (87.5 - 90)/1 = -2.5 (assuming C=1, SVR=1, and converting CO to per sec)
        # Increased Afterload dMAP/dt approx (4800/60 - 110/1)/1 = (80 - 110)/1 = -30
        # So, the new MAP in the "increased afterload" case should be lower than its own starting MAP,
        # and potentially lower than the new MAP from baseline if the drop is significant.
        # The key is that the *change* in MAP (dMAP_dt * dt) is more negative (or less positive).
        
        # New MAP for baseline will be initial_map + dmap_dt*dt = 90 + (-2.5 * 1) = 87.5
        # New MAP for increased afterload = 110 + (-30*1) = 80
        # This means map_new_increased_afterload (80) < map_new_baseline (87.5)
        self.assertAlmostEqual(map_new_baseline, 87.5, delta=0.1) # Based on manual calculation
        self.assertAlmostEqual(map_new_increased_afterload, 80.0, delta=0.1) # Based on manual calculation

        self.assertTrue(map_new_increased_afterload < map_new_baseline,
                        "New MAP with increased afterload should be lower than new MAP from baseline due to SV reduction.")

    def test_edv_dynamics_filling_ratio_gt_1(self):
        solver = PressureHROxySolver(filling_ratio_factor=1.2)
        initial_edv = solver.target_edv # 120
        initial_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0, # MAP = 90
            "heart_rate": 75.0, "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state
        # dEDV/dt = (HR/60 * SV * (fill_ratio - 1)) + edv_recovery_rate * (target_edv - edv_old)
        # SV at baseline EDV and MAP = base_stroke_volume (70)
        # HR/60 * SV * (1.2 - 1.0) = (75/60) * 70 * 0.2 = 1.25 * 70 * 0.2 = 87.5 * 0.2 = 17.5
        # edv_recovery_rate * (target_edv - edv_old) = 0 since edv_old = target_edv
        # So, dEDV_dt = 17.5. edv_new = 120 + 17.5 * 1 = 137.5
        self.assertAlmostEqual(new_state["end_diastolic_volume"], 137.5, delta=0.1,
                               "EDV should increase with filling_ratio > 1")

    def test_edv_dynamics_filling_ratio_lt_1(self):
        solver = PressureHROxySolver(filling_ratio_factor=0.8)
        initial_edv = solver.target_edv # 120
        initial_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0, # MAP = 90
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

    def test_stroke_volume_max_clamp_effect(self):
        solver = PressureHROxySolver(
            base_stroke_volume=100.0,
            k_preload=2.0,
            max_stroke_volume=130.0, # Clamp here
            target_edv=120.0,
            map_setpoint=90.0, # Keep other effects neutral
            k_afterload=0.1 # Minimize afterload effect for this test
        )
        initial_edv = 140.0 # Preload: k_preload * (140-120) = 2.0 * 20 = 40
        # Calculated SV before clamp: base_sv + 40 = 100 + 40 = 140 mL
        # This should be clamped to max_stroke_volume = 130 mL

        initial_data = {
            "systolic_bp": 110.0, "diastolic_bp": 80.0, # MAP = 90 (solver.map_setpoint)
            "heart_rate": 60.0, # Use 60 BPM for easier CO calculation (1 BPS)
            "oxy_saturation": 98.0,
            "end_diastolic_volume": initial_edv,
            "oxygen_debt": 0.0
        }
        
        # Expected CO with clamping: HR_bps * clamped_SV = (60/60) * 130 = 130 mL/s
        # Expected dMAP_dt = (CO_clamped - MAP/SVR) / C
        # = (130 - 90/1) / 1 = 40 (assuming C=1, SVR=1 from defaults)
        # Expected new_map = 90 + 40 * 1.0 = 130

        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state
        self.assertAlmostEqual(new_state["blood_pressure"], 130.0, delta=0.1,
                               "MAP should reflect max_stroke_volume clamping")

    def test_stroke_volume_min_clamp_effect(self):
        solver = PressureHROxySolver(
            base_stroke_volume=10.0,
            k_afterload=0.5,
            target_edv=120.0, # Keep preload neutral
            map_setpoint=90.0,
            k_preload=0.1 # Minimize preload effect
        )
        # SV = base (10) + k_preload*(EDV-target) - k_afterload*(MAP_old - map_setpoint)
        # To trigger min clamp (5mL):
        # Let EDV = target_edv (120mL) -> preload effect = 0
        # We need base_sv - k_afterload*(MAP_old - map_setpoint) < 5
        # 10 - 0.5 * (MAP_old - 90) < 5
        # -0.5 * (MAP_old - 90) < -5
        # (MAP_old - 90) > 10
        # MAP_old > 100. Let MAP_old = 120.
        # Afterload term: 0.5 * (120 - 90) = 0.5 * 30 = 15
        # Calculated SV before clamp: 10 - 15 = -5 mL. Should be clamped to 5 mL.

        initial_map_for_high_afterload = 120.0
        # SBP=160, DBP=100 => (160+200)/3 = 360/3 = 120
        initial_data = {
            "systolic_bp": 160.0, 
            "diastolic_bp": 100.0, # MAP = 120
            "heart_rate": 60.0, # 1 BPS
            "oxy_saturation": 98.0,
            "end_diastolic_volume": solver.target_edv, # 120, no preload effect
            "oxygen_debt": 0.0
        }

        # Expected CO with clamping: HR_bps * clamped_SV = (60/60) * 5 = 5 mL/s
        # Expected dMAP_dt = (CO_clamped - MAP/SVR) / C
        # = (5 - 120/1) / 1 = -115 (assuming C=1, SVR=1 from defaults)
        # Expected new_map = 120 + (-115) * 1.0 = 5

        dt = 1.0
        new_state = solver.solve(initial_data.copy(), dt).state
        self.assertAlmostEqual(new_state["blood_pressure"], 5.0, delta=0.1,
                               "MAP should reflect min_stroke_volume clamping")


if __name__ == '__main__':
    unittest.main()
