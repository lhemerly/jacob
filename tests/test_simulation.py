"""
Integration test demonstrating how to run the simulation with all available solvers.
Tests the interaction between different solvers, couplers, and scenarios in the simulation.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

import taichi as ti

# Try to initialize with GPU, fall back to CPU if GPU is not available
try:
    ti.init(arch=ti.gpu)
    print("Running simulation with GPU acceleration")
except Exception as e:
    print(f"GPU initialization failed ({str(e)}), falling back to CPU")
    ti.init(arch=ti.cpu)

from master import Master
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

from couplers.meds_vitals import MedsVitalsCoupler
from couplers.fluid_electrolytes import FluidElectrolyteCoupler
from couplers.fever_metabolic import FeverMetabolicCoupler
from couplers.infection_hemogram import InfectionHemogramCoupler
from couplers.coagulation_fluid import CoagulationFluidCoupler

# Import scenarios
from scenarios import FeverScenario, SepsisScenario


def main():
    # Set logging level
    logging.basicConfig(level=logging.DEBUG)

    # Instantiate all solvers
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
    
    # Instantiate coupler objects
    couplers = [
        MedsVitalsCoupler(),
        FluidElectrolyteCoupler(),
        FeverMetabolicCoupler(),
        InfectionHemogramCoupler(),
        CoagulationFluidCoupler()
    ]
    
    # Create scenario objects with different parameters
    scenarios = [
        FeverScenario(peak_temp=39.2, onset_duration=1800, peak_duration=3600, resolution_duration=2700),  # Rapid onset fever
        FeverScenario(peak_temp=40.5, onset_duration=3600, peak_duration=7200, resolution_duration=5400),  # High fever
        SepsisScenario(severity=0.6, onset_duration=3600, duration=10800),  # Mild sepsis
        SepsisScenario(severity=1.8, onset_duration=1800, duration=14400),  # Severe sepsis
    ]

    # Create the master simulation
    sim = Master(solvers=solvers, dt=1.0, couplers=couplers, scenarios=scenarios)

    # Run initial steps to establish baseline
    print("\n--- Running initial 5 steps ---")
    for step_i in range(5):
        sim.step()
        print(f"\nStep {step_i+1} completed")
        print_vital_stats(sim)
    
    # Test various clinical scenarios using both direct actions and scenarios
    
    # First, demonstrate immediate actions (medications, procedures)
    print("\n--- Testing immediate actions ---")
    print("Applying fluid bolus and vasopressor")
    
    actions = {
        "fluid_volume": 500.0,  # Add 500mL of fluid
        "epinephrine": 2.0,     # Start epinephrine infusion
        "sedation_level": 3.0   # Moderate sedation
    }
    
    sim.actions(actions)
    
    # Print state immediately after actions are applied
    print("\nState immediately after actions:")
    print_vital_stats(sim)
    
    # Simulate a few steps to allow actions to take effect
    for step_i in range(5):
        sim.step()
        print(f"\nStep {step_i+1} after actions")
        print_vital_stats(sim)
    
    # Now use the scenario system for complex, time-evolving conditions
    print("\n--- Testing mild fever scenario ---")
    sim.apply_scenario("Fever")  # Activate the first fever scenario
    
    # Run simulation with active scenario
    for step_i in range(10):
        sim.step()
        print(f"\nStep {step_i+1} with mild fever scenario")
        print_vital_stats(sim)
        # Show active scenarios
        print(f"Active scenarios: {[s.name for s in sim.active_scenarios]}")
        
    # Deactivate fever and activate sepsis
    sim.deactivate_scenario("Fever")
    print("\n--- Testing severe sepsis scenario ---")
    
    # Use the severe sepsis scenario (second in the list)
    sim.apply_scenario("Sepsis")
    
    # Run simulation with active scenario
    for step_i in range(20):
        sim.step()
        print(f"\nStep {step_i+1} with severe sepsis scenario")
        print_vital_stats(sim)
        # Show active scenarios
        print(f"Active scenarios: {[s.name for s in sim.active_scenarios]}")
        
        # Apply treatment actions during sepsis
        if step_i == 10:  # Apply treatment halfway through
            print("\n--- Applying treatment actions during sepsis ---")
            treatment_actions = {
                "fluid_volume": 1000.0,  # Fluid resuscitation
                "epinephrine": 5.0,      # Increase vasopressors
                "antibiotics": 10.0      # Add antibiotics
            }
            sim.actions(treatment_actions)
    
    # Test having multiple scenarios active simultaneously
    print("\n--- Testing multiple concurrent scenarios ---")
    sim.apply_scenario("Fever")  # Add fever on top of sepsis
    
    for step_i in range(10):
        sim.step()
        print(f"\nStep {step_i+1} with multiple scenarios")
        print_vital_stats(sim)
        # Show active scenarios
        print(f"Active scenarios: {[s.name for s in sim.active_scenarios]}")
    
    # Deactivate all scenarios and return to baseline
    for scenario in list(sim.active_scenarios):
        sim.deactivate_scenario(scenario.name)
    
    print("\n--- Returning to baseline ---")
    recovery_actions = {
        "fluid_volume": 500.0,  # Add more fluid
        "epinephrine": -5.0,    # Decrease vasopressors
        "antibiotics": 5.0,     # Continue antibiotics
        "lactate": -2.0,        # Decreasing lactate
        "crp": -150.0           # Decreasing CRP
    }
    sim.actions(recovery_actions)
    
    for step_i in range(5):
        sim.step()
        print(f"\nStep {step_i+1} recovery phase")
        print_vital_stats(sim)


def print_vital_stats(sim):
    """Print key physiological parameters from the simulation."""
    stats = [
        ("Heart Rate", "heart_rate", "bpm"),
        ("Systolic BP", "systolic_bp", "mmHg"),
        ("Diastolic BP", "diastolic_bp", "mmHg"),
        ("Mean Arterial Pressure", "blood_pressure", "mmHg"),  # Now labeled as MAP for clarity
        ("Oxygen Saturation", "oxygen_saturation", "%"),
        ("Body Temperature", "body_temperature", "°C"),
        ("Fluid Volume", "fluid_volume", "mL"),
        ("TSS Severity", "tss_severity", "%"),
        ("Tissue Damage", "tissue_damage", "%"),
        ("Toxin Level", "toxin_level", "%"),
        ("Immune Response", "immune_response", "%"),
        ("Kidney Function", "kidney_function", "%"),
        ("Urine Output", "urine_output", "mL/hr"),
        ("Metabolic Rate", "metabolic_rate", "x"),
        ("Lactate", "lactate", "mmol/L"),
        ("WBC", "wbc", "K/µL"),
        ("Hemoglobin", "hemoglobin", "g/dL"),
        ("Platelets", "platelets", "K/µL"),
        ("CRP", "crp", "mg/L"),
        ("INR", "inr", "ratio"),
        ("Bleeding Rate", "bleeding_rate", "units"),
        ("Infection Level", "infection_level", "%"),
        ("Sodium", "sodium", "mEq/L"),
        ("Potassium", "potassium", "mEq/L"),
        ("Epinephrine", "epinephrine", "µg/min"),
        ("Respiratory Rate", "respiratory_rate", "breaths/min"),
    ]
    
    for name, key, unit in stats:
        value = sim.state.get(key, "N/A")
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.1f} {unit}")
        else:
            print(f"  {name}: {value} {unit}")


if __name__ == "__main__":
    main()