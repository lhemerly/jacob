"""
Integration test demonstrating how to run the simulation with all available solvers.
Tests the interaction between different solvers and couplers in the simulation.
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

    # Create the master simulation
    sim = Master(solvers=solvers, dt=1.0, couplers=couplers)

    # Run initial steps to establish baseline
    print("\n--- Running initial 5 steps ---")
    for step_i in range(5):
        sim.step()
        print(f"\nStep {step_i+1} completed")
        print_vital_stats(sim)
        
    # Store baseline values to calculate relative changes
    baseline = {}
    for key in ["blood_pressure", "kidney_function", "fluid_volume", "tss_severity", "tissue_damage", "platelets", "wbc"]:
        baseline[key] = sim.state.get(key, 0)

    # Test various clinical scenarios
    test_scenarios = [
        {
            "name": "Septic shock simulation",
            "actions": {
                "epinephrine": 5.0,  # Increase epinephrine
                "fluid_volume": 1000.0,  # Add fluid
                "tss_severity": 50.0,  # Increase severity
                "tissue_damage": 40.0,  # Increase damage
                "temperature": 39.2,   # Fever
                "infection_level": 60.0  # Infection
            }
        },
        {
            "name": "Hemorrhagic shock",
            "actions": {
                "platelets": 80.0,        # Low platelets
                "bleeding_rate": 10.0,    # Active bleeding
                "inr": 1.8,               # Coagulopathy
                "fluid_volume": -500.0    # Decrease fluid
            }
        },
        {
            "name": "Acute kidney injury",
            "actions": {
                # Set blood_pressure to around 60 by applying a relative change
                "blood_pressure": baseline["blood_pressure"] - 90.0 if "blood_pressure" in baseline else -30.0,
                "fluid_volume": -800.0,  # Decrease fluid
                # Reduce kidney function to around 40% by applying a negative change
                "kidney_function": -60.0  # Decrease function by 60%
            }
        },
        {
            "name": "Recovery phase",
            "actions": {
                "fluid_volume": 500.0,  # Add fluid
                "epinephrine": -2.0,  # Decrease epinephrine
                "tss_severity": -20.0,  # Decrease severity
                "kidney_function": 20.0,  # Increase function by 20%
                "platelets": 50.0,     # Improve platelets
                "bleeding_rate": -5.0, # Reduce bleeding
                "temperature": -0.8    # Reduce fever
            }
        }
    ]

    # Run through each scenario
    for scenario in test_scenarios:
        print(f"\n--- Testing scenario: {scenario['name']} ---")
        
        # Log the action values being applied for debugging
        print("Applying actions:")
        for key, value in scenario["actions"].items():
            current = sim.state.get(key, "N/A")
            if isinstance(current, (int, float)):
                print(f"  {key}: current={current:.1f}, change={value:.1f}")
            else:
                print(f"  {key}: current={current}, change={value}")
        
        sim.actions(scenario["actions"])
        
        # Print state immediately after actions are applied
        print("\nState immediately after actions:")
        print_vital_stats(sim)
        
        for step_i in range(5):
            sim.step()
            print(f"\nStep {step_i+1} after {scenario['name']}")
            print_vital_stats(sim)


def print_vital_stats(sim):
    """Print key physiological parameters from the simulation."""
    stats = [
        ("Heart Rate", "heart_rate", "bpm"),
        ("Blood Pressure", "blood_pressure", "mmHg"),
        ("Oxygen Saturation", "oxygen_saturation", "%"),
        ("Fluid Volume", "fluid_volume", "mL"),
        ("TSS Severity", "tss_severity", "%"),
        ("Tissue Damage", "tissue_damage", "%"),
        ("Toxin Level", "toxin_level", "%"),
        ("Immune Response", "immune_response", "%"),
        ("Kidney Function", "kidney_function", "%"),
        ("Urine Output", "urine_output", "mL/hr"),
        ("Temperature", "temperature", "°C"),
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
        ("Epinephrine", "epinephrine", "µg/min")
    ]
    
    for name, key, unit in stats:
        value = sim.state.get(key, "N/A")
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.1f} {unit}")
        else:
            print(f"  {name}: {value} {unit}")


if __name__ == "__main__":
    main()