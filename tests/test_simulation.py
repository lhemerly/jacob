"""
Integration test demonstrating how to run the simulation with all available solvers.
Tests the interaction between different solvers and couplers in the simulation.
"""

import logging
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent))

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
from solvers.pcr import PCRSolver
from solvers.rhythm import RhythmSolver
from solvers.sedation import SedationSolver

from couplers.meds_vitals import MedsVitalsCoupler
from couplers.fluid_electrolytes import FluidElectrolyteCoupler


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
        PCRSolver(),
        RhythmSolver(),
        SedationSolver()
    ]
    
    # Instantiate coupler objects
    couplers = [
        MedsVitalsCoupler(),
        FluidElectrolyteCoupler()
    ]

    # Create the master simulation
    sim = Master(solvers=solvers, dt=1.0, couplers=couplers)

    # Run initial steps to establish baseline
    print("\n--- Running initial 5 steps ---")
    for step_i in range(5):
        sim.step()
        print(f"\nStep {step_i+1} completed")
        print_vital_stats(sim)

    # Test various clinical scenarios
    test_scenarios = [
        {
            "name": "Septic shock simulation",
            "actions": {
                "epinephrine": 5.0,
                "fluid_volume": 1000.0,
                "tss_severity": 80.0,
                "tissue_damage": 60.0
            }
        },
        {
            "name": "Acute kidney injury",
            "actions": {
                "blood_pressure": 60.0,
                "fluid_volume": -800.0,
                "kidney_function": 40.0
            }
        },
        {
            "name": "Recovery phase",
            "actions": {
                "fluid_volume": 500.0,
                "epinephrine": -2.0,
                "tss_severity": -20.0,
                "kidney_function": 20.0
            }
        }
    ]

    # Run through each scenario
    for scenario in test_scenarios:
        print(f"\n--- Testing scenario: {scenario['name']} ---")
        sim.actions(scenario["actions"])
        
        for step_i in range(5):
            sim.step()
            print(f"\nStep {step_i+1} after {scenario['name']}")
            print_vital_stats(sim)


def print_vital_stats(sim):
    """Print key physiological parameters from the simulation."""
    stats = [
        ("Heart Rate", "heart_rate", "bpm"),
        ("Blood Pressure", "blood_pressure", "mmHg"),
        ("Fluid Volume", "fluid_volume", "mL"),
        ("TSS Severity", "tss_severity", "%"),
        ("Kidney Function", "kidney_function", "%"),
        ("Urine Output", "urine_output", "mL/hr"),
        ("Temperature", "temperature", "°C"),
        ("Lactate", "lactate", "mmol/L"),
        ("WBC", "wbc", "K/µL"),
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