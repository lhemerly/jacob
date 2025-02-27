"""
Benchmark script for testing simulation performance with all available solvers.
Tests parallel simulation instances to measure throughput and performance.
"""

import logging
logging.disable(logging.CRITICAL)

import time
import concurrent.futures
import sys
from pathlib import Path
import taichi as ti

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
from solvers.crp import CRPSolver
from solvers.rhythm import RhythmSolver
from solvers.sedation import SedationSolver

from couplers.meds_vitals import MedsVitalsCoupler
from couplers.fluid_electrolytes import FluidElectrolyteCoupler
from couplers.fever_metabolic import FeverMetabolicCoupler
from couplers.infection_hemogram import InfectionHemogramCoupler
from couplers.coagulation_fluid import CoagulationFluidCoupler

# Configuration for benchmark
NUM_SIMULATIONS = 50      # Number of parallel simulation instances
N_STEPS = 1000           # Number of simulation steps per instance
SCENARIOS = [
    {
        "name": "Baseline",
        "actions": {}
    },
    {
        "name": "Septic shock",
        "actions": {
            "epinephrine": 5.0,
            "fluid_volume": 1000.0,
            "tss_severity": 50.0,
            "tissue_damage": 40.0
        }
    },
    {
        "name": "Renal failure",
        "actions": {
            "blood_pressure": -30.0,  # Decrease blood pressure
            "kidney_function": -60.0  # Decrease function by 60%
        }
    },
    {
        "name": "Hemorrhagic shock",
        "actions": {
            "platelets": 80.0,        # Reduced platelets
            "bleeding_rate": 10.0,    # Active bleeding
            "inr": 1.8,              # Coagulopathy
            "fluid_volume": 1500.0    # Volume depletion
        }
    },
    {
        "name": "Severe infection",
        "actions": {
            "infection_level": 70.0,  # Severe infection
            "temperature": 39.5,      # High fever
            "wbc": 15.0,             # Elevated WBC
            "crp": 180.0             # Elevated CRP
        }
    }
]

def create_simulation():
    """Create a simulation instance with all solvers."""
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
    
    couplers = [
        MedsVitalsCoupler(),
        FluidElectrolyteCoupler(),
        FeverMetabolicCoupler(),
        InfectionHemogramCoupler(),
        CoagulationFluidCoupler()
    ]
    
    return Master(solvers=solvers, dt=1.0, couplers=couplers)

def run_simulation(sim_id):
    """Run a single simulation instance."""
    # Initialize Taichi with GPU if available, otherwise use CPU
    try:
        ti.init(arch=ti.gpu, offline_cache=False)
        if sim_id == 0:  # Only print this once to avoid spamming
            print("Running with GPU acceleration")
    except Exception:
        ti.init(arch=ti.cpu, offline_cache=False)
        if sim_id == 0:  # Only print this once to avoid spamming
            print("Running with CPU (GPU not available)")

    sim = create_simulation()
    scenario_times = {}
    
    for scenario in SCENARIOS:
        start = time.time()
        
        # Apply scenario actions
        sim.actions(scenario["actions"])
        
        # Run simulation steps
        for _ in range(N_STEPS):
            sim.step()
            
        end = time.time()
        real_time = end - start
        simulated_time = N_STEPS * sim.dt
        speedup = simulated_time / real_time if real_time > 0 else float('inf')
        
        scenario_times[scenario["name"]] = (simulated_time, real_time, speedup)
    
    return scenario_times

def main():
    print(f"Running {NUM_SIMULATIONS} parallel simulations")
    print(f"Testing {len(SCENARIOS)} scenarios with {N_STEPS} steps each")
    
    overall_start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, i) for i in range(NUM_SIMULATIONS)]
        
        # Aggregate results by scenario
        scenario_results = {scenario["name"]: [] for scenario in SCENARIOS}
        
        for future in concurrent.futures.as_completed(futures):
            scenario_times = future.result()
            for scenario_name, (sim_time, real_time, speedup) in scenario_times.items():
                scenario_results[scenario_name].append(speedup)
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # Print results for each scenario
    print("\nBenchmark Results:")
    print("-" * 50)
    for scenario_name, speedups in scenario_results.items():
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        print(f"\nScenario: {scenario_name}")
        print(f"  Average speedup: {avg_speedup:.2f}x real-time")
        print(f"  Min speedup: {min_speedup:.2f}x")
        print(f"  Max speedup: {max_speedup:.2f}x")
    
    print(f"\nTotal benchmark time: {overall_time:.2f} seconds")

if __name__ == '__main__':
    main()