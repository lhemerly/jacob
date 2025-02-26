"""
Benchmark script for testing simulation performance with all available solvers.
Tests parallel simulation instances to measure throughput and performance.
"""

import logging
logging.disable(logging.CRITICAL)

import time
import concurrent.futures
import os
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
from solvers.pcr import PCRSolver
from solvers.rhythm import RhythmSolver
from solvers.sedation import SedationSolver

from couplers.meds_vitals import MedsVitalsCoupler
from couplers.fluid_electrolytes import FluidElectrolyteCoupler

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
            "tss_severity": 80.0
        }
    },
    {
        "name": "Renal failure",
        "actions": {
            "blood_pressure": 60.0,
            "kidney_function": 30.0
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
        PCRSolver(),
        RhythmSolver(),
        SedationSolver()
    ]
    
    couplers = [
        MedsVitalsCoupler(),
        FluidElectrolyteCoupler()
    ]
    
    return Master(solvers=solvers, dt=1.0, couplers=couplers)

def run_simulation(sim_id):
    """Run a single simulation instance."""
    # Initialize Taichi without caching
    ti.init(arch=ti.cpu, offline_cache=False)

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