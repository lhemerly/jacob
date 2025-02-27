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

# Import scenarios
from scenarios import FeverScenario, SepsisScenario

# Configuration for benchmark
NUM_SIMULATIONS = 50      # Number of parallel simulation instances
N_STEPS = 1000           # Number of simulation steps per instance

# Benchmark configurations
BENCHMARK_CONFIGS = [
    {
        "name": "Baseline",
        "actions": {},
        "scenarios": []
    },
    {
        "name": "Actions Only",
        "actions": {
            "epinephrine": 5.0,
            "fluid_volume": 1000.0,
            "tss_severity": 50.0,
            "tissue_damage": 40.0
        },
        "scenarios": []
    },
    {
        "name": "Fever Scenario",
        "actions": {},
        "scenarios": ["Fever"]
    },
    {
        "name": "Sepsis Scenario",
        "actions": {},
        "scenarios": ["Sepsis"]
    },
    {
        "name": "Multiple Scenarios",
        "actions": {
            "fluid_volume": 500.0,  # Add some fluid resuscitation
            "epinephrine": 3.0      # Add some vasopressor support
        },
        "scenarios": ["Fever", "Sepsis"]  # Both scenarios active simultaneously
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
    
    # Create scenario objects with different parameters
    scenarios = [
        FeverScenario(peak_temp=39.5, onset_duration=3600, peak_duration=7200, resolution_duration=5400),
        SepsisScenario(severity=1.5, onset_duration=3600, duration=14400)
    ]
    
    return Master(solvers=solvers, dt=1.0, couplers=couplers, scenarios=scenarios)

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
    config_times = {}
    
    for config in BENCHMARK_CONFIGS:
        start = time.time()
        
        # Reset the simulation instance by creating a new one
        # This ensures each benchmark configuration starts from a clean state
        sim = create_simulation()
        
        # Apply actions specified in the configuration
        if config["actions"]:
            sim.actions(config["actions"])
        
        # Activate scenarios specified in the configuration
        for scenario_name in config["scenarios"]:
            sim.apply_scenario(scenario_name)
        
        # Run simulation steps
        for _ in range(N_STEPS):
            sim.step()
            
        end = time.time()
        real_time = end - start
        simulated_time = N_STEPS * sim.dt
        speedup = simulated_time / real_time if real_time > 0 else float('inf')
        
        config_times[config["name"]] = (simulated_time, real_time, speedup)
    
    return config_times

def main():
    print(f"Running {NUM_SIMULATIONS} parallel simulations")
    print(f"Testing {len(BENCHMARK_CONFIGS)} configurations with {N_STEPS} steps each")
    
    overall_start = time.time()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_simulation, i) for i in range(NUM_SIMULATIONS)]
        
        # Aggregate results by configuration
        config_results = {config["name"]: [] for config in BENCHMARK_CONFIGS}
        
        for future in concurrent.futures.as_completed(futures):
            config_times = future.result()
            for config_name, (sim_time, real_time, speedup) in config_times.items():
                config_results[config_name].append(speedup)
    
    overall_end = time.time()
    overall_time = overall_end - overall_start
    
    # Print results for each configuration
    print("\nBenchmark Results:")
    print("-" * 50)
    for config_name, speedups in config_results.items():
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        print(f"\nConfiguration: {config_name}")
        print(f"  Average speedup: {avg_speedup:.2f}x real-time")
        print(f"  Min speedup: {min_speedup:.2f}x")
        print(f"  Max speedup: {max_speedup:.2f}x")
    
    print(f"\nTotal benchmark time: {overall_time:.2f} seconds")

if __name__ == '__main__':
    main()