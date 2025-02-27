# Medical Simulation Framework

A sophisticated medical simulation framework built with Python and Taichi, designed to simulate various physiological systems and their interactions in parallel.

## Overview

This framework implements a modular, extensible medical simulation system that can handle multiple physiological systems simultaneously. It uses Taichi for parallel computation and provides a flexible architecture for adding new physiological systems and their interactions.

## Architecture

The framework is built on three main components:

### 1. Solvers
Individual solvers handle specific physiological systems:
- Pressure, Heart Rate, and Oxygenation (PressureHROxy)
- Medications (Meds)
- Fluids
- Coagulation
- Drains
- Electrolytes
- Fever
- Hemogram
- Lactate
- Metabolytes
- PCR
- Rhythm
- Sedation
- TSS (Toxic Shock Syndrome)
- Urine

### 2. Couplers
Couplers manage interactions between different physiological systems:
- Medications-Vitals Coupler: Handles how medications affect vital signs
- Fluid-Electrolyte Coupler: Manages interactions between fluid status and electrolyte levels

### 3. Master Simulation
The Master class orchestrates the entire simulation:
- Maintains a global state dictionary
- Coordinates solvers and couplers
- Handles parallel computation using Taichi
- Provides unified logging and time tracking

## Getting Started

### Prerequisites
- Python 3.x
- Taichi
- Additional dependencies (TBD)

### Basic Usage

```python
from master import Master
from solvers.pressure_HR_Oxy import PressureHROxySolver
from solvers.meds import MedsSolver
from solvers.fluids import FluidsSolver
from couplers.meds_vitals import MedsVitalsCoupler
from couplers.fluid_electrolytes import FluidElectrolyteCoupler

# Initialize solvers
phro_solver = PressureHROxySolver()
meds_solver = MedsSolver()
fluids_solver = FluidsSolver()

# Initialize couplers
meds_vitals_coupler = MedsVitalsCoupler()
fluid_electrolyte_coupler = FluidElectrolyteCoupler()

# Create simulation
sim = Master(
    solvers=[phro_solver, meds_solver, fluids_solver],
    dt=1.0,
    couplers=[meds_vitals_coupler, fluid_electrolyte_coupler]
)

# Run simulation steps
sim.step()
```

## Performance

The framework includes benchmarking capabilities:
- Can run multiple parallel simulation instances
- Configurable number of simulation steps
- Measures real-time vs simulated time performance
- Uses Taichi for GPU acceleration (when available)

## Development

### Adding New Solvers
Extend the base `Solver` class and implement:
- `state` property: Define owned state variables
- `solve` method: Implement the system's physics/logic

### Adding New Couplers
Extend the base `Coupler` class and implement:
- `input_keys`: Define required state variables
- `output_keys`: Define modified state variables
- `couple` method: Implement interaction logic

## Testing

The project includes:
- Unit tests in the `tests/` directory
- Example scripts demonstrating usage
- Benchmark tools for performance testing

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

We welcome contributions to the Medical Simulation Framework! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contributing guidelines.

### Code Style

We use industry-standard tools to maintain code quality:

- **Ruff**: For linting and Python code formatting
  ```bash
  pip install ruff
  ruff check .
  ```

- **Black**: For consistent code formatting
  ```bash
  pip install black
  black .
  ```

Please ensure your code passes both ruff and black checks before submitting pull requests.