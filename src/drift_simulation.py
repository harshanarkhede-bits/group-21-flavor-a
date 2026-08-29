"""Legacy entrypoint for drift simulation."""
from src.monitoring.drift_simulation import run_drift_simulation_experiment, simulate_operational_drift

if __name__ == "__main__":
    run_drift_simulation_experiment()
