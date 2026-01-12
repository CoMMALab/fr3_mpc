#!/usr/bin/env python3
"""
Unified sim-to-real example: calibration/data collection with a single flag.

This script can collect trajectory data from either:
- FR3Simulation (MuJoCo simulator)
- FR3Robot (real Franka robot via C++ bindings)

Simply toggle USE_SIMULATION to switch between sim and real.
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from fr3_mpc import HOME
from fr3_mpc.simulation import FR3Simulation, Trajectory
from torques import make_excitation

plt.rcParams['figure.figsize'] = (36, 10)
plt.style.use('seaborn-v0_8-whitegrid')  # or 'ggplot', 'bmh', etc.
plt.rcParams.update({
    'figure.figsize': (36, 10),
    'axes.titlesize': 24,
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 20,
    'lines.linewidth': 3,
})


# ============================================================================
# Configuration
# ============================================================================

USE_SIMULATION = True  # Set to False for real robot
ROBOT_IP = "172.16.0.2"  # Only used if USE_SIMULATION=False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect trajectory data from simulation or real robot"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real robot instead of simulation"
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default=ROBOT_IP,
        help="Robot IP address (only for real robot)"
    )
    parser.add_argument(
        "-f", "--func",
        type=str,
        default="chirp",
        help="Excitation function (chirp, multisine, sin)"
    )
    parser.add_argument(
        "-j", "--joint",
        type=int,
        default=4,
        help="Joint index (0-6)"
    )
    parser.add_argument(
        "-a", "--amp",
        type=float,
        default=1.0,
        help="Amplitude of excitation"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--horizon",
        type=float,
        default=5.0,
        help="Experiment duration in seconds"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Control timestep in seconds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./models/franka_fr3_v2/fr3v2.xml",
        help="MuJoCo model path (only for simulation)"
    )
    return parser.parse_args()


def run_experiment(robot, func, horizon: float, dt: float):
    """Run data collection experiment.

    Args:
        robot: FR3Simulation or FR3Robot instance
        func: Excitation function (callable)
        horizon: Experiment duration in seconds
        dt: Control timestep in seconds

    Returns:
        Trajectory object with collected data
    """
    traj = Trajectory()
    t0 = time.time()
    iter_count = 0

    print(f"Starting experiment for {horizon} seconds...")
    print("Press Ctrl+C to stop early")

    try:
        while True:
            elapsed = iter_count * dt
            if elapsed >= horizon:
                break

            # Compute excitation torque
            tau = func(elapsed)

            # Send command to robot
            ok = robot.push(tau)
            if not ok:
                print(f"Warning: push() failed at t={elapsed:.3f}s")

            # Read state
            state = robot.read()
            traj.append(state)

            # Periodic status update
            if int(elapsed / 0.5) > int((elapsed - dt) / 0.5):
                print(f"  t = {elapsed:.2f}s / {horizon:.2f}s", end="\r")

            iter_count += 1

            # Deterministic timing: sleep until next timestep
            target_time = t0 + iter_count * dt
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")

    finally:
        print("\nStopping robot...")
        robot.stop()

        # Check for errors
        err = robot.last_error()
        if err:
            print("Robot reported error:")
            print(err)
        else:
            print("No errors reported")

    # Timing diagnostics
    print(f"\n=== Timing Report ===")
    print(f"Samples collected: {len(traj.time)}")

    return traj


def main():
    args = parse_args()

    # Override USE_SIMULATION with command-line flag
    use_simulation = not args.real

    # Create excitation function
    func = make_excitation(args.func, args.joint, args.amp)

    # Initialize robot (simulation or real)
    if use_simulation:
        print(f"Using SIMULATION mode")
        print(f"Model: {args.model}")
        robot = FR3Simulation(args.model, dt=args.dt, qpos0=HOME)
        print("Waiting for simulation to be ready...")
        if not robot.wait_until_ready(timeout=10.0):
            print("ERROR: Simulation failed to start within timeout")
            return 1
        print("Simulation ready!")
    else:
        print(f"Using REAL ROBOT mode")
        print(f"Robot IP: {args.robot_ip}")
        try:
            from fr3_mpc.control import FR3Robot
        except ImportError:
            print("ERROR: FR3Robot not available (C++ module not built)")
            print("Build the project first or use --simulation flag")
            return 1

        robot = FR3Robot(args.robot_ip)

    print(f"Excitation: {args.func} on joint {args.joint} with amplitude {args.amp}")
    print(f"Horizon: {args.horizon}s, dt: {args.dt}s")
    print()

    # Run experiment
    traj = run_experiment(robot, func, args.horizon, args.dt)

    # Save results
    os.makedirs(args.outdir, exist_ok=True)

    # Save trajectory data
    traj_path = os.path.join(args.outdir, "traj.npz")
    traj.save(traj_path)
    print(f"Saved trajectory to {traj_path}")

    # Save command signal
    ts = np.arange(0, args.horizon, args.dt)
    cmd = np.array([func(t) for t in ts])
    cmd_path = os.path.join(args.outdir, "cmd.dat")
    np.savetxt(cmd_path, cmd, fmt="%.6f")
    print(f"Saved command signal to {cmd_path}")

    # Plot and save figure
    print("Generating plots...")
    fig = traj.plot()
    fig_path = os.path.join(args.outdir, "traj.png")
    fig.savefig(fig_path, dpi=200)
    print(f"Saved plot to {fig_path}")

    # Show interactive plot
    plt.show()

    return 0


if __name__ == "__main__":
    exit(main())
