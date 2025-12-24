#!/usr/bin/env python3
"""
Minimal import + API sanity test for franka_control
"""

import time
import math

from fr3_mpc.simulation import FR3Simulation, Trajectory
from fr3_mpc.control import FR3Robot


def main():
    print("Import OK")

    # Replace with your robot IP if you actually want to connect
    robot_ip = "172.16.0.2"  # loopback → will fail safely if no robot

    traj = Trajectory()
    try:
        # robot = FR3Robot(robot_ip)
        robot = FR3Simulation("models/franka_fr3_v2/fr3v2.xml")
        print("Controller constructed")

        # Wait for simulation to be ready
        print("Waiting for simulation to start...")
        if not robot.wait_until_ready(timeout=10.0):
            print("ERROR: Simulation failed to start")
            return
        print("Simulation ready!")

        # Send a few dummy torque commands
        t0 = time.time()
        for i in range(2000):
            tau = [0.1 * math.sin(i)] * 7
            ok = robot.push(tau)
            state = robot.read()
            print(f"elapsed = {state.time - t0}")
            traj.append(state)
            print(f"push({tau}) -> {ok}")
            time.sleep(0.01)

        print("Stopping controller")
        robot.stop()

        # Check error reporting (non-RT)
        err = robot.last_error()
        if err:
            print("Controller reported error:")
            print(err)
        else:
            print("No controller error reported")

    except Exception as e:
        robot.stop()
        print("Exception from Python:")
        print(e)

    import matplotlib.pyplot as plt
    fig = traj.plot()
    plt.show()


if __name__ == "__main__":
    main()
