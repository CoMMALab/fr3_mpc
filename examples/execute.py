#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

"""
Joint Impedance Target Example

This example shows a joint impedance controller that renders a spring damper system
to move the robot through a sequence of target joint configurations.
The controller will generate smooth trajectories between positions and run in a continuous loop.
"""

import argparse
import sys
import time

import numpy as np

from pylibfranka import Robot, Torques


class SimpleMotionGenerator:
    """Simple minimum jerk trajectory generator for smooth joint motion."""

    def __init__(self, start_position, end_position, duration=3.0):
        """Initialize the trajectory generator.

        Args:
            start_position: Starting joint positions (array of 7 values)
            end_position: Target joint positions (array of 7 values)
            duration: Duration of the trajectory in seconds
        """
        self.start_position = np.array(start_position)
        self.end_position = np.array(end_position)
        self.duration = duration
        self.elapsed_time = 0.0

    def update(self, dt):
        """Update the trajectory with the elapsed time.
        
        Args:
            dt: Time step in seconds
        """
        self.elapsed_time += dt

    def get_position(self):
        """Get the current target position along the trajectory."""
        s = self._minimum_jerk(min(self.elapsed_time / self.duration, 1.0))
        return self.start_position + s * (self.end_position - self.start_position)

    def is_finished(self):
        """Check if the trajectory is complete."""
        return self.elapsed_time >= self.duration

    def _minimum_jerk(self, t):
        """Minimum jerk trajectory profile (normalized [0,1])."""
        return 10 * (t**3) - 15 * (t**4) + 6 * (t**5)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Define a sequence of target joint configurations
    target_joint_positions = [
        # Home position (slightly bent arm)
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # Extended arm pointing forward
        [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0],
        # Arm pointing to the right
        [0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # Arm pointing to the left
        [-0.5, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
        # Home position again
        [0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.0],
    ]

    # Compliance parameters
    joint_stiffness = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    joint_damping = [2.0 * np.sqrt(k) for k in joint_stiffness]

    try:
        # Connect to robot
        robot = Robot(args.ip)

        # Set default behavior
        robot.set_collision_behavior(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        )

        # Get initial state
        initial_state = robot.read_once()
        current_position = np.array(initial_state.q)

        print("Starting joint impedance control...")
        print(f"Initial position: {current_position}")

        # Start torque control
        active_control = robot.start_torque_control()

        # Create a model instance from robot
        model = robot.load_model()

        # Main control variables
        wait_time = 0.5  # Time to wait at each position before moving to next
        current_wait_time = 0.0

        # Initialize first trajectory
        trajectory_index = 0
        trajectory = SimpleMotionGenerator(
            current_position,
            target_joint_positions[trajectory_index],
            duration=3.0,
        )
        target_reached = False

        print(f"\nMoving to target {trajectory_index + 1}/{len(target_joint_positions)}")

        # Main control loop - runs continuously
        while trajectory_index < len(target_joint_positions):
            # Read robot state - this provides the dt through duration
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()  # Get actual time step from robot

            # Get state variables
            coriolis = np.array(model.coriolis(robot_state))
            q = np.array(robot_state.q)
            dq = np.array(robot_state.dq)

            # Update trajectory timing
            if not target_reached:
                trajectory.update(dt)

            # Get current target from trajectory
            q_goal = trajectory.get_position()

            # Compute error to desired equilibrium joint configuration
            position_error = q - q_goal

            # Compute joint-space impedance control
            tau_task = np.zeros(7)
            for i in range(7):
                tau_task[i] = -joint_stiffness[i] * position_error[i] - joint_damping[i] * dq[i]

            # Add coriolis compensation
            tau_d = tau_task + coriolis

            # Convert to Torques command
            torque_command = Torques(tau_d.tolist())
            torque_command.motion_finished = False
            active_control.writeOnce(torque_command)

            # State machine for trajectory progression
            if not target_reached:
                # Check if trajectory is finished
                if trajectory.is_finished():
                    target_reached = True
                    current_wait_time = 0.0
                    print(f"Target {trajectory_index + 1} reached, waiting...")
            else:
                # Wait at target position
                current_wait_time += dt
                
                if current_wait_time >= wait_time:
                    # Move to next target
                    trajectory_index += 1
                    
                    if trajectory_index < len(target_joint_positions):
                        # Initialize next trajectory
                        current_position = q_goal
                        trajectory = SimpleMotionGenerator(
                            current_position,
                            target_joint_positions[trajectory_index],
                            duration=3.0,
                        )
                        target_reached = False
                        print(f"\nMoving to target {trajectory_index + 1}/{len(target_joint_positions)}")
                    else:
                        print("\nAll targets reached!")
                        break

        print("Motion sequence complete!")

    except Exception as e:
        print(f"\nError occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1

    return 0


if __name__ == "__main__":
    sys.exit(main())