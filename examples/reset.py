#!/usr/bin/env python3
# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE
import argparse
import numpy as np
from pylibfranka import ControllerMode, JointPositions, Robot
from fr3_mpc import HOME, ALT_HOME


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    parser.add_argument("--duration", type=float, default=3.0, help="Motion duration in seconds")
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip)
    
    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        
        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        print("WARNING: This example will move the robot!")
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")

        # Use JointImpedance mode instead - better for pure joint position control
        active_control = robot.start_joint_position_control(
            ControllerMode.JointImpedance
        )

        time_elapsed = 0.0
        motion_finished = False
        motion_duration = args.duration
        initial_position = None
        # target_position = np.array(ALT_HOME)
        target_position = np.array(HOME)
        
        print(f"Moving to HOME position over {motion_duration} seconds...")

        # Control loop
        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()

            # Always use q_d (desired/commanded position)
            current_position = np.array(robot_state.q_d)

            # Initialize on first iteration
            if initial_position is None:
                initial_position = current_position.copy()
                print(f"Initial joints: {initial_position}")
                print(f"Target joints:  {target_position}")
                print(f"Max joint delta: {np.max(np.abs(target_position - initial_position)):.4f} rad")
                
                # Echo current position to establish continuity
                new_positions = initial_position.tolist()
            else:
                # Increment time
                time_elapsed += dt
                
                # Smooth interpolation with 5th order polynomial (C² continuous vel & accel)
                # This ensures smooth velocity and acceleration profiles
                s = min(time_elapsed / motion_duration, 1.0)
                
                # 5th order polynomial: 6s^5 - 15s^4 + 10s^3
                alpha = 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3

                # Interpolate
                new_positions = ((1.0 - alpha) * initial_position + alpha * target_position).tolist()

                # Check if done
                if s >= 1.0:
                    motion_finished = True
                    print("Motion complete!")
                    print(f"Final error: {np.max(np.abs(np.array(new_positions) - target_position)):.6f} rad")

            # Send command
            joint_positions = JointPositions(new_positions)
            joint_positions.motion_finished = motion_finished
            active_control.writeOnce(joint_positions)

        print("Successfully reached HOME position")

    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()