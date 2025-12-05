import numpy as np
import mujoco

from hydrax.algs import PredictiveSampling, MPPI
from hydrax.simulation.deterministic import run_interactive
# from hydrax.simulation.asynchronous import run_interactive as run_async
from fr3_mpc.reach_pose import ReachPose
from fr3_mpc.reach_joints import ReachJoints
from fr3_mpc.asynchronous import run_interactive as run_async

"""
Run an interactive simulation of an FE3 reaching task.
"""

pi = np.pi
HOME = np.array([0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4])
HOME_POS = np.array([ 0.307, -0., 0.49])
HOME_ROT = np.diag(np.array([1., -1., -1.]))

# parser = argparse.ArgumentParser(
#     description="Run an interactive simulation of the cube rotation task."
# )
# subparsers = parser.add_subparsers(
#     dest="algorithm", help="Sampling algorithm (choose one)"
# )
# subparsers.add_parser("", type=float, help="Predictive Sampling")
# args = parser.parse_args()

if __name__ == "__main__":

    if False:  # Fails
        task = ReachPose(
            target_pos = HOME_POS,
            target_rot = HOME_ROT
        )
    else:  # Succeeds
        task = ReachJoints(target_q = HOME)

    # Set up the controller
    # ctrl = PredictiveSampling(
    #     task,
    #     num_samples=1024,
    #     noise_level=4.0,
    #     plan_horizon=0.6,
    #     spline_type="cubic",
    #     num_knots=4,
    # )

    ctrl = MPPI(
        task,
        num_samples=1024,
        noise_level=4.0,
        temperature=0.1,
        plan_horizon=0.6,
        spline_type="zero",
        num_knots=4,
    )

    # Define the model used for simulation
    mj_model = task.mj_model
    mj_data = mujoco.MjData(mj_model)

    if True:
        # Tighten up the simulator parameters, since it's running on CPU and
        # therefore won't slow down the planner
        mj_model.opt.timestep = 0.005
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        run_async(
            ctrl,
            mj_model,
            mj_data,
            record_video=True
        )
    else:
        # Run the interactive simulation
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            fixed_camera_id=None,
            show_traces=True,
            max_traces=1,
            record_video=True
        )