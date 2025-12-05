import mujoco
import mujoco.viewer
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from torques import *

@dataclass
class Trajectory:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    bias: np.ndarray

    def __init__(self):
        self.qpos = np.empty((0,9))
        self.qvel = np.empty((0,9))
        self.ctrl = np.empty((0,9))
        self.bias = np.empty((0,9))

def run_mujoco(
    mj_model,
    mj_data,
    func: callable
):
    traj = Trajectory()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        t = 0

        while viewer.is_running():
            start_time = time.time()
            bias = np.zeros(mj_model.nv)
            mujoco.mj_rne(mj_model, mj_data, 0, bias)
            mj_data.ctrl[:] = bias[:7] + func(mj_data.time)
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            t += 1

            elapsed_time = time.time() - start_time
            print(f"time: {elapsed_time}")

            traj.bias = np.concat([traj.bias, [bias]], axis=0)
            ctrl = np.concat([mj_data.ctrl, [0, 0]])
            traj.ctrl = np.concat([traj.ctrl, [ctrl]], axis=0)
            traj.qpos = np.concat([traj.qpos, [mj_data.qpos]], axis=0)
            traj.qvel = np.concat([traj.qvel, [mj_data.qvel]], axis=0)
            time.sleep(0.01)

    return traj

def plot_trajectory(fig, traj):
    ts = np.arange(traj.bias.shape[0])

    # Compute magnitudes
    bias_mag = np.linalg.norm(traj.bias, axis=1)
    ctrl_mag = np.linalg.norm(traj.ctrl, axis=1)
    diff_mag = np.linalg.norm(traj.ctrl - traj.bias, axis=1)

    # Create 1×3 layout inside the given figure object
    axes = fig.subplots(1, 3)

    # -----------------------
    # Panel 1: Torque magnitudes
    # -----------------------
    axes[0].plot(ts, bias_mag, label="‖bias‖")
    axes[0].plot(ts, ctrl_mag, label="‖ctrl‖")
    axes[0].plot(ts, diff_mag, label="‖ctrl - bias‖")
    axes[0].set_title("Torque Magnitudes")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Torque Norm [Nm]")
    axes[0].legend()

    # -----------------------
    # Panel 2: Joint positions
    # -----------------------
    for j in range(7):
        axes[1].plot(ts, traj.qpos[:, j], label=f"q{j+1}")
    axes[1].set_title("Joint Positions")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Position [rad]")
    axes[1].legend(loc="upper right", fontsize=8)

    # -----------------------
    # Panel 3: Joint velocities
    # -----------------------
    for j in range(7):
        axes[2].plot(ts, traj.qvel[:, j], label=f"q̇{j+1}")
    axes[2].set_title("Joint Velocities")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Velocity [rad/s]")
    axes[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()


if __name__ == "__main__":
    pi = np.pi
    HOME = np.array([0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4])
    mj_model = mujoco.MjModel.from_xml_path("../models/panda_mjx.xml")
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:7] = HOME

    # func = lambda t: apply_to_joint(imp, t, joint=4, length=0.5, amp=20.0)
    func = lambda t: apply_to_joint(chirp, t, joint=4, T=20.0)
    traj = run_mujoco(
        mj_model,
        mj_data,
        func
    )

    fig = plt.figure(figsize=(18,5))
    plot_trajectory(fig, traj)
    plt.show()

    exit()