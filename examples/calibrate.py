import argparse
import os
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from fr3_mpc import HOME
from torques import *

@dataclass
class Trajectory:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    bias: np.ndarray

    def __init__(self):
        self.qpos = np.empty((0,7))
        self.qvel = np.empty((0,7))
        self.ctrl = np.empty((0,7))
        self.bias = np.empty((0,7))

    def save(self, path):
        np.savez(path, **self.__dict__)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Trajectory(**{k: data[k] for k in data.files})

    def plot(self):
        fig = plt.figure(figsize=(18,5))
        axes = fig.subplots(1, 3)
        ts = np.arange(traj.bias.shape[0])

        # Panel 1: Torque Magnitudes
        bias_mag = np.linalg.norm(traj.bias, axis=1)
        ctrl_mag = np.linalg.norm(traj.ctrl, axis=1)
        diff_mag = np.linalg.norm(traj.ctrl - traj.bias, axis=1)

        axes[0].plot(ts, bias_mag, label="‖bias‖")
        axes[0].plot(ts, ctrl_mag, label="‖ctrl‖")
        axes[0].plot(ts, diff_mag, label="‖ctrl - bias‖")
        axes[0].set_title("Torque Magnitudes")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Torque Norm [Nm]")
        axes[0].legend()

        # Panel 2: Joint Positions
        for j in range(7):
            axes[1].plot(ts, traj.qpos[:, j], label=f"q{j+1}")
        axes[1].set_title("Joint Positions")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Position [rad]")
        axes[1].legend(loc="upper right", fontsize=8)

        # Panel 3: Joint Velocities
        for j in range(7):
            axes[2].plot(ts, traj.qvel[:, j], label=f"q̇{j+1}")
        axes[2].set_title("Joint Velocities")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Velocity [rad/s]")
        axes[2].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        
        return fig

def run_mujoco(
    mj_model,
    mj_data,
    func: callable,
    horizon: float = 5.0
):
    traj = Trajectory()
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            bias = np.zeros(mj_model.nv)
            mujoco.mj_rne(mj_model, mj_data, 0, bias)
            mj_data.ctrl = func(mj_data.time) + bias
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            traj.bias = np.concat([traj.bias, [        bias]], axis=0)
            traj.ctrl = np.concat([traj.ctrl, [mj_data.ctrl]], axis=0)
            traj.qpos = np.concat([traj.qpos, [mj_data.qpos]], axis=0)
            traj.qvel = np.concat([traj.qvel, [mj_data.qvel]], axis=0)
            if mj_data.time > horizon:
                break
    return traj

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-f", "--func", type=str, default="chirp",
        help="Excitation function (chirp, sin, triangle, impulse, const, lin)"
    )
    parser.add_argument(
        "-j", "--joint", type=int, default=4,
        help="Joint index (0–6)"
    )
    parser.add_argument(
        "-a", "--amp", type=float, default=1.0,
        help="Amplitude of excitation"
    )
    parser.add_argument(
        "--outdir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--horizon", type=float, default=5.0,
        help="Simulation horizon in seconds"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mj_model = mujoco.MjModel.from_xml_path("./models/franka_fr3_v2/scene.xml")
    mj_model.opt.timestep = 0.001
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos = HOME
    func = make_excitation(args.func, args.joint, args.amp)
    traj = run_mujoco(mj_model, mj_data, func, args.horizon)

    # Export data
    ts = np.arange(0, args.horizon, mj_model.opt.timestep)
    cmd = np.array([func(t) for t in ts])
    os.makedirs(args.outdir, exist_ok=True)
    np.savetxt(os.path.join(args.outdir, "cmd.dat"), cmd, fmt="%.6f")
    traj.save(os.path.join(args.outdir, "traj.npz"))
    fig = traj.plot()
    fig.savefig(os.path.join(args.outdir, "traj.png"), dpi=200)
    plt.show()