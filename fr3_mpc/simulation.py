import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import multiprocessing as mp

from typing import NamedTuple
from dataclasses import dataclass
from multiprocessing import Event, shared_memory


class State(NamedTuple):
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    tau: np.ndarray

@dataclass
class Trajectory:
    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    tau: np.ndarray

    def __init__(self):
        self.time = np.empty((0,))
        self.qpos = np.empty((0,7))
        self.qvel = np.empty((0,7))
        self.tau = np.empty((0,7))

    def append(self, state: State):
        self.time = np.concat([self.time, [state.time]], axis=0)
        self.qpos = np.concat([self.qpos, [state.qpos]], axis=0)
        self.qvel = np.concat([self.qvel, [state.qvel]], axis=0)
        self.tau  = np.concat([self.tau , [state.tau ]], axis=0)

    def save(self, path):
        np.savez(path, **self.__dict__)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Trajectory(**{k: data[k] for k in data.files})

    def plot(self):
        fig = plt.figure(figsize=(18,5))
        axes = fig.subplots(1, 3)
        ts = np.arange(self.time.shape[0])

        # Panel 1: Torque Magnitudes
        # bias_mag = np.linalg.norm(self.bias, axis=1)
        tau_mag = np.linalg.norm(self.tau, axis=1)
        # diff_mag = np.linalg.norm(self.ctrl - self.bias, axis=1)

        # axes[0].plot(ts, bias_mag, label="‖bias‖")
        axes[0].plot(ts, tau_mag, label="‖tau‖")
        # axes[0].plot(ts, diff_mag, label="‖tau - bias‖")
        axes[0].set_title("Torque Magnitudes")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Torque Norm [Nm]")
        axes[0].legend()

        # Panel 2: Joint Positions
        for j in range(7):
            axes[1].plot(ts, self.qpos[:, j], label=f"q{j+1}")
        axes[1].set_title("Joint Positions")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Position [rad]")
        axes[1].legend(loc="upper right", fontsize=8)

        # Panel 3: Joint Velocities
        for j in range(7):
            axes[2].plot(ts, self.qvel[:, j], label=f"q̇{j+1}")
        axes[2].set_title("Joint Velocities")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Velocity [rad/s]")
        axes[2].legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        
        return fig

class SharedMemoryNumpyArray:
    """Helper class to store a numpy array in shared memory."""

    def __init__(self, arr: np.ndarray, ctx: mp.context.BaseContext):
        """Create a shared memory numpy array.

        Args:
            arr: The numpy array to store in shared memory. Size and dtype must
                 be fixed.
            ctx: The multiprocessing context to use for shared memory.
        """
        self.shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shared_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self.shm.buf)
        shared_arr[:] = arr[:]
        self.shape = arr.shape
        self.dtype = arr.dtype
        self.lock = ctx.Lock()

    def __getitem__(self, key: int) -> np.ndarray:
        """Get an item from the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return np.copy(arr[key])  # Need to copy here to avoid segfaults

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        """Set an item in the shared array."""
        with self.lock:
            shm = shared_memory.SharedMemory(name=self.shm.name)
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
            arr[key] = value

    def __str__(self) -> str:
        """Return the string representation of the shared array."""
        shm = shared_memory.SharedMemory(name=self.shm.name)
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return str(arr)

    def __del__(self) -> None:
        """Clean up the shared memory on deletion."""
        self.shm.close()
        self.shm.unlink()

class SharedMemorySimulation:
    """Helper class for passing mujoco data between concurrent processes."""

    def __init__(self, mj_data: mujoco.MjData, ctx: mp.context.BaseContext):
        """Create shared memory objects for state and control data.

        Note that this does not copy the full mj_data object, only those fields
        that we want to share between the simulator and controller.

        Args:
            mj_data: The mujoco data object to store in shared memory.
            ctx: The multiprocessing context to use.
        """
        # N.B. we use float32 to match JAX's default precision
        self.qpos = SharedMemoryNumpyArray(
            np.array(mj_data.qpos, dtype=np.float32), ctx
        )
        self.qvel = SharedMemoryNumpyArray(
            np.array(mj_data.qvel, dtype=np.float32), ctx
        )
        self.ctrl = SharedMemoryNumpyArray(
            np.zeros(mj_data.ctrl.shape, dtype=np.float32), ctx
        )

class FR3Simulation:
    def __init__(self, xml_path: str, dt: float = 0.001):
        # Load MuJoCo model + data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        # Control dimensions
        self.nu = self.model.nu
        self.torque_buffer = np.zeros(self.nu)

        # Runtime state
        self.error_message = None

        self.ctx = mp.get_context("spawn")  # Need to use spawn for jax compatibility
        # self.shm_data = SharedMemorySimulation(self.data, self.ctx)
        self.ready = self.ctx.Event()
        self.finished = self.ctx.Event()
        self._start_process()

    # -------------------------
    # Public API (matches C++)
    # -------------------------

    def push(self, torque) -> bool:
        """Non-blocking, best-effort write."""
        try:
            tau = np.asarray(torque, dtype=float)
            if tau.shape != self.torque_buffer.shape:
                return False
            self.torque_buffer[:] = tau
            return True
        except Exception:
            return False

    def stop(self):
        was_running = not self.finished.is_set()
        self.finished.set()
        self.data = None
        print("was running?")
        print(was_running)
        if was_running and self._process is not None:
            self._process.terminate()

    def last_error(self) -> str:
        return self.error_message or ""

    # -------------------------
    # libfranka-style callbacks
    # -------------------------

    def _read(self, cb) -> bool:
        """
        Read-only loop.
        cb(state) -> bool
        """
        try:
            while self.running:
                state = self._read_state()
                cont = cb(state)
                if not cont:
                    return False
                time.sleep(self.dt)
            return False

        except Exception as e:
            self.error_message = str(e)
            return False

    def _control(self, cb) -> np.ndarray:
        """
        Compute control torques via callback.
        cb(state, dt) -> tau
        """
        state = self._read_state()
        tau = cb(state, self.dt)

        tau = np.asarray(tau, dtype=float)
        if tau.shape != self.torque_buffer.shape:
            raise ValueError("Bad torque shape")

        return tau

    # -------------------------
    # Control loop
    # -------------------------

    def _control_loop(self):
        try:
            self.running = True
            iter_count = 0

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running() and self.running:
                    # Apply last commanded torque
                    bias = np.zeros(self.model.nv)
                    mujoco.mj_rne(self.model, self.data, 0, bias)
                    self.data.ctrl = bias # + self.torque_buffer
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()

                    # Throttled debug (~1 Hz)
                    if iter_count % int(1 / self.dt) == 0:
                        print(
                            "[SIM READ] q[0..2] =",
                            self.data.qpos[:3],
                        )

                    iter_count += 1
                    time.sleep(self.dt)

        except Exception as e:
            self.error_message = str(e)

        finally:
            self.finished.set()
            self.data = None

    # -------------------------
    # Helpers
    # -------------------------

    def read(self) -> State:
        return State(
            time.time(),
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy()
        )

    def _start_process(self):
        try:
            self._process = self.ctx.Process(
                target=self._control_loop
            )
            self._process.start()
        except Exception as e:
            # Constructor must not throw — mirror libfranka
            self.error_message = str(e)
            self.finished.set()