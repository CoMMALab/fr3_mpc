import time
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import multiprocessing as mp

from typing import NamedTuple, Optional
from dataclasses import dataclass

from . import HOME
from .shared_memory import SharedMemoryNumpyArray


class State(NamedTuple):
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    tau: np.ndarray

    @staticmethod
    def from_data(data):
        return State(
            data.time,  # Use MuJoCo simulation time
            data.qpos.copy(),
            data.qvel.copy(),
            data.ctrl.copy()
        )

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

    def append(self, state):
        """Append a state to the trajectory.

        Handles both Python State (qpos/qvel) and C++ State (q/dq).
        """
        # Handle both Python simulation State and C++ FR3Robot State
        qpos = getattr(state, 'qpos', getattr(state, 'q', None))
        qvel = getattr(state, 'qvel', getattr(state, 'dq', None))

        if qpos is None or qvel is None:
            raise ValueError(f"State must have qpos/qvel or q/dq attributes")

        self.time = np.concat([self.time, [state.time]], axis=0)
        self.qpos = np.concat([self.qpos, [np.array(qpos)[:7]]], axis=0)
        self.qvel = np.concat([self.qvel, [np.array(qvel)[:7]]], axis=0)
        self.tau  = np.concat([self.tau , [np.array(state.tau)[:7]]], axis=0)

    def save(self, path):
        np.savez(path, **self.__dict__)

    @staticmethod
    def load(path):
        data = np.load(path)
        return Trajectory(**{k: data[k] for k in data.files})

    def plot(self):
        fig = plt.figure()
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
        axes[1].legend(loc="upper right")

        # Panel 3: Joint Velocities
        for j in range(7):
            axes[2].plot(ts, self.qvel[:, j], label=f"q̇{j+1}")
        axes[2].set_title("Joint Velocities")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Velocity [rad/s]")
        axes[2].legend(loc="upper right")

        fig.tight_layout()
        
        return fig

class SharedMemoryTorqueBuffer:
    """
    Shared-memory SPSC ring buffer for 7-dim torque vectors.

    Semantics match the C++ TorqueBuffer<N>.
    """

    def __init__(self, N: int, ctx: mp.context.BaseContext):
        assert (N & (N - 1)) == 0, "N must be power of two"
        self.N = N
        self.mask = N - 1

        # data buffer: shape (N, 7)
        self.data = SharedMemoryNumpyArray(
            np.zeros((N, 7), dtype=np.float64), ctx
        )

        # head / tail counters
        self.head = ctx.Value("Q", 0)  # uint64_t
        self.tail = ctx.Value("Q", 0)

        # locks approximate acquire/release ordering
        self.head_lock = ctx.Lock()
        self.tail_lock = ctx.Lock()

    def try_write(self, v: np.ndarray) -> bool:
        """
        Attempt to write a torque vector.
        Returns True on success, False if buffer full.
        """
        assert v.shape == (7,)

        with self.head_lock:
            h = self.head.value
            t = self.tail.value

            if h - t < self.N:
                self.data[h & self.mask] = v
                self.head.value = h + 1
                return True

        return False

    def try_read(self) -> Optional[np.ndarray]:
        """
        Attempt to read a torque vector (FIFO order).
        Returns vector on success, None if buffer empty.
        """
        with self.tail_lock:
            t = self.tail.value
            h = self.head.value

            if h > t:
                v = self.data[t & self.mask].copy()
                self.tail.value = t + 1
                return v

        return None

class SharedMemoryState:
    """Shared-memory container for a single State.

    Exposes read() / write() instead of indexing.
    """

    def __init__(
        self,
        state: State,
        ctx: mp.context.BaseContext,
    ):
        # store time as length-1 array for uniformity
        self.time = SharedMemoryNumpyArray(
            np.array([state.time], dtype=np.float64), ctx
        )
        self.qpos = SharedMemoryNumpyArray(
            np.asarray(state.qpos, dtype=np.float32), ctx
        )
        self.qvel = SharedMemoryNumpyArray(
            np.asarray(state.qvel, dtype=np.float32), ctx
        )
        self.tau = SharedMemoryNumpyArray(
            np.asarray(state.tau, dtype=np.float32), ctx
        )

        # single lock for atomic State semantics
        self.lock = ctx.Lock()

    def read(self) -> State:
        """Atomically read the full State."""
        with self.lock:
            return State(
                time=float(self.time[0]),
                qpos=self.qpos[:],
                qvel=self.qvel[:],
                tau=self.tau[:],
            )

    def write(self, state: State) -> None:
        """Atomically write the full State."""
        with self.lock:
            self.time[0] = state.time
            self.qpos[:] = state.qpos
            self.qvel[:] = state.qvel
            self.tau[:] = state.tau

    def __str__(self) -> str:
        return str(self.read())


class FR3Simulation:
    def __init__(self, xml_path: str, dt: float = 0.001, qpos0=None):
        # Load MuJoCo model + data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        # Set initial joint configuration (if provided)
        if qpos0 is not None:
            self.data.qpos[:len(qpos0)] = qpos0
            self.data.qvel[:] = 0  # Start from rest
            mujoco.mj_forward(self.model, self.data)  # Update derived quantities

        # Control dimensions
        self.nu = self.model.nu

        # Runtime state
        self.error_message = None

        self.ctx = mp.get_context("spawn")  # Need to use spawn for jax compatibility
        self.shm_state = SharedMemoryState(
            State.from_data(self.data),
            self.ctx
        )
        # Shared memory torque buffer (SPSC ring buffer)
        self.torque_buffer = SharedMemoryTorqueBuffer(N=1024, ctx=self.ctx)

        self.ready = self.ctx.Event()
        self.finished = self.ctx.Event()
        self._start_process()

    # -------------------------
    # Public API (matches C++)
    # -------------------------

    def wait_until_ready(self, timeout: float = 10.0) -> bool:
        """Wait until the control loop is ready to accept commands.

        Returns:
            True if ready, False if timeout occurred
        """
        return self.ready.wait(timeout)

    def push(self, torque) -> bool:
        """Non-blocking, best-effort write."""
        try:
            tau = np.asarray(torque, dtype=float)
            if tau.shape != (self.nu,):
                return False
            return self.torque_buffer.try_write(tau)
        except Exception:
            return False

    def stop(self):
        was_running = not self.finished.is_set()
        self.finished.set()
        self.data = None
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

            # Signal that we're ready to accept commands
            self.ready.set()

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # Track start time for deterministic timing
                t0 = time.time()

                while viewer.is_running() and self.running:
                    # Compute bias forces (gravity + coriolis compensation)
                    bias = np.zeros(self.model.nv)
                    mujoco.mj_rne(self.model, self.data, 0, bias)

                    # Read commanded torques from buffer (if any)
                    tau_cmd = self.torque_buffer.try_read()
                    if tau_cmd is not None:
                        self.data.ctrl = bias + tau_cmd
                    else:
                        self.data.ctrl = bias

                    mujoco.mj_step(self.model, self.data)
                    self.shm_state.write(State.from_data(self.data))

                    # Sync viewer at reduced rate to avoid slowing down simulation
                    if iter_count % 10 == 0:  # 100 Hz viewer update
                        viewer.sync()

                    # Throttled debug (~1 Hz)
                    if iter_count % int(1 / self.dt) == 0:
                        print(
                            "[SIM READ] q[0..2] =",
                            self.data.qpos[:3],
                        )

                    iter_count += 1

                    # Deterministic timing: sleep until next timestep
                    target_time = t0 + iter_count * self.dt
                    sleep_time = target_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            self.error_message = str(e)

        finally:
            self.finished.set()
            self.data = None

    # -------------------------
    # Helpers
    # -------------------------

    def read(self) -> State:
        return self.shm_state.read()

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