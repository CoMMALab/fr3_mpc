import threading
import time
import numpy as np
import mujoco
import mujoco.viewer


class FR3Simulator:
    def __init__(self, xml_path: str, dt: float = 0.001):
        # Load MuJoCo model + data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        # Control dimensions
        self.nu = self.model.nu
        self.torque_buffer = np.zeros(self.nu)

        # Runtime state
        self.running = False
        self.error_message = None

        self._thread = None

        # Auto-start to mirror real robot semantics
        self._start_thread()

    # -------------------------
    # Public API (matches C++)
    # -------------------------

    def send_torque(self, torque) -> bool:
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
        was_running = self.running
        self.running = False
        if was_running and self._thread is not None:
            self._thread.join()

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
    # Control loop (authoritative)
    # -------------------------

    def _control_loop(self):
        try:
            self.running = True
            iter_count = 0

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while viewer.is_running() and self.running:
                    # Apply last commanded torque
                    self.data.ctrl[:] = self.torque_buffer

                    mujoco.mj_step(self.model, self.data)

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
            self.running = False

    # -------------------------
    # Helpers
    # -------------------------

    def _read_state(self):
        return {
            "q": self.data.qpos.copy(),
            "dq": self.data.qvel.copy(),
            "tau": self.torque_buffer.copy(),
        }

    def _start_thread(self):
        try:
            self._thread = threading.Thread(
                target=self._control_loop,
                daemon=True,
            )
            self._thread.start()
        except Exception as e:
            # Constructor must not throw — mirror libfranka
            self.error_message = str(e)
            self.running = False
