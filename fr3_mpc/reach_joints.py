import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task

EE_SITE = "gripper"
TARGET_BODY = "target_body"


# data = mujoco.MjData(model)
# data.qpos[:7] = q
# mujoco.mj_forward(model, data)

# sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
# pos = data.site_xpos[sid].copy()
# rot = data.site_xmat[sid].reshape(3, 3).copy()
# return pos, rot

class ReachJoints(Task):
    """A reaching task for 7D joints in configuration space."""

    def __init__(self, target_q: jax.Array) -> None:
        """
        Args:
            target_q: Desired 7D joint configuration (q_target).
        """
        mj_model = mujoco.MjModel.from_xml_path("./models/panda_mjx.xml")

        # Store desired joint configuration
        self.target_q = target_q

        # (Optional) keep end-effector site for logging/trace
        self.ee_sid = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE
        )
        self.target_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)

        self.arm_slice = slice(0, 7)

        # For visual marker
        _tmp_data = mujoco.MjData(mj_model)
        _tmp_data.qpos[:7] = target_q
        mujoco.mj_forward(mj_model, _tmp_data)
        pos = _tmp_data.site_xpos[self.ee_sid].copy()
        # rot = _tmp_data.site_xmat[self.ee_sid].reshape(3, 3).copy()
        mj_model.body_pos[self.target_bid] = pos
        del _tmp_data

        super().__init__(mj_model, trace_sites=[EE_SITE])

    def _config_space_cost(self, state: mjx.Data) -> jax.Array:
        """Quadratic cost in configuration space (joint space).

        Uses the first 7 qpos entries as the arm joints.
        """
        # Current joint configuration
        q = state.qpos[self.arm_slice]      # (7,)

        # Joint error
        dq = q - self.target_q              # (7,)

        # ||dq||^2
        return jnp.dot(dq, dq)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Quadratic in joint error and control effort:
          ℓ = ||q - q_target||² + 1e-3 * ||u||²
        """
        pose_cost = self._config_space_cost(state)
        control_cost = jnp.dot(control, control)  # ||u||²

        return pose_cost + 1e-3 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Stronger emphasis on being at the target joint configuration:
          ϕ = 10 * ||q - q_target||²
        """
        pose_cost = self._config_space_cost(state)
        return 10.0 * pose_cost
