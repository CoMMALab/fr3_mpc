import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax.task_base import Task

EE_SITE = "gripper"
TARGET_BODY = "target_body"


class ReachPose(Task):
    """A reaching task for poses in SE(3)."""

    def __init__(self, target_pos: jax.Array, target_rot: jax.Array) -> None:
        mj_model = mujoco.MjModel.from_xml_path("./models/panda_mjx.xml")

        self.target_pos = target_pos
        self.target_rot = target_rot

        self.ee_sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        self.target_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)

        # set initial position of the mocap marker
        mj_model.body_pos[self.target_bid] = target_pos

        super().__init__(mj_model, trace_sites=["gripper"])

    def _frobenius_norm(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the target pose.

        Combines:
          - squared Euclidean distance in position
          - squared Frobenius norm of rotation matrix difference
        """
        # Current end-effector pose
        pos = state.site_xpos[self.ee_sid]               # (3,)
        rot_flat = state.site_xmat[self.ee_sid]          # (9,)
        rot = rot_flat.reshape(3, 3)                     # (3, 3)

        # Position error
        pos_err = pos - self.target_pos                  # (3,)
        pos_cost = jnp.dot(pos_err, pos_err)             # ||Δp||²

        # Orientation error (matrix space)
        rot_err = rot - self.target_rot                  # (3, 3)
        rot_cost = jnp.sum(jnp.square(rot_err))          # ||ΔR||_F²

        # Weight orientation a bit less than position
        return pos_cost + 0.1 * rot_cost

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Quadratic in pose error and control effort:
          ℓ = pose_cost + 1e-3 * ||u||²
        """
        pose_cost = self._frobenius_norm(state)
        control_cost = jnp.dot(control, control)         # ||u||²

        return pose_cost + 1e-3 * control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Stronger emphasis on being at the target pose at the final time:
          ϕ = 10 * pose_cost
        """
        pose_cost = self._frobenius_norm(state)
        return 10.0 * pose_cost