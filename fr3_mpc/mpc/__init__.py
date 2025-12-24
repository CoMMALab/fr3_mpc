"""
MPC planning submodule.

This submodule contains JAX-dependent code for model predictive control,
including task definitions and kinematic utilities. Import from this module
only when you need planning/optimization capabilities.

Fast imports (no JAX):
    >>> from fr3_mpc import HOME, DEFAULT_MODEL
    >>> from fr3_mpc.simulation import FR3Simulation

Heavy imports (loads JAX/hydrax):
    >>> from fr3_mpc.mpc import ReachPose, ReachJoints
    >>> from fr3_mpc.mpc import fk_point, jac_spatial
"""

from .reach_pose import ReachPose
from .reach_joints import ReachJoints

from .jacobian import (
    expand_qpos,
    fk_point,
    fk_rotmat,
    jac_point,
    jac_angular,
    jac_spatial,
)

__all__ = [
    "ReachPose",
    "ReachJoints",
    "expand_qpos",
    "fk_point",
    "fk_rotmat",
    "jac_point",
    "jac_angular",
    "jac_spatial",
]
