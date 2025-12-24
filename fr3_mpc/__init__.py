"""
fr3_mpc: Model Predictive Control for Franka Research 3.

Fast imports (< 50ms, no JAX dependencies):
    Constants: ROOT, DEFAULT_MODEL, HOME, ALT_HOME, HOME_POS, HOME_ROT
    Simulation: simulation.FR3Simulation, simulation.Trajectory, simulation.State
    Real robot: control.FR3Controller (C++ module)

Heavy imports (loads JAX/hydrax, use only when needed):
    MPC planning: mpc.ReachPose, mpc.ReachJoints
    Kinematics: mpc.fk_point, mpc.jac_spatial, etc.

Example:
    >>> import fr3_mpc  # Fast!
    >>> from fr3_mpc.simulation import FR3Simulation
    >>> robot = FR3Simulation("models/franka_fr3_v2/fr3v2.xml")

    >>> # Only import when doing MPC:
    >>> from fr3_mpc.mpc import ReachJoints
    >>> task = ReachJoints(target_q=fr3_mpc.HOME)
"""

import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.path.join(ROOT, "models", "franka_emika_panda", "mjx_panda.xml")

HOME = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
ALT_HOME = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, -np.pi/4])

HOME_POS = np.array([ 0.307, -0., 0.49])
HOME_ROT = np.diag(np.array([1., -1., -1.]))

__all__ = [
    "ROOT",
    "DEFAULT_MODEL",
    "HOME",
    "ALT_HOME",
    "HOME_POS",
    "HOME_ROT",
]
